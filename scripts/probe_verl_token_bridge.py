"""CPU-only check with the frozen tokenizer and installed verl Hermes parser.

Generation is scripted and explicitly NOT GPU acceptance evidence.
"""
import argparse
import asyncio
import json
from types import SimpleNamespace

from transformers import AutoTokenizer
from verl.experimental.agent_loop.tool_parser import ToolParser
from src.integrations.verl.native_transport import NativeTokenTransport
from src.integrations.verl.bridge import BridgeCallRecord, build_per_call_segments
from src.integrations.verl.sequence import assemble_episode_sequence


async def probe(model):
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    observed = []

    class ScriptedServer:
        async def generate(self, **kwargs):
            observed.append(kwargs["prompt_ids"])
            text = ('<tool_call>{"name":"read","arguments":{"path":"task"}}</tool_call>'
                    if len(observed) == 1 else "Finished.") + "<|im_end|>"
            ids = tokenizer.encode(text, add_special_tokens=False)
            return SimpleNamespace(token_ids=ids, log_probs=[-0.1] * len(ids),
                                   extra_fields={"global_steps": 0})

    transport = NativeTokenTransport(
        loop=asyncio.get_running_loop(), server_manager=ScriptedServer(), tokenizer=tokenizer,
        parser=ToolParser.get_tool_parser("hermes", tokenizer), episode_id="cpu-probe",
        policy_revision="SCRIPTED_NOT_ACCEPTANCE", sampling_params={"temperature": 1.0},
        max_prompt=4096, max_response=4096, max_tokens=1024, max_requests=8, timeout=30,
    )
    tools = [{"type": "function", "function": {"name": "read", "description": "read file",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    messages = [{"role": "user", "content": "Read task, then explain the result."}]
    first = await transport.generate({"model": "probe", "messages": messages, "tools": tools})
    assistant = first["choices"][0]["message"]
    messages += [assistant, {"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"],
                             "content": "Implement addition."}]
    second = await transport.generate({"model": "probe", "messages": messages, "tools": tools})
    records = [BridgeCallRecord(request_id=str(i),
                prompt_token_ids=tuple(r["agent_data_plane_evidence"]["prompt_token_ids"]),
                response_token_ids=tuple(r["agent_data_plane_evidence"]["response_token_ids"]),
                response_logprobs=tuple(r["agent_data_plane_evidence"]["response_logprobs"]))
               for i, r in enumerate([first, second])]
    segments, prompt = build_per_call_segments(records)
    sequence = assemble_episode_sequence(episode_id="cpu-probe", prompt_ids=prompt,
                                         per_call_segments=segments)
    assert sequence.num_tool_rounds == 1
    assert 0 in sequence.response_mask and 1 in sequence.response_mask
    print(json.dumps({"status": "PASSED", "evidence_kind": "SCRIPTED_CPU_ONLY",
                      "model_calls": 2, "prompt_tokens": len(prompt),
                      "response_tokens": len(sequence.response_ids),
                      "observation_tokens": sequence.response_mask.count(0)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    asyncio.run(probe(parser.parse_args().model))
