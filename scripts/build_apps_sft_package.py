#!/usr/bin/env python3
"""Build token-level LoRA SFT data from certified APPS episodes."""
import argparse, json, hashlib
from pathlib import Path

def canon(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--model', required=True); ap.add_argument('--max-length', type=int, default=4096)
    a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model)
    rows=[]; tasks=set(); rejected=0
    for root in a.roots:
      for q in sorted(Path(root).glob('*/finalized/sft-eligibility.json')):
        try: elig=json.loads(q.read_text())
        except Exception: continue
        if elig.get('verdict') != 'ELIGIBLE': rejected+=1; continue
        ep=q.with_name('episode.json')
        if not ep.exists(): continue
        d=json.loads(ep.read_text()); events=d.get('events',[]); last_req=None
        for e in events:
          typ=e.get('event_type'); at=e.get('attributes',{})
          if typ=='MODEL_REQUEST': last_req=at.get('messages')
          elif typ=='MODEL_RESPONSE' and last_req and not at.get('error_message'):
            content=at.get('content') or []
            prompt=''.join('['+m.get('role','')+'] '+canon(m.get('content',''))+'\n' for m in last_req)
            completion=canon(content)
            p=tok.encode(prompt, add_special_tokens=True); c=tok.encode(completion+tok.eos_token, add_special_tokens=False)
            if p and c:
              if len(p)+len(c)>a.max_length: p=p[max(0,len(p)+len(c)-a.max_length):]
              if p:
                rows.append({'task_id':d.get('task_id'),'episode_id':d.get('episode_id'),'prompt_ids':p,'completion_ids':c,'prompt_text':prompt,'completion_text':completion})
                tasks.add(d.get('task_id'))
            last_req=None
    (out/'sft_tokens.jsonl').write_text('\n'.join(canon(x) for x in rows)+'\n')
    (out/'examples.jsonl').write_text('\n'.join(canon({k:v for k,v in x.items() if not k.endswith('_ids')}) for x in rows)+'\n')
    report={'format':'apps-token-sft/v1','examples':len(rows),'tasks':len(tasks),'rejected_eligibility':rejected,'max_length':a.max_length,'model':a.model,'source_roots':a.roots}
    (out/'quality-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    manifest=dict(report); manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'package-manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
