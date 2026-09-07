"""Generate one self-contained, read-only HTML evidence explorer."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from src.analysis.compare import ComparisonReport
from src.analysis.metrics import EpisodeMetrics
from src.analysis.regression_gate import GateResult
from src.contracts.agent_episode import AgentEpisode


def render_observatory_html(
    *,
    episodes: Iterable[AgentEpisode],
    metrics: Iterable[EpisodeMetrics],
    diagnoses: Iterable[Mapping[str, Any]],
    comparison: ComparisonReport,
    gate: GateResult,
) -> str:
    payload = {
        "episodes": [episode.to_dict() for episode in episodes],
        "metrics": [metric.to_dict() for metric in metrics],
        "diagnoses": list(diagnoses),
        "comparison": comparison.to_dict(),
        "gate": gate.to_dict(),
    }
    # Script data is a raw-text element: HTML entities would not be decoded.
    # Escape only a closing tag sequence so embedded evidence cannot terminate
    # the application/json block.
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Harness Observatory V2</title>
<style>
:root{{--bg:#0b1020;--panel:#131a2e;--line:#2b3553;--text:#edf2ff;--muted:#9eaccd;--good:#55d6a2;--bad:#ff7b8b;--accent:#7aa2ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}}
main{{max-width:1400px;margin:auto;padding:28px}}h1,h2{{font-family:system-ui,sans-serif}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.muted{{color:var(--muted)}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}button{{background:#1b2542;color:var(--text);border:1px solid var(--line);padding:7px 10px;border-radius:7px;cursor:pointer}}
pre{{white-space:pre-wrap;word-break:break-word;max-height:520px;overflow:auto}}select{{background:var(--panel);color:var(--text);padding:8px;border:1px solid var(--line)}}
</style></head><body><main>
<h1>Harness Observatory <span class=\"muted\">V2 / read-only evidence</span></h1>
<div id=\"summary\" class=\"grid\"></div>
<h2>Harness Compare</h2><div class=\"card\"><table><thead><tr><th>Pair</th><th>Outcome</th><th>Control</th><th>Candidate</th><th>Δ metrics</th></tr></thead><tbody id=\"pairs\"></tbody></table></div>
<h2>Regression Gate</h2><div class=\"card\"><table><thead><tr><th>Rule</th><th>Actual</th><th>Threshold</th><th>Result</th><th>Evidence</th></tr></thead><tbody id=\"checks\"></tbody></table></div>
<h2>Episode Explorer</h2><div class=\"card\"><select id=\"episodeSelect\"></select><pre id=\"episode\"></pre></div>
<details><summary>Embedded canonical evidence</summary><script id=\"evidence\" type=\"application/json\">{encoded}</script></details>
<script>
const data=JSON.parse(document.getElementById('evidence').textContent);
const a=data.comparison.aggregate,g=data.gate;
document.getElementById('summary').innerHTML=[['Gate',g.decision],['Comparable',data.comparison.comparable],['Paired coverage',data.comparison.paired_coverage],['Success Δ',a.success_rate_delta],['Infra invalid Δ',a.infra_invalid_rate_delta],['Target slice',a.control_target_slice_count+' → '+a.candidate_target_slice_count]].map(([k,v])=>`<div class=\"card\"><div class=\"muted\">${{k}}</div><strong>${{v}}</strong></div>`).join('');
document.getElementById('pairs').innerHTML=data.comparison.pairs.map(p=>`<tr><td>${{p.pair_key}}</td><td>${{p.outcome_transition}}</td><td>${{p.control_episode_id}}<br>${{p.control_reason_codes.join(', ')}}</td><td>${{p.candidate_episode_id}}<br>${{p.candidate_reason_codes.join(', ')}}</td><td><pre>${{JSON.stringify(p.metric_deltas,null,2)}}</pre></td></tr>`).join('');
document.getElementById('checks').innerHTML=g.checks.map(c=>`<tr><td>${{c.rule}}</td><td>${{JSON.stringify(c.actual)}}</td><td>${{JSON.stringify(c.threshold)}}</td><td class=\"${{c.passed===true?'good':c.passed===false?'bad':'muted'}}\">${{c.passed}}</td><td>${{c.evidence.join('<br>')}}</td></tr>`).join('');
const sel=document.getElementById('episodeSelect');sel.innerHTML=data.episodes.map(e=>`<option value=\"${{e.episode_id}}\">${{e.episode_id}} · ${{e.outcome.task_status}} · ${{e.integrity.state}}</option>`).join('');
function show(){{const e=data.episodes.find(x=>x.episode_id===sel.value);document.getElementById('episode').textContent=JSON.stringify(e,null,2)}}sel.onchange=show;show();
</script></main></body></html>"""
