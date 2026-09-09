#!/usr/bin/env python3
# Copyright © 2025 Technology Matters
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.

"""
Render a Soil ID explain trace (JSON, from rank_soils[_global](..., explain=Recorder()))
into a graphically rich, self-contained HTML report.

    python -m scripts.render_soil_explain trace.json > report.html

The renderer is generic over the trace's `score_components`, so the same code
handles US and global.
"""

import argparse
import html
import json


def esc(x):
    return html.escape(str(x))


def fmt(x, pct=False):
    if x is None:
        return "<span class='na'>—</span>"
    if isinstance(x, float):
        return f"{x * 100:.1f}%" if pct else f"{x:.3f}"
    return esc(x)


def bar(score, width=120):
    v = 0 if score is None else max(0.0, min(1.0, score))
    hue = int(120 * v)  # red→green
    return (
        f"<span class='bar' style='width:{width}px'>"
        f"<span style='width:{v * 100:.0f}%;background:hsl({hue},70%,45%)'></span></span>"
        f"<span class='barval'>{fmt(score)}</span>"
    )


def render_location(c):
    return (
        f"<div class='comp'><div class='ctitle'>Location <b>{fmt(c.get('score'))}</b></div>"
        f"<div class='formula'>distance {fmt(c.get('distance_m'))} m · "
        f"share {fmt(c.get('share_pct'))}% · "
        f"decay max(e<sup>{c.get('exp_coeff')}·dist</sup>, 0.25) = {fmt(c.get('decay_multiplier'))}"
        f" &rarr; cond_prob <b>{fmt(c.get('score'))}</b></div></div>"
    )


def render_horizon(c):
    segs = c.get("segments", [])
    if not segs:
        return ""
    feat_names = []
    for s in segs:
        for f in s["features"]:
            if f["name"] not in feat_names:
                feat_names.append(f["name"])
    head = "".join(f"<th colspan='2'>{esc(n)}</th>" for n in feat_names)
    sub = "".join("<th>you</th><th>candidate</th>" for _ in feat_names)
    rows = []
    for s in segs:
        fmap = {f["name"]: f for f in s["features"]}
        cells = []
        for n in feat_names:
            f = fmap.get(n)
            if not f:
                cells.append("<td class='na'>—</td><td class='na'>—</td>")
                continue
            cls = "" if f["status"] == "both" else "one"
            nd = "" if f["norm_diff"] is None else f" <small>Δ{f['norm_diff']:.3f}</small>"
            cells.append(
                f"<td>{fmt(f['user'])}</td><td class='{cls}'>{fmt(f['candidate'])}{nd}</td>"
            )
        rows.append(
            f"<tr><td class='depth'>{s['top']}–{s['bottom']}cm</td>"
            f"<td class='w'>×{s['depth_weight']}</td>{''.join(cells)}"
            f"<td class='dist'>{fmt(s.get('slice_distance'))}</td></tr>"
        )
    return (
        f"<div class='comp'><div class='ctitle'>Horizon (properties) "
        f"<b>{fmt(c.get('score'))}</b></div>"
        f"<table class='hz'><tr><th>depth</th><th>wt</th>{head}<th>slice&nbsp;dist</th></tr>"
        f"<tr><th></th><th></th>{sub}<th></th></tr>{''.join(rows)}</table></div>"
    )


def render_color(c):
    de = c.get("delta_e")
    de_s = ", ".join(f"{x:.1f}" for x in de) if de else "—"
    return (
        f"<div class='comp'><div class='ctitle'>Color <b>{fmt(c.get('score'))}</b></div>"
        f"<div class='formula'>ΔE2000 vs white/red/yellow = [{de_s}] · "
        f"weight {c.get('weight')} &rarr; similarity <b>{fmt(c.get('score'))}</b></div></div>"
    )


def render_site(c):  # US site score (slope/elev/depth)
    feats = "".join(
        f"<tr><td>{esc(f['name'])}</td><td>{fmt(f['user'])}</td>"
        f"<td class='{'' if f['status'] == 'both' else 'one'}'>{fmt(f['candidate'])}</td>"
        f"<td>{fmt(f.get('norm_diff'))}</td></tr>"
        for f in c.get("features", [])
    )
    return (
        f"<div class='comp'><div class='ctitle'>Site <b>{fmt(c.get('score'))}</b></div>"
        f"<table class='hz'><tr><th>feature</th><th>you</th><th>candidate</th><th>Δ</th></tr>"
        f"{feats}</table></div>"
    )


RENDERERS = {
    "location": render_location,
    "horizon": render_horizon,
    "color": render_color,
    "site": render_site,
}


def render_candidate(cand):
    comps = "".join(RENDERERS.get(c["type"], lambda _: "")(c) for c in cand["score_components"])
    ov = ""
    for o in cand.get("overrides", []):
        ov = (
            f"<div class='override'>⚑ override: <b>{esc(o['rule'])}</b> — "
            f"score {fmt(o.get('score_before'))} &rarr; {fmt(o.get('score_after'))}</div>"
        )
    return (
        f"<div class='card'><div class='chead'>"
        f"<span class='rank'>#{cand.get('rank')}</span>"
        f"<span class='name'>{esc(cand['name'])}</span>"
        f"<span class='combined'>combined {bar(cand.get('combined_score'))}</span>"
        f"<span class='props'>properties {fmt(cand.get('properties_score'))}</span></div>"
        f"{ov}{comps}</div>"
    )


CSS = """
body{font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:20px;color:#222;background:#f6f7f8}
h1{font-size:18px} .site{color:#666;margin-bottom:14px}
.card{background:#fff;border:1px solid #dde;border-radius:8px;margin:12px 0;padding:10px 14px;
  box-shadow:0 1px 2px rgba(0,0,0,.05)}
.chead{display:flex;align-items:center;gap:14px;border-bottom:1px solid #eee;padding-bottom:6px;margin-bottom:8px}
.rank{font-weight:700;color:#fff;background:#456;border-radius:12px;padding:1px 9px}
.name{font-weight:700;font-size:15px;flex:1} .props{color:#666}
.bar{display:inline-block;height:11px;background:#eee;border-radius:6px;vertical-align:middle;overflow:hidden}
.bar>span{display:block;height:100%} .barval{margin-left:6px;font-variant-numeric:tabular-nums}
.comp{margin:8px 0} .ctitle{font-weight:600;color:#345;margin-bottom:3px}
.formula{color:#555;font-family:ui-monospace,monospace;font-size:12px}
table.hz{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
table.hz th,table.hz td{border:1px solid #e6e6ee;padding:2px 6px;text-align:right}
table.hz th{background:#eef} td.depth{text-align:left;font-weight:600} td.w{color:#999}
td.dist{font-weight:600;background:#fafaff} td.one{background:#fff3e0} .na,td.na{color:#bbb}
.override{background:#fde;border:1px solid #eab;border-radius:6px;padding:4px 8px;margin:6px 0;color:#933}
small{color:#999}
"""


def render_html(trace):
    site = trace.get("site", {})
    inp = trace.get("inputs", {})
    hz = inp.get("horizons", [])
    hrows = "".join(
        f"<tr><td>{h.get('top')}–{h.get('bottom')}cm</td><td>{esc(h.get('texture'))}</td>"
        f"<td>rfv {esc(h.get('rfv'))}</td></tr>"
        for h in hz
    )
    cards = "".join(render_candidate(c) for c in trace.get("candidates", []))
    return (
        f"<!doctype html><meta charset=utf-8><style>{CSS}</style>"
        f"<h1>Soil ID explanation — {esc(trace.get('region'))}</h1>"
        f"<div class='site'>lat {site.get('lat')}, lon {site.get('lon')} · "
        f"your pit ({inp.get('effective_bedrock_cm')} cm): "
        f"<table class='hz' style='display:inline-table'>{hrows}</table></div>"
        f"{cards}"
    )


def main():
    ap = argparse.ArgumentParser("render_soil_explain")
    ap.add_argument("trace_json")
    args = ap.parse_args()
    with open(args.trace_json) as f:
        trace = json.load(f)
    print(render_html(trace))


if __name__ == "__main__":
    main()
