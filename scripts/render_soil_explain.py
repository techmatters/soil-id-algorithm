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


# Human-friendly labels for the internal feature/column names.
FEATURE_LABELS = {
    "sandpct_intpl": "sand %",
    "claypct_intpl": "clay %",
    "rfv_intpl": "rock fragments %",
    "l": "color L*",
    "a": "color a*",
    "b": "color b*",
    "slope_r": "slope %",
    "elev_r": "elevation (m)",
    "bottom_depth": "depth to bedrock (cm)",
}


def label(name):
    return FEATURE_LABELS.get(name, name)


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
    # The decay multiplier is only shown when the path exposes its decay
    # coefficient (global). For US the coefficient is data-source dependent and
    # baked into cond_prob upstream, so show distance/share as context only.
    if c.get("decay_multiplier") is not None:
        detail = (
            f"decay = max(e<sup>{c.get('exp_coeff')} × {fmt(c.get('distance_m'))}m</sup>, 0.25) "
            f"= {fmt(c.get('decay_multiplier'))} · share {fmt(c.get('share_pct'))}% "
            f"&rarr; location score = decay × share = <b>{fmt(c.get('location_score'))}</b><br>"
            f"cond_prob = location score ÷ (sum over all candidates) "
            f"&rarr; <b>{fmt(c.get('score'))}</b> "
            f"<span class='hint'>(all candidates' cond_prob sum to 1)</span>"
        )
    else:
        detail = (
            f"distance {fmt(c.get('distance_m'))} m · share {fmt(c.get('share_pct'))}% "
            f"&rarr; cond_prob = distance-decayed share, normalized across all "
            f"candidates = <b>{fmt(c.get('score'))}</b> "
            f"<span class='hint'>(all candidates' cond_prob sum to 1)</span>"
        )
    return (
        f"<div class='comp'><div class='ctitle'>Location <b>{fmt(c.get('score'))}</b></div>"
        f"<div class='formula'>{detail}</div></div>"
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
    # each feature is a color-grouped pair of sub-columns (you / candidate)
    head = "".join(
        f"<th colspan='2' class='g{i % 2} gstart'>{esc(label(n))}</th>"
        for i, n in enumerate(feat_names)
    )
    sub = "".join(
        f"<th class='g{i % 2} gstart'>you</th><th class='g{i % 2}'>candidate</th>"
        for i, _ in enumerate(feat_names)
    )
    rows = []
    for s in segs:
        fmap = {f["name"]: f for f in s["features"]}
        cells = []
        for i, n in enumerate(feat_names):
            f = fmap.get(n)
            g = f"g{i % 2}"
            if not f or f.get("user") is None and f.get("candidate") is None:
                cells.append(f"<td class='{g} gstart na'>—</td><td class='{g} na'>—</td>")
                continue
            one = "" if f["status"] == "both" else " one"
            nd = "" if f["norm_diff"] is None else f" <small>Δ{f['norm_diff']:.3f}</small>"
            cells.append(
                f"<td class='{g} gstart'>{fmt(f['user'])}</td>"
                f"<td class='{g}{one}'>{fmt(f['candidate'])}{nd}</td>"
            )
        rowcls = "" if s.get("compared", True) else " class='nocompare'"
        rows.append(
            f"<tr{rowcls}><td class='depth'>{s['top']}–{s['bottom']}cm</td>"
            f"<td class='w'>×{s['depth_weight']}</td>{''.join(cells)}"
            f"<td class='dist distcol'>{fmt(s.get('slice_distance'))}</td></tr>"
        )
    compared = [s for s in segs if s.get("compared", True)]
    win = (
        f"{min(s['top'] for s in compared)}–{max(s['bottom'] for s in compared)} cm"
        if compared
        else "none"
    )
    # Weighted combine over depths: weight = wt × band thickness (cm). This lands on
    # the horizon score, making the role of wt explicit.
    tw = tc = 0.0
    for s in compared:
        d = s.get("slice_distance")
        if d is None:
            continue
        w = s["depth_weight"] * (s["bottom"] - s["top"])
        tw += w
        tc += w * d
    hz_dist = tc / tw if tw else None
    combine = (
        f"<div class='note'>→ horizon distance = Σ(wt × cm × slice&nbsp;dist) ÷ "
        f"Σ(wt × cm) = <b>{fmt(hz_dist)}</b>; horizon score = 1 − {fmt(hz_dist)} = "
        f"<b>{fmt(c.get('score'))}</b></div>"
        if hz_dist is not None
        else ""
    )
    return (
        f"<div class='comp'><div class='ctitle'>Horizon (properties) "
        f"<b>{fmt(c.get('score'))}</b></div>"
        f"<div class='note'>Full depth range shown, 0 to max(your pit, this soil). "
        f"The algorithm only compares your recorded depths ({win}); "
        f"<span class='nocompare' style='padding:0 4px'>greyed rows</span> are outside "
        f"that window (not compared). <b>you</b>/<b>candidate</b> = the values at that "
        f"depth (— = none there); Δ = normalized difference; <b>slice dist</b> = the "
        f"<i>average of the Δ's</i> in the band (wt is <i>not</i> applied here — it's "
        f"applied when combining bands, below).</div>"
        f"<table class='hz'><tr><th rowspan='2'>depth</th><th rowspan='2'>wt</th>{head}"
        f"<th rowspan='2' class='distcol'>slice&nbsp;dist</th></tr>"
        f"<tr>{sub}</tr>{''.join(rows)}</table>{combine}</div>"
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
        f"<tr><td>{esc(label(f['name']))}</td><td>{fmt(f['user'])}</td>"
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
table.hz{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums;margin-top:4px}
table.hz th,table.hz td{border:1px solid #e6e6ee;padding:2px 7px;text-align:right}
table.hz th{background:#eef;text-align:center} /* center all header labels */
td.depth{text-align:left;font-weight:600} td.w{color:#999;text-align:center}
td.dist{font-weight:600;background:#fafaff} .na,td.na{color:#bbb}
/* per-feature column groups: alternating tint + a heavier divider at each group start */
.g0{background:#eef5ff} .g1{background:#eefaf0}
th.g0{background:#dbe8ff} th.g1{background:#daf3e1}
.gstart{border-left:2px solid #9ab!important}
.distcol{border-left:2px solid #9ab!important} /* separate slice dist */
td.one{background:#fff3e0!important} /* one-sided value */
tr.nocompare td{background:#f4f4f6;color:#a8a8a8} tr.nocompare td.depth{color:#888}
td.deriv{color:#367;font-style:italic} th.deriv{background:#e3edf6;color:#356}
.hint{color:#999;font-size:11px}
.note{color:#666;font-size:11.5px;margin:2px 0;max-width:760px}
.legend{color:#444;font-size:12px;background:#fffbe9;border:1px solid #eeddaa;
  border-radius:6px;padding:7px 10px;margin:10px 0;max-width:900px}
.pit{margin-bottom:12px}
.override{background:#fde;border:1px solid #eab;border-radius:6px;padding:4px 8px;margin:6px 0;color:#933}
small{color:#999}
"""


def render_html(trace):
    site = trace.get("site", {})
    inp = trace.get("inputs", {})
    hz = inp.get("horizons", [])

    def rfv(v):
        return "—" if v is None else esc(v)

    hrows = "".join(
        f"<tr><td class='depth'>{h.get('top')}–{h.get('bottom')}cm</td>"
        f"<td>{esc(h.get('texture'))}</td>"
        f"<td class='deriv'>{fmt(h.get('sand'))}</td><td class='deriv'>{fmt(h.get('clay'))}</td>"
        f"<td>{rfv(h.get('rfv'))}</td><td class='deriv'>{fmt(h.get('rfv_pct'))}</td></tr>"
        for h in hz
    )
    pit = (
        f"<table class='hz'><tr><th rowspan=2>depth</th>"
        f"<th>texture</th><th class='deriv' colspan=2>→ derived</th>"
        f"<th>rock frag</th><th class='deriv'>→ derived</th></tr>"
        f"<tr><th>class</th><th class='deriv'>sand %</th><th class='deriv'>clay %</th>"
        f"<th>class</th><th class='deriv'>rfv %</th></tr>{hrows}</table>"
    )
    legend = (
        "<div class='legend'>You enter a <b>texture class</b> (e.g. Clay) and a "
        "<b>rock-fragment class</b>; the algorithm converts them to representative "
        "<b>sand %</b>, <b>clay %</b>, and <b>rock-fragment %</b> (shown as → above), "
        "and the horizon tables below compare on <i>those</i> numbers.<br>"
        "Each candidate's <b>combined score</b> ≈ (properties + location) ÷ (weight + 1): "
        "<b>Location</b> = how likely the soil is mapped here (distance-decayed share, "
        "normalized); <b>Properties</b> = how well its profile matches yours (Gower "
        "distance per depth band) plus color. Rule <b>overrides</b> can force a "
        "promote/demote.</div>"
    )
    cards = "".join(render_candidate(c) for c in trace.get("candidates", []))
    return (
        f"<!doctype html><meta charset=utf-8><style>{CSS}</style>"
        f"<h1>Soil ID explanation — {esc(trace.get('region'))}</h1>"
        f"<div class='site'>lat {site.get('lat')}, lon {site.get('lon')}</div>"
        f"<div class='pit'><div class='ctitle'>Your pit "
        f"(deepest recorded {inp.get('effective_bedrock_cm')} cm)</div>{pit}</div>"
        f"{legend}{cards}"
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
