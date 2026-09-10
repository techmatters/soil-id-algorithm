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


def _locbox(title, body):
    return f"<div class='locbox'><div class='locbt'>{title}</div>{body}</div>"


def _dist(x):
    return "<span class='na'>—</span>" if x is None else f"{x:,.0f} m"


def render_location(c):
    """Location as an inputs → decayed → [combine] → normalized flow, mirroring
    process_distance_scores (Fan et al.)."""
    ds = c.get("distance_score")
    if ds is None:
        # No real intermediates available: fall back to a one-line summary.
        return (
            f"<div class='comp'><div class='ctitle blue'>Location <b>{fmt(c.get('score'))}</b></div>"
            f"<div class='formula'>distance {_dist(c.get('distance_m'))} · share "
            f"{fmt(c.get('share_pct'))}% &rarr; cond_prob (distance-decayed share, "
            f"normalized) = <b>{fmt(c.get('score'))}</b></div></div>"
        )

    total = c.get("total_distance_score")
    comp = c.get("comp_distance_score")
    # A component whose total exceeds this one instance also occurs in other map
    # units (some may be filtered before ranking, so we don't itemize them).
    multi = comp is not None and ds is not None and (comp - ds) > 0.0005

    # inputs
    inputs = (
        f"<table class='lt'><tr><td>distance</td><td>{_dist(c.get('distance_m'))}</td></tr>"
        f"<tr><td>share</td><td>{fmt(c.get('share_pct'))}%</td></tr></table>"
    )

    # decayed (distance score = share × decay). The decay factor is shown only for
    # the global path (its coefficient is known); US bakes it in upstream.
    if c.get("decay_multiplier") is not None:
        dstr = _dist(c.get("distance_m")).replace(" m", "")
        decayed = (
            f"<div class='lf'>decay = max(0.25,</div>"
            f"<div class='lf'>&nbsp; e<sup>{c.get('exp_coeff')} × {dstr}</sup>)"
            f" = {fmt(c.get('decay_multiplier'))}</div>"
            f"<div class='lf'>score = share × decay</div>"
            f"<div class='lf'>= <b>{fmt(ds)}</b></div>"
        )
    else:
        # US decay coefficient is data-source-dependent and not exposed, but the
        # effective decay = distance_score ÷ share is recoverable for display.
        sp = c.get("share_pct")
        eff = f" (decay ≈ {ds / (sp / 100):.3f})" if sp else ""
        decayed = (
            f"<div class='lf'>score = share × decay{esc(eff)}</div>"
            f"<div class='lf'>= <b>{fmt(ds)}</b></div>"
        )

    dlabel = "decayed (this map unit)" if multi else "decayed"
    boxes = [
        _locbox("inputs", inputs),
        "<div class='locarrow'>&rarr;</div>",
        _locbox(dlabel, decayed),
    ]

    # component total (only when the component spans more than this one map unit)
    if multi:
        combine = (
            f"<div class='lf'>sum over this</div><div class='lf'>component's map units</div>"
            f"<div class='lf'>= <b>{fmt(comp)}</b></div>"
        )
        boxes += ["<div class='locarrow'>&rarr;</div>", _locbox("component total", combine)]

    # normalized
    num = comp if multi else ds
    normalized = (
        f"<div class='lf'>cond_prob =</div>"
        f"<div class='lf'>{fmt(num)} ÷ {fmt(total)}</div>"
        f"<div class='lf'>= <b>{fmt(c.get('score'))}</b></div>"
    )
    boxes += ["<div class='locarrow'>&rarr;</div>", _locbox("normalized", normalized)]

    decay_note = (
        "<b>decay</b> = max(0.25, e<sup>−0.00036888 × distance_m</sup>) — an "
        "exponential fall-off with distance, floored at 0.25 (which it reaches at "
        "~3.8 km, then stays flat; the 0.00036888 rate ≈ a halving every ~1.9 km). "
        if c.get("decay_multiplier") is not None
        else ""
    )
    note = (
        f"<div class='note'>distance = nearest map-unit edge (0 if inside); "
        f"share = component %. {decay_note}A soil's <b>cond_prob</b> = its total "
        f"distance score across all the map units it occurs in ÷ the total over all "
        f"candidates (they sum to 1).</div>"
    )
    return (
        f"<div class='comp'><div class='ctitle blue'>Location <b>{fmt(c.get('score'))}</b></div>"
        f"<div class='locflow'>{''.join(boxes)}</div>{note}</div>"
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
    # Properties the user never entered anywhere: shown for context (candidate
    # values) but greyed and flagged, since they don't take part in matching.
    entered = {
        n: any(fm.get("user") is not None for s in segs for fm in s["features"] if fm["name"] == n)
        for n in feat_names
    }
    # each feature is a color-grouped pair of sub-columns (you / candidate)
    head = "".join(
        f"<th colspan='2' class='g{i % 2} gstart{'' if entered[n] else ' notentered'}'>"
        f"{esc(label(n))}{'' if entered[n] else ' <small>(not entered)</small>'}</th>"
        for i, n in enumerate(feat_names)
    )
    sub = "".join(
        f"<th class='g{i % 2} gstart'>soil pit</th><th class='g{i % 2}'>candidate</th>"
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
            # A property the user never entered is informational, not a one-sided
            # mismatch — mute it rather than flagging it orange.
            if not entered[n]:
                cells.append(
                    f"<td class='{g} gstart na'>—</td>"
                    f"<td class='{g} notentered'>{fmt(f['candidate'])}</td>"
                )
                continue
            one = "" if f["status"] == "both" else " one"
            # Show Δ and, when both values are present, the |pit−cand| ÷ range that
            # produced it, as a hover title so the normalization is inspectable.
            nd, title = "", ""
            if f["norm_diff"] is not None:
                nd = f" <small>Δ{f['norm_diff']:.3f}</small>"
                if f.get("user") is not None and f.get("candidate") is not None and f.get("range"):
                    title = (
                        f" title='|{f['user']} − {f['candidate']}| ÷ {f['range']} "
                        f"= {f['norm_diff']:.3f}'"
                    )
            cells.append(
                f"<td class='{g} gstart'>{fmt(f['user'])}</td>"
                f"<td class='{g}{one}'{title}>{fmt(f['candidate'])}{nd}</td>"
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
        f"<div class='comp'><div class='ctitle blue'>Soil horizons — depth-layer properties "
        f"<b>{fmt(c.get('score'))}</b></div>"
        f"<div class='note'>A <i>horizon</i> is a soil depth layer; this scores how well "
        f"the candidate's layered profile (sand/clay/rock-fragments"
        f"{', color' if any(fn in ('l', 'a', 'b') for fn in feat_names) else ''} by depth) "
        f"matches the soil pit's. Full depth range shown, 0 to max(soil pit, this soil). "
        f"The algorithm only compares the recorded depths ({win}); "
        f"<span class='nocompare' style='padding:0 4px'>greyed rows</span> are outside "
        f"that window (not compared). <b>soil pit</b>/<b>candidate</b> = the values at that "
        f"depth (— = none there); <b>Δ</b> = |soil pit − candidate| ÷ range, where "
        f"<i>range</i> is the spread of that property across the soils compared at this "
        f"depth, but never less than 10% of the property's fixed plausible range (a "
        f"floor so a near-constant property can't inflate tiny differences); "
        f"<b>slice dist</b> = the <i>equal-weighted average of the Δ's</i> present in "
        f"the band (wt is <i>not</i> applied here — it's applied when combining bands, "
        f"below).</div>"
        f"<table class='hz'><tr><th rowspan='2'>depth</th><th rowspan='2'>wt</th>{head}"
        f"<th rowspan='2' class='distcol'>slice&nbsp;dist</th></tr>"
        f"<tr>{sub}</tr>{''.join(rows)}</table>{combine}</div>"
    )


def render_color(c):
    de = c.get("delta_e")
    de_s = ", ".join(f"{x:.1f}" for x in de) if de else "—"
    return (
        f"<div class='comp'><div class='ctitle blue'>Color <b>{fmt(c.get('score'))}</b></div>"
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
        f"<div class='comp'><div class='ctitle blue'>Site <b>{fmt(c.get('score'))}</b></div>"
        f"<table class='hz'><tr><th>feature</th><th>soil pit</th><th>candidate</th><th>Δ</th></tr>"
        f"{feats}</table></div>"
    )


RENDERERS = {
    "location": render_location,
    "horizon": render_horizon,
    "color": render_color,
    "site": render_site,
}


def render_combined(cand):
    """Roll-up shown at the bottom: how the blue scores combine.
    properties = weighted avg(horizon, site|color); combined = (properties + location) ÷ 2."""
    comps = {c["type"]: c for c in cand["score_components"]}
    hz = comps.get("horizon", {}).get("score")
    loc = comps.get("location", {}).get("score")
    props = cand.get("properties_score")
    combined = cand.get("combined_score")
    site, color = comps.get("site"), comps.get("color")

    if site is not None:
        s = site.get("score")
        props_line = (
            f"<b class='blue'>properties</b> = (<b class='blue'>horizon</b> {fmt(hz)} + "
            f"<b class='blue'>site</b> {fmt(s)}) ÷ 1.5 = <b>{fmt(props)}</b>"
        )
        inputs = "horizon (wt 1), site (wt 0.5)"
    elif color is not None:
        cc, w = color.get("score"), color.get("weight", 0.3)
        props_line = (
            f"<b class='blue'>properties</b> = (<b class='blue'>horizon</b> {fmt(hz)} + "
            f"{w} × <b class='blue'>color</b> {fmt(cc)}) ÷ {round(1 + w, 3)} = <b>{fmt(props)}</b>"
        )
        inputs = f"horizon (wt 1), color (wt {w})"
    else:
        props_line = f"<b class='blue'>properties</b> = <b class='blue'>horizon</b> {fmt(hz)}"
        inputs = "horizon"

    combined_line = (
        f"<b class='blue'>combined</b> = (properties {fmt(props)} + "
        f"<b class='blue'>location</b> {fmt(loc)}) ÷ 2 = <b class='blue'>{fmt(combined)}</b>"
    )
    ov_line = ""
    for o in cand.get("overrides", []):
        ov_line = (
            f"<div class='lf'>⚑ a rule then overrides combined "
            f"{fmt(o.get('score_before'))} &rarr; <b>{fmt(o.get('score_after'))}</b> "
            f"— {esc(o.get('rule', ''))}</div>"
        )
    return (
        f"<div class='combinedbox'><div class='ctitle blue'>Combined score "
        f"<b>{fmt(combined)}</b></div>"
        f"<div class='lf'>{props_line}</div>"
        f"<div class='lf'>{combined_line}</div>{ov_line}"
        f"<div class='lf' style='margin-top:4px'><b>weights</b> → {inputs}; then "
        f"<b class='blue'>properties</b> and <b class='blue'>location</b> equally "
        f"(1 each) — so <b class='blue'>location</b> is ~half the final score.</div>"
        f"<div class='note'>Every score is normalized to 0–1 by these "
        f"weighted-average denominators, so combined stays ≤ 1 (the only exception "
        f"is a rule override, which force-sets it to 1.001 / 0.001).</div></div>"
    )


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
        f"<span class='rank'>Candidate #{cand.get('rank')}</span>"
        f"<span class='name'>{esc(cand['name'])}</span>"
        f"<span class='combined'>combined {bar(cand.get('combined_score'))}</span>"
        f"<span class='props'>properties {fmt(cand.get('properties_score'))}</span></div>"
        f"{ov}{comps}{render_combined(cand)}</div>"
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
.ctitle.blue{color:#1560c4} .blue{color:#1560c4}
.combinedbox{margin-top:10px;border-top:2px solid #cdd6e6;padding-top:8px}
.combinedbox .lf{margin:2px 0}
.formula{color:#555;font-family:ui-monospace,monospace;font-size:12px}
/* location inputs -> decayed -> [combine] -> normalized flow */
.locflow{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:4px 0}
.locbox{border:1px solid #cdd6e6;border-radius:7px;padding:5px 9px;background:#fbfcff;
  font-size:12px;font-variant-numeric:tabular-nums}
.locbt{font-weight:700;font-size:10px;color:#678;text-transform:uppercase;letter-spacing:.04em;
  margin-bottom:3px}
.locarrow{color:#89a;font-size:20px;flex:0 0 auto}
.lf{color:#445;font-family:ui-monospace,monospace;font-size:12px;white-space:nowrap}
table.lt{border-collapse:collapse;font-size:11.5px}
table.lt td,table.lt th{padding:1px 6px 1px 0;text-align:left;color:#445}
table.lt th{color:#789;font-weight:600}
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
td.notentered{color:#8890a0;font-style:italic} /* property user didn't enter — context only */
th.notentered{font-weight:500}
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
        "<div class='legend'>The <b>soil pit</b> is the profile you recorded. You enter a "
        "<b>texture class</b> (e.g. Clay) and a <b>rock-fragment class</b>; the algorithm "
        "converts them to representative <b>sand %</b>, <b>clay %</b>, and "
        "<b>rock-fragment %</b> (shown as → above), and the horizon tables below compare "
        "on <i>those</i> numbers. A property the pit doesn't record (e.g. rock fragments) "
        "is shown greyed and marked <i>(not entered)</i> — the candidate's values are "
        "there for context but it takes no part in matching.<br>"
        "Each candidate's <b>combined score</b> ≈ (properties + location) ÷ (weight + 1): "
        "<b>Location</b> = how likely the soil is mapped here (distance-decayed share, "
        "normalized); <b>Properties</b> = how well its profile matches the pit's (Gower "
        "distance per depth band) plus color. Rule <b>overrides</b> can force a "
        "promote/demote.</div>"
    )
    cards = "".join(render_candidate(c) for c in trace.get("candidates", []))
    return (
        f"<!doctype html><meta charset=utf-8><style>{CSS}</style>"
        f"<h1>Soil ID explanation — {esc(trace.get('region'))}</h1>"
        f"<div class='site'>lat {site.get('lat')}, lon {site.get('lon')}</div>"
        f"<div class='pit'><div class='ctitle'>Soil pit "
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
