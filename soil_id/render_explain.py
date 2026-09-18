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

"""Render a Soil ID explain trace into a self-contained HTML report.

`render_html(trace)` takes a single explain-trace dict (as produced by
``rank_soils[_global](..., explain=Recorder())`` and captured by
``soil_id.explain.build_trace``) and returns a complete, styled HTML document as
a string. The renderer is generic over the trace's ``score_components``, so the
same code handles the US and global algorithms.

This module is the importable home of the renderer. ``scripts/render_soil_explain.py``
is a thin CLI wrapper around it (file/URL loading and browser preview), and other
services (e.g. the terraso-backend explain endpoint) import ``render_html`` from here.
"""

import html


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
            f"<div class='lf'>&nbsp; e<sup>{c.get('exp_coeff'):g} × {dstr}</sup>)"
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

    # component total (only when the component spans more than this one map unit).
    # Itemize the visible map-unit occurrences (candidates of the same series) that
    # sum toward it; note any remainder from occurrences filtered before ranking.
    if multi:
        occ = sorted(
            (o for o in (c.get("occurrences") or []) if o.get("distance_score") is not None),
            key=lambda o: -o["distance_score"],
        )
        parts = " + ".join(f"{esc(o['name'])} {o['distance_score']:.3f}" for o in occ)
        shown = sum(o["distance_score"] for o in occ)
        rem = (comp - shown) if comp is not None else 0.0
        extra = f" + {rem:.3f} <span class='hint'>(other map units)</span>" if rem > 0.005 else ""
        combine = (
            f"<div class='lf lfwrap'>sum over this series' map units:</div>"
            f"<div class='lf lfwrap'>{parts}{extra} = <b>{fmt(comp)}</b></div>"
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

    src = c.get("data_source")
    coeff = c.get("exp_coeff")
    floor_m = c.get("floor_m")
    if floor_m is None:
        floor_txt = "?"
    elif floor_m < 1000:
        floor_txt = f"~{floor_m:.0f} m"
    else:
        floor_txt = f"~{floor_m / 1000:.1f} km"
    if coeff is not None:
        decay_note = (
            f"<b>decay = max(0.25, e<sup>{coeff:g} × distance_m</sup>)</b> — an "
            f"exponential fall-off, floored at 0.25 (25%). The coefficient <b>{coeff:g}</b> "
            f"is set for the <b>{esc(str(src))}</b> data source (finer sources decay "
            f"faster: SSURGO −0.008, HWSD2 −0.00036888, STATSGO −0.0002772). The 0.25 "
            f"floor is reached at <b>{floor_txt}</b>; beyond that every candidate sits at "
            f"the floor. "
        )
    else:
        decay_note = "<b>decay</b> = distance-decayed factor, floored at 0.25 (25%). "
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


def render_horizon(c, overrides=None):
    segs = c.get("segments", [])
    if not segs:
        return ""
    # A horizon-targeted override (e.g. the shallow-soil demote) zeroes the horizon
    # score; show it inline here rather than at the combined roll-up.
    override = (overrides or [None])[0]
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

    # The Δ denominator (range) per feature across the compared bands. If it's the
    # same at every depth, show it once in the header; otherwise it's shown per
    # cell (and the header shows the min–max span).
    def _ranges(n):
        return sorted(
            {
                f["range"]
                for s in segs
                if s.get("compared", True)
                for f in s["features"]
                if f["name"] == n and f.get("range") is not None
            }
        )

    feat_ranges = {n: _ranges(n) for n in feat_names}

    def _range_hdr(n):
        rs = feat_ranges[n]
        if not entered[n] or not rs:
            return ""
        span = f"{rs[0]:g}" if len(rs) == 1 else f"{rs[0]:g}–{rs[-1]:g}"
        return f"<br><small class='rng'>Δ ÷ [{span}]</small>"

    # each feature is a color-grouped pair of sub-columns (you / candidate)
    head = "".join(
        f"<th colspan='2' class='g{i % 2} gstart{'' if entered[n] else ' notentered'}'>"
        f"{esc(label(n))}{'' if entered[n] else ' <small>(not entered)</small>'}{_range_hdr(n)}</th>"
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
                # When the range varies across depths, show this band's range next
                # to Δ; when constant it's already in the column header.
                rng = f.get("range")
                per_cell = (
                    f" <span class='rng'>[{rng:g}]</span>"
                    if (rng is not None and len(feat_ranges.get(n, [])) > 1)
                    else ""
                )
                nd = f" <small>Δ{f['norm_diff']:.3f}{per_cell}</small>"
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
    # The algorithm's horizon score is 1 − D_sum, where D_sum is the depth-weighted
    # mean of the per-slice distances (weight = depth_weight per cm), masking depths
    # with no comparable property (soil-id-algorithm#389). So the authoritative
    # distance is 1 − score; we take it straight from the score rather than
    # re-deriving it from the bands below (band grouping + masking don't reproduce
    # it exactly). The per-depth table below shows the slice distances that feed it.
    hz_score = c.get("score")
    # When overridden, the component score is the post-override value (0). The
    # per-slice table and the combine note below still describe how the *earned*
    # horizon score was computed, so base that math on score_before and then note
    # the override.
    earned = (
        override["score_before"]
        if override and override.get("score_before") is not None
        else hz_score
    )
    hz_dist = None if earned is None else 1 - earned
    override_note = (
        f"; then <b>overridden to {fmt(override['score_after'])}</b> "
        f"(shallow soil can't match at this depth)"
        if override
        else ""
    )
    combine = (
        f"<div class='note'>→ horizon distance (depth-weighted mean of the per-slice "
        f"distances below, masking depths with no comparable property) = "
        f"<b>{fmt(hz_dist)}</b>; horizon score = 1 − {fmt(hz_dist)} = "
        f"<b>{fmt(earned)}</b>{override_note}</div>"
        if hz_dist is not None
        else ""
    )
    ov_html = ""
    if override:
        ov_html = (
            f"<div class='override'>⚑ override: <b>{esc(override['rule'])}</b> — "
            f"horizon score {fmt(override.get('score_before'))} &rarr; "
            f"<b>{fmt(override.get('score_after'))}</b></div>"
        )
    return (
        f"<div class='comp'><div class='ctitle blue'>Soil horizons — depth-layer properties "
        f"<b>{fmt(hz_score)}</b></div>{ov_html}"
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
        f"floor so a near-constant property can't inflate tiny differences). The "
        f"<span class='rng'>Δ ÷ [range]</span> shown in each column header is that "
        f"denominator when it's the same at every depth; when it varies with depth "
        f"the header shows its <span class='rng'>[min–max]</span> span and each cell "
        f"shows its own <span class='rng'>[range]</span>. "
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
    features = c.get("features", [])
    feats = "".join(
        f"<tr><td>{esc(label(f['name']))}</td><td>{fmt(f['user'])}</td>"
        f"<td class='{'' if f['status'] == 'both' else 'one'}'>{fmt(f['candidate'])}</td>"
        f"<td>{fmt(f.get('norm_diff'))}</td><td>{fmt(f.get('weight'))}</td></tr>"
        for f in features
    )
    site_wt = c.get("weight")
    # Headline is the RAW site similarity (1 − site distance); the 0.5 site weight
    # is applied in the Combined roll-up, not baked in here. The authoritative site
    # distance is 1 − similarity (taken straight from the score), NOT the
    # Σ(Δ×weight)÷Σweight recompute over the rows below: the algorithm's site
    # distance can include an elevation comparison against a server-fetched
    # elevation (fetched when elevation isn't entered), which the per-feature rows
    # here don't capture. The table below shows the entered features that feed it.
    sim = (c.get("score") / site_wt) if site_wt else c.get("score")
    site_dist = None if sim is None else 1 - sim
    combine = ""
    if site_dist is not None:
        combine = (
            f"<div class='lf'>site distance (weighted mean of the Δ's below, plus a "
            f"server-fetched elevation when elevation isn't entered) = {fmt(site_dist)}</div>"
            f"<div class='lf'>site similarity = 1 − {fmt(site_dist)} = <b>{fmt(sim)}</b></div>"
        )
    return (
        f"<div class='comp'><div class='ctitle blue'>Site <b>{fmt(sim)}</b></div>"
        f"<table class='hz'><tr><th>feature</th><th>soil pit</th><th>candidate</th>"
        f"<th>Δ</th><th>weight</th></tr>{feats}</table>{combine}"
        f"<div class='note'>Δ = |soil pit − candidate| ÷ range (same normalization as "
        f"the horizon). Features are weighted (slope 1, elevation 0.5, depth-to-bedrock "
        f"1.5). This is the raw site similarity; it counts toward the combined score at "
        f"weight {fmt(site_wt)} (folded into the % in the roll-up below).</div></div>"
    )


RENDERERS = {
    "location": render_location,
    "horizon": render_horizon,
    "color": render_color,
    "site": render_site,
}


def render_combined(cand):
    """Bottom roll-up: the combined score as a flat weighted sum of the base
    scores (a re-expression of the nested properties/combined math)."""
    comps = {c["type"]: c for c in cand["score_components"]}
    hz = comps.get("horizon", {}).get("score")
    loc = comps.get("location", {}).get("score")
    props = cand.get("properties_score")
    combined = cand.get("combined_score")
    site, color = comps.get("site"), comps.get("color")

    # The "properties" partner of horizon (site for US, color for global) and its
    # weight. Site is stored pre-scaled by its weight, so recover the raw value.
    if site is not None:
        pw = site.get("weight") or 0.5
        praw = (site.get("score") / pw) if pw else site.get("score")
        plabel = "site"
    elif color is not None:
        pw = color.get("weight", 0.3)
        praw = color.get("score")  # colour similarity is already raw
        plabel = "color"
    else:
        pw, praw, plabel = 0.0, None, None

    # Flatten the nested weights into each base score's share of the final:
    #   location: 1/2; horizon: 1/2 · 1/(1+pw); partner: 1/2 · pw/(1+pw).
    wl = 0.5
    wh = 0.5 * (1 / (1 + pw))
    wp = 0.5 * (pw / (1 + pw)) if plabel else 0.0

    def row(name, val, w):
        contrib = val * w if val is not None else None
        return (
            f"<tr><td><b class='blue'>{name}</b></td>"
            f"<td>{fmt(val)}</td><td>× {w * 100:.1f}%</td><td>= {fmt(contrib)}</td></tr>"
        )

    body = row("location", loc, wl) + row("horizon", hz, wh)
    if plabel:
        body += row(plabel, praw, wp)
    body += (
        f"<tr class='ctot'><td><b class='blue'>combined</b></td><td></td><td></td>"
        f"<td>= <b class='blue'>{fmt(combined)}</b></td></tr>"
    )

    ov_line = ""
    for o in cand.get("overrides", []):
        if _ov_target(o) != "combined":
            continue
        ov_line = (
            f"<div class='lf'>⚑ a rule then overrides combined "
            f"{fmt(o.get('score_before'))} &rarr; <b>{fmt(o.get('score_after'))}</b> "
            f"— {esc(o.get('rule', ''))}</div>"
        )

    exact = (
        f"exact (nested): properties = (horizon + {pw:g} × {plabel}) ÷ {1 + pw:g}; "
        f"combined = (properties {fmt(props)} + location) ÷ 2"
        if plabel
        else f"exact: combined = (horizon {fmt(props)} + location) ÷ 2"
    )
    return (
        f"<div class='combinedbox'><div class='ctitle blue'>Combined score "
        f"<b>{fmt(combined)}</b></div>"
        f"<table class='lt ctbl'>{body}</table>{ov_line}"
        f"<div class='note'><b class='blue'>location</b> is always half the score; the "
        f"other half is properties, split between <b class='blue'>horizon</b> and "
        f"<b class='blue'>{plabel or '—'}</b> (so {plabel or 'the partner'} is the "
        f"smallest share). The percentages shift if a component is absent. "
        f"<span class='hint'>{exact}</span></div></div>"
    )


def _ov_target(o):
    # v1 traces have no `target`; those overrides were all combined-level.
    return o.get("target", "combined")


def render_candidate(cand):
    overrides = cand.get("overrides", [])
    horizon_ovs = [o for o in overrides if _ov_target(o) == "horizon"]
    combined_ovs = [o for o in overrides if _ov_target(o) == "combined"]
    parts = []
    for c in cand["score_components"]:
        if c["type"] == "horizon":
            parts.append(render_horizon(c, horizon_ovs))
        else:
            parts.append(RENDERERS.get(c["type"], lambda _: "")(c))
    comps = "".join(parts)
    ov = ""
    for o in combined_ovs:
        ov = (
            f"<div class='override'>⚑ override: <b>{esc(o['rule'])}</b> — "
            f"score {fmt(o.get('score_before'))} &rarr; {fmt(o.get('score_after'))}</div>"
        )
    # App visibility: the app shows one entry per series (the best-scoring one).
    ar, rep = cand.get("app_rank"), cand.get("app_repr")
    if ar is not None:
        appnote = f"<span class='appnote'>shown in the app as #{ar}</span>"
    elif rep:
        appnote = (
            f"<span class='appnote dup'>duplicate of this series — not shown "
            f"separately in the app; folded into {esc(rep['name'])} (app #{rep['app_rank']})</span>"
        )
    else:
        appnote = ""
    return (
        f"<div class='card'><div class='chead'>"
        f"<span class='rank'>Candidate #{cand.get('rank')}</span>"
        f"<span class='name'>{esc(cand['name'])}</span>{appnote}"
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
.name{font-weight:700;font-size:15px} .props{color:#666}
.appnote{flex:1;font-size:11px;color:#3a7a3a;font-weight:600}
.appnote.dup{color:#98812e;font-weight:500;font-style:italic}
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
table.ctbl{font-size:12.5px;margin:2px 0} table.ctbl td{padding:1px 10px 1px 0}
table.ctbl .ctot td{border-top:1px solid #cdd6e6;padding-top:3px}
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
.rng{color:#9178a8;font-weight:400} th .rng{color:#7a6a90}
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
