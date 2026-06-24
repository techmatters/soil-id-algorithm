# Soil profiles deeper than 200 cm — analysis & decision

**Status:** analysis complete; implementation **deferred**.
**Date:** 2026-06-23.

## Question

A scientist (Ukraine — i.e. the **global/international** path) asked to be able to
record soil data below 200 cm. The soil-ID algorithm currently caps depth at
200 cm in several places. Before lifting that cap, we wanted to know whether
deeper data would actually change soil-ID **results**, or whether "recording
> 200 cm" is purely a data-capture concern.

There are two independent code paths:

- **Global / international** — `soil_id/global_soil.py::rank_soils_global`, reference data from **HWSD2** (in our `soil-id-db`).
- **US** — `soil_id/us_soil.py::rank_soils`, reference data from **SSURGO** via the live SoilWeb / NRCS SDA APIs.

## Bottom line

| Path | Does reference data exist > 200 cm? | Worth lifting the cap? |
|---|---|---|
| **Global (HWSD2)** | **No** — data hard-stops at 200 cm | **No.** Allow *recording* > 200 in the app, but the algorithm structurally cannot use it. |
| **US (SSURGO)** | **Yes** — ~16% of major components go deeper | **Yes, modestly.** Real but bounded win, almost entirely in the 200–250 cm band. |

## Global path: the 200 cm cap is a data boundary, not a software choice

HWSD2 — the only component-profile source for the global path — is published on
fixed standard layers. The deepest layer in the entire dataset is 200 cm:

```sql
-- against soil-id-db (hwsd2_data)
SELECT botdep, COUNT(*) FROM hwsd2_data GROUP BY botdep ORDER BY botdep;
--  20 | 57115
--  40 | 52570
--  60 | 47313
--  80 | 47313
-- 100 | 47313
-- 150 | 47313
-- 200 | 47313   <-- deepest; nothing below 200
```

WISE (`wise_soil_data`) is shallower still (max 120 cm) and is only joined via
`wise30s_smu_id`, not used as a separate profile source.

Consequences for feeding user data > 200 cm on the global path:

1. **Texture/property matching > 200 is impossible** — there is no component data
   there to compare against (every candidate is NaN below 200).
2. **Depth-extent (soil/non-soil) > 200 cannot re-rank** — since every component
   stops at ≤ 200, a user reporting soil to, say, 300 cm is "deeper than
   component" against *every* candidate equally; a uniform penalty doesn't change
   ordering.

So lifting the global cap yields no discriminating information. **Decision:
allow recording > 200 cm at the app/data-model layer, leave the global algorithm
unchanged.** (A prototype that lifted the cap in `rank_soils_global`,
commit `741a173` on `fix/368-…`, was reverted for this reason.)

## US path: SSURGO genuinely goes deeper than 200 cm

SSURGO horizon data comes from the live SoilWeb/SDA APIs
(`soil_id/services.py`) with **no depth limit in the query**, and follows real
soil profiles. We measured how often US **major** components are described deeper
than 200 cm via NRCS Soil Data Access.

### SDA query — component count by max depth

```sql
SELECT CASE
         WHEN maxdep <= 100 THEN 'a_0-100'
         WHEN maxdep <= 150 THEN 'b_100-150'
         WHEN maxdep <= 200 THEN 'c_150-200'
         WHEN maxdep <= 250 THEN 'd_200-250'
         ELSE 'e_gt250' END AS depth_bucket,
       COUNT(*) AS n_components,
       SUM(comppct_r) AS sum_comppct
FROM (
  SELECT c.cokey, MAX(c.comppct_r) AS comppct_r, MAX(ch.hzdepb_r) AS maxdep
  FROM component c
  INNER JOIN chorizon ch ON c.cokey = ch.cokey
  WHERE c.majcompflag = 'Yes' AND ch.hzdepb_r IS NOT NULL
  GROUP BY c.cokey
) t
GROUP BY <same CASE>
ORDER BY 1;
```

(POST as `{"format":"JSON+COLUMNNAME","query":"…"}` to
`https://sdmdataaccess.nrcs.usda.gov/Tabular/SDMTabularService/post.rest`.)

### Results (national, major components only)

| Max component depth | Components | % of components | Σ comppct | % of comppct | avg comppct |
|---|---:|---:|---:|---:|---:|
| 0–100 cm | 49,007 | 10.2% | 2,106,611 | 7.6% | 43.0 |
| 100–150 cm | 43,439 | 9.0% | 2,154,794 | 7.8% | 49.6 |
| 150–200 cm | 312,885 | 64.8% | 18,238,589 | 66.0% | 58.3 |
| **200–250 cm** | **75,307** | **15.6%** | **4,993,046** | **18.1%** | **66.3** |
| **> 250 cm** | **2,059** | **0.4%** | **124,550** | **0.5%** | **60.5** |
| **Total** | **482,697** | | **27,617,590** | | |

Takeaways:

- **~16% of major components (and ~18.5% by component-percentage weighting)
  carry real data below 200 cm** that the algorithm currently discards.
- The deep tail is **thin**: essentially all of it is in **200–250 cm**; only
  **0.4%** of components go past 250 cm. A **250 cm ceiling captures ~99.6%** of
  the available deep data — no need for an unbounded profile.
- Deep components have a **higher** average `comppct_r` (43 → 66 as depth
  increases), i.e. they tend to be the **dominant** component of their map unit,
  not marginal inclusions. So the affected soils are not edge cases.

> Caveat: `comppct_r` is share-within-map-unit, **not** multiplied by map-unit
> acreage. True area weighting would require national polygon-geometry
> summation (too heavy for SDA). `comppct_r` is a reasonable
> landscape-composition proxy, and the realized benefit further depends on how
> often users actually record past 200 cm at deep-soil sites.

## How the US path handles > 200 cm today

It **truncates at 200** — same shape as the pre-fix global path:

| Location | Cap |
|---|---|
| `us_soil.py:1630` | `pedon_slice_index = [x for x in … if x < 200]` (drops user horizons below 200) |
| `us_soil.py:1770-1772` | `max_depth = 200` (soil matrix) |
| `us_soil.py:1902` | `depth_weight` = hardcoded 200-long array |
| `us_soil.py:1839` | `.loc[pedon_slice_index]` (would `KeyError` if the cap were lifted — needs `reindex`) |
| `utils.py:513-514` `max_comp_depth` | clamps each component's `c_very_bottom` to 200 (shared) |
| `utils.py` `getProfile` | pads/truncates every component (and OSD) profile to 200 rows (shared) |

## Estimated work to enable US > 200 (ceiling ≈ 250 cm)

A **medium** change — larger than the global one because it spans both
`list_soils` and `rank_soils`, touches **shared** `utils` functions (which must
stay a no-op for global), and the US snapshot suite is live-API-based.

**A. Shared `utils.py` (keep global byte-identical via a `max_depth=200` default):**
- `max_comp_depth` — make the 200 cap a parameter/ceiling. *Low.*
- `getProfile` — parameterize the profile length (~4 branches + all-NaN early returns hardcode 200). *Medium; shared → needs a global regression check.*

**B. `us_soil.py::rank_soils` (mirrors the reverted global change, ~6 spots):**
- Compute `max_observed_depth = max(200, deepest user + component bottom)`, capped at the ceiling.
- Make `pedon_slice_index` cutoff (1630), `max_depth` (1770-1772), and `depth_weight` (1902) dynamic.
- Replace `.loc[pedon_slice_index]` with `.reindex(pedon_slice_index)` (1839) — the proven KeyError-avoidance pattern.

**C. `us_soil.py::list_soils` (where SSURGO deep data actually enters):**
- Component **and OSD** profiles are built with `getProfile`/`max_comp_depth` at *list* time and baked into `rank_data_csv`. Pass the deeper ceiling and confirm the 200–250 cm rows survive the list→rank CSV handoff. *Main correctness risk.*

**D. Decisions:** ceiling = **250 cm**; depth-weight stays 1.0 below 20 cm.

**E. Testing / snapshots:**
- Verify a deep-soil US location where the 200–250 cm horizon now moves the result.
- Regenerate the US `@pytest.mark.api_snapshot` snapshots — these hit the **live**
  SoilWeb/SDA API (`make test_api_snapshot`, `api-snapshot-check.yml`), so regen
  is network-dependent and slower/flakier than the global DB-backed snapshots.

**Rough sizing:** ~2× the global change (~60–80 lines across 3 files) plus
meaningful validation time for the live-API snapshots. No new data dependency —
SSURGO already returns the deep horizons; we're only lifting software caps.

## Recommendation

- **Global / Ukraine:** allow **recording** > 200 cm; **do not** change the global
  algorithm (no reference data to use).
- **US:** a **real but modest** improvement (~16% of major components, ~18.5% by
  composition, almost all 200–250 cm). **Defer**; if picked up, scope to a
  **250 cm ceiling** and keep global untouched via defaulted shared-util params.

## Related

- Issue #368 (the depth-interval *alignment* fix) operates **within** 0–200 cm and
  is independent of this depth-cap question.
