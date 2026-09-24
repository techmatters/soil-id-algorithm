# Soil-ID algorithm version (semver: MAJOR.MINOR.PATCH).
#
# MAJOR = model generation (== the legacy `model` metadata: "v2" -> MAJOR 2).
# MINOR = a result-affecting change (rank scores/order can move) -> clients flush
#         their cached soil-ID matches when MAJOR or MINOR changes.
# PATCH = a change that does NOT affect ranking output (explain report, perf,
#         packaging, refactors) -> clients do NOT flush on a PATCH-only bump.
#
# Bump MINOR whenever the rank snapshots (tests/us, tests/global) change.
# 2.1.0 = model gen 2 + the global/US correctness fixes (#375/#377/#378/#389 ...).
# 2.2.0 = shallow-soil demote applied to the horizon score (soft demotion) instead
#         of forcing the combined score to ~0.
# 2.3.0 = rock fragment ranked on the candidate's raw % instead of the binned class
#         midpoint (removes the class-boundary cliff + within-class flattening).
# 2.4.0 = US site weight (slope/elev vs horizon) 0.5 -> 0.25 (+0.34 pt top1).
__version__ = "2.4.0"
