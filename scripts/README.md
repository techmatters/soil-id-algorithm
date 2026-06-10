# soil-id scripts

## `wrb_descriptions_sync.py`

Syncs the WRB soil **Description**/**Management** narratives from the
terraso-mobile-client i18n files (the source of truth) into the `wrb_fao90_desc`
table. Default mode is a read-only HTML diff; `--write` rebuilds the table from the
JSON. Full run + deploy instructions are in the **Localization** guide in
`terraso-wiki` (`docs/development-guide/localization.md`).

Languages are auto-discovered from the translation files. The only special case is a
column-suffix mismatch, captured in one place:

```python
DB_SUFFIX_EXCEPTIONS = {"sw": "ks"}  # the DB column suffix for Swahili is "ks"
```

---

## Planned migration: rename the Swahili column suffix `ks` → `sw`

The table stores Swahili in `*_ks` columns ("Kiswahili"), but everything else — the
mobile i18n, ISO 639-1, POEditor, device locales — uses `sw`. In fact `ks` is the ISO
code for **Kashmiri**, so `ks`-for-Swahili is a latent footgun. The sync tool papers
over it with the `DB_SUFFIX_EXCEPTIONS` entry above. This runbook removes that
exception by renaming the DB side to `sw`.

**Why it must be coordinated.** The backend selects `Description_ks`/`Management_ks`.
If the columns are renamed while a deployed backend still reads `_ks` (or vice-versa),
global soil descriptions break. So the soil-id **code** change and the column
**rename** must go live together — do this right before / together with a backend
redeploy.

### 1. soil-id code change (one PR, then tag)

- `soil_id/db.py` — in `get_WRB_descriptions` **and** `getSG_descriptions`, change
  `Description_ks`/`Management_ks` → `Description_sw`/`Management_sw` (the `SELECT`
  and the DataFrame column lists).
- `soil_id/global_soil.py` — change the `siteDescription` dict keys
  `Description_ks`/`Management_ks` → `_sw` (HWSD path **and** SoilGrids path), plus any
  `TAXNWRB_pd.get("…_ks")`.
- `scripts/wrb_descriptions_sync.py` — empty the exception: `DB_SUFFIX_EXCEPTIONS = {}`.
- Regenerate the global test snapshots (`soil_id/tests/global/__snapshots__/`), which
  embed the `Description_ks`/`Management_ks` keys, and fix any other `_ks` test refs.
- Merge, tag a new soil-id release, and bump the backend's `requirements/base.in` pin.

### 2. The schema rename (apply to every database)

```sql
ALTER TABLE wrb_fao90_desc RENAME COLUMN description_ks TO description_sw;
ALTER TABLE wrb_fao90_desc RENAME COLUMN management_ks  TO management_sw;
ALTER TABLE wrb_fao90_desc RENAME COLUMN wrb_tax_ks     TO wrb_tax_sw;
```

`RENAME COLUMN` is metadata-only (fast, preserves data). Apply it to:

- the local `soil-id-db` — then `make dump_soil_id_db` → `make build_docker_image` →
  `docker push`, so local dev + CI get the renamed image; and
- the **staging** then **production** primary database (where the soil-id tables
  actually live in those environments — `SOIL_ID_DATABASE_URL` is unset there).

### 3. Coordinate the cutover

The renamed columns and the `_sw`-reading backend must be live together. Two ways:

- **Simple (brief blip — acceptable for this non-critical field):** apply the rename
  to the prod DB and ship the backend deploy (new soil-id tag) in the same window.
  Between the two, the old backend briefly reads a missing `_ks` and global
  descriptions error — tolerable, since the app falls back to its own i18n.
- **Zero-downtime (expand / contract):** first add `_sw` as copies
  (`ADD COLUMN … ; UPDATE … SET description_sw = description_ks; …`), deploy the
  backend (now reads `_sw`), then `DROP COLUMN … _ks`. No mismatch window, more steps.

### 4. Verify

Run the sync tool's diff (step 1 in the localization guide) against each database.
With the exception removed and the columns renamed, it should report `changed=0` and
no `new_lang`, confirming `sw` lines up end to end.
