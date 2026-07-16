# soil-id scripts

## Containerized tests / snapshots on macOS

GDAL doesn't build under uv/pip on macOS, and the output snapshots are
environment-sensitive — they must be produced against an interpreter matching CI
**and** the exact pinned `soil-id-db` image CI uses (`docker-compose.yml`
`db.image`), **not** the `:latest` your local terraso-backend runs. Two pieces make
this transparent:

- **`snapshot-runner.Dockerfile`** — a runner image mirroring the CI `test` job
  (Ubuntu + ubuntugis + Python 3.13 + the pinned requirements). Built once, layer-cached
  on the requirements files.
- **`run_in_container.sh`** — starts the pinned DB on a private network (no host port, so
  it can't collide with a running terraso-backend on 5432), then runs the given Makefile
  target(s) in the runner with the repo bind-mounted read-write, and tears everything
  down afterward.

You don't normally call `run_in_container.sh` directly: the Makefile's `test`,
`test_unit`, and `test_update_unit_snapshots` targets **auto-detect** when GDAL isn't
importable and delegate to it, so the same `make …` command works on macOS and CI. Set
`NATIVE=1` to force the local interpreter (it's set automatically inside the runner to
avoid recursion).

## `regen_snapshots.sh`

Regenerates the unit-test **output** snapshots
(`soil_id/tests/{us,global}/__snapshots__/test_soil_location[...].json`) **and verifies
them** in one DB session: it runs `make test_update_unit_snapshots test_unit` (update,
then re-run without `--snapshot-update`; the second pass must pass clean or the snapshots
are non-deterministic), then shows `git status` for the snapshot dirs.

```sh
make regen_snapshots            # regenerate + verify
./scripts/regen_snapshots.sh --no-verify
REBUILD=1 make regen_snapshots  # force-rebuild the runner image (after a deps bump)
```

> Not to be confused with `make test_update_api_snapshots`, which refreshes the
> **input** fixtures (raw SoilWeb/elevation API responses under
> `__snapshots__/test_api_snapshots`) by hitting the **live** external APIs. That's a
> different, drift-prone operation and is not what re-baselining algorithm output needs.

---

## `wrb_descriptions_sync.py`

Syncs the WRB soil **Description**/**Management** narratives from the
terraso-mobile-client i18n files (the source of truth) into the `wrb_fao90_desc`
table. Default mode is a read-only HTML diff; `--write` rebuilds the table from the
JSON. Full run + deploy instructions are in the **Localization** guide in
`terraso-wiki` (`docs/development-guide/localization.md`).

Languages are auto-discovered from the translation files, and each language's code is
its column suffix (`description_<lang>` / `management_<lang>`), so a new language needs
no edit. The backend reads **only the English columns** (`description_en` /
`management_en`); the other languages are kept current and staged for a possible
future multilingual API, but nothing reads them yet.

---

## Deferred cleanup: drop the obsolete `*_ks` columns

Historically the table stored Swahili under a `*_ks` suffix ("Kiswahili"), but `ks` is
the ISO code for **Kashmiri**, not Swahili (`sw`). The sync now writes Swahili to the
correctly-named `description_sw`/`management_sw`, so the old `description_ks`,
`management_ks`, and `wrb_tax_ks` columns are **obsolete** — left in place (all `NULL`)
only so the drop can happen on its own schedule.

This cleanup is **window-free**: the backend reads only `description_en`, so nothing
references `*_ks`, and they can be dropped at any time.

### Already done
The three `*_ks` columns have been relaxed to nullable in local, staging, and prod, so
the exception-free rebuild can stop populating them:
```sql
ALTER TABLE wrb_fao90_desc
  ALTER COLUMN description_ks DROP NOT NULL,
  ALTER COLUMN management_ks  DROP NOT NULL,
  ALTER COLUMN wrb_tax_ks     DROP NOT NULL;
```

### The drop (run when ready, per database)
After the exception-free sync has run (Swahili now lives in `description_sw`), the
`*_ks` columns are all `NULL` and safe to remove:
```sql
ALTER TABLE wrb_fao90_desc
  DROP COLUMN description_ks, DROP COLUMN management_ks, DROP COLUMN wrb_tax_ks;
```
Apply to the local `soil-id-db` (then rebuild + push the image) and to the staging then
production primary databases. `DROP COLUMN` takes any `NOT NULL`/default with it — no
other constraint changes needed.

> `ALTER TABLE` needs a brief `ACCESS EXCLUSIVE` lock. If it hangs, a leaked
> `idle in transaction` session is holding `wrb_fao90_desc` — terminate it
> (`SELECT pg_terminate_backend(pid) …`) and re-run with `SET lock_timeout='5s'`.
