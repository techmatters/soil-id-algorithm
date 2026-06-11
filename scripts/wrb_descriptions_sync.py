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

"""Compare WRB soil descriptions in the mobile client's i18n files against the
soil-id ``wrb_fao90_desc`` table, treating the JSON as the source of truth.

The mobile client (terraso-mobile-client) keeps hand-cleaned, multi-language
copies of the WRB soil "Description"/"Management" narratives under
``soil.match_info.<key>.{description,management}``. The backend reads the same
narratives from the ``wrb_fao90_desc`` Postgres table (columns
``Description_<lang>`` / ``Management_<lang>``), where they still carry literal
``<br>`` tags and can drift from the client copies.

Default mode is a **read-only** HTML diff (``<br>`` differences are ignored, so
only genuine wording/translation drift shows). ``--write`` rebuilds the table
from the JSON (delete + reinsert, adding columns for new languages), emitting the
applied SQL for review; regenerate and redistribute the DB dump afterwards.

Run it where the soil-id DB is reachable, e.g. inside the backend container:

    python scripts/wrb_descriptions_sync.py \
        --translations-dir /path/to/terraso-mobile-client/dev-client/src/translations \
        --out wrb_diff.html
"""

import argparse
import difflib
import html
import json
import os
import re
from pathlib import Path

import psycopg

# A JSON language code is used directly as its wrb_fao90_desc column suffix
# (description_<lang> / management_<lang>), so a new language needs no edit here.
FIELDS = ("description", "management")

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def normalize(text):
    """Strip <br> tags and collapse whitespace (the client copies are cleaned)."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", _BR_RE.sub(" ", text)).strip()


def normalize_soil_name(name):
    """WRB_tax -> i18n key, matching the app (lowercase, spaces -> underscores)."""
    return name.strip().lower().replace(" ", "_")


def load_translations(translations_dir):
    """{lang: {soil_key: {"name":, "description":, "management":}}} for every *.json
    that has soil descriptions. Languages are auto-discovered from the files."""
    out = {}
    for path in sorted(Path(translations_dir).glob("*.json")):
        match_info = json.loads(path.read_text()).get("soil", {}).get("match_info", {})
        if match_info:
            out[path.stem] = match_info
    return out


def load_db(conn):
    """Return ({soil_key: {"wrb_tax":, <suffix>: {description, management}}}, existing).

    Only reads language columns that actually exist, so the tool works both before
    and after --write has added the new-language columns.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'wrb_fao90_desc'"
        )
        columns = {r[0].lower() for r in cur.fetchall()}
    prefix = "description_"
    existing = sorted(
        col[len(prefix) :]
        for col in columns
        if col.startswith(prefix) and f"management_{col[len(prefix) :]}" in columns
    )
    select_cols = ", ".join(f"description_{s}, management_{s}" for s in existing)
    out = {}
    with conn.cursor() as cur:
        cur.execute(f"SELECT wrb_tax, {select_cols} FROM wrb_fao90_desc ORDER BY wrb_tax")
        for row in cur.fetchall():
            wrb_tax = row[0]
            rec = {"wrb_tax": wrb_tax}
            for i, s in enumerate(existing):
                rec[s] = {"description": row[1 + i * 2], "management": row[2 + i * 2]}
            out[normalize_soil_name(wrb_tax)] = rec
    return out, existing


def word_diff_html(old, new, side):
    """Render one side of a word-level diff, bolding changed/added/removed words."""
    old_tokens = old.split(" ")
    new_tokens = new.split(" ")
    sm = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(html.escape(" ".join(old_tokens[i1:i2])))
        elif side == "old" and tag in ("delete", "replace"):
            parts.append(f"<b>{html.escape(' '.join(old_tokens[i1:i2]))}</b>")
        elif side == "new" and tag in ("insert", "replace"):
            parts.append(f"<b>{html.escape(' '.join(new_tokens[j1:j2]))}</b>")
    return " ".join(p for p in parts if p)


def compare(translations, db, existing_suffixes):
    """Categorize every (soil, lang, field) and the soil-level set differences."""
    json_keys = set(translations.get("en", {}).keys())
    db_keys = set(db.keys())

    result = {
        "changed": [],  # real wording drift (both sides non-empty)
        "fill": [],  # DB blank, JSON has text
        "json_blank": [],  # DB has text, JSON blank
        "new_lang": [],  # a language whose DB column doesn't exist yet
        "db_only_soils": sorted(db_keys - json_keys),
        "json_only_soils": sorted(json_keys - db_keys),
        "same": 0,
    }

    for soil_key in sorted(json_keys & db_keys):
        for lang in sorted(translations):
            suffix = lang
            t = translations[lang].get(soil_key, {})
            for field in FIELDS:
                json_val = t.get(field)
                nj = normalize(json_val)
                if suffix not in existing_suffixes:
                    if nj:
                        result["new_lang"].append((soil_key, lang, field, json_val))
                    continue
                db_val = db[soil_key][suffix].get(field)
                nd = normalize(db_val)
                if nj == nd:
                    result["same"] += 1
                elif nj and not nd:
                    result["fill"].append((soil_key, lang, field, json_val))
                elif nd and not nj:
                    result["json_blank"].append((soil_key, lang, field, db_val))
                else:
                    result["changed"].append((soil_key, lang, field, db_val, json_val))
    return result


def load_algorithm_soils(conn):
    """Normalized names of every soil the algorithm can emit (HWSD2 fao90_name)."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT fao90_name FROM hwsd2_data WHERE fao90_name IS NOT NULL")
        return {normalize_soil_name(r[0]) for r in cur.fetchall()}


def table_columns(conn):
    """Lowercased column names currently on wrb_fao90_desc."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'wrb_fao90_desc'"
        )
        return {r[0].lower() for r in cur.fetchall()}


def build_write_plan(translations, columns):
    """(alter_statements, delete_statement, [(insert_sql, params), …]) that rebuild
    wrb_fao90_desc from the JSON source of truth: widen/add the description &
    management columns, delete all rows, then insert one row per JSON soil.

    Per language the table stores a translated name (wrb_tax_<suffix>), a
    description and a management column; `wrb_tax` is the lookup key (= English
    name). Rebuild is safe because every JSON `name` exactly matches an existing
    WRB_tax / the algorithm's fao90_name; the caller guards that first.
    """
    langs = sorted(translations)
    # Widen/add only the description & management columns (names are short varchar).
    alters = []
    for lang in langs:
        suffix = lang
        for field in FIELDS:
            col = f"{field}_{suffix}"
            if col in columns:
                # Widen existing varchar(2000) columns — some source strings are longer.
                alters.append(f"ALTER TABLE wrb_fao90_desc ALTER COLUMN {col} TYPE text")
            else:
                alters.append(f"ALTER TABLE wrb_fao90_desc ADD COLUMN IF NOT EXISTS {col} text")

    # Build the INSERT column list + a parallel (json_field, lang) source for each.
    # description/management exist for every language after the ALTERs above; the
    # translated-name column (wrb_tax_<suffix>) is only written where it exists.
    cols, sources = ["wrb_tax"], [("name", "en")]  # wrb_tax (key) = English name
    for lang in langs:
        suffix = lang
        if f"wrb_tax_{suffix}" in columns:
            cols.append(f"wrb_tax_{suffix}")
            sources.append(("name", lang))
        cols.append(f"description_{suffix}")
        sources.append(("description", lang))
        cols.append(f"management_{suffix}")
        sources.append(("management", lang))
    insert_sql = (
        f"INSERT INTO wrb_fao90_desc ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})"
    )

    inserts = []
    for soil_key in sorted(translations["en"]):
        params = [
            translations.get(lang, {}).get(soil_key, {}).get(field) for field, lang in sources
        ]
        inserts.append((insert_sql, params))
    return alters, "DELETE FROM wrb_fao90_desc", inserts


def apply_write(conn, translations, sql_out):
    """Rebuild the table in one transaction; optionally emit the equivalent SQL."""
    alters, delete_sql, inserts = build_write_plan(translations, table_columns(conn))
    sql_lines = [
        "-- Generated by wrb_descriptions_sync.py --write",
        "-- Source of truth: terraso-mobile-client i18n. Rebuilds wrb_fao90_desc.",
        "BEGIN;",
    ]
    with conn.cursor() as cur:
        for stmt in alters:
            cur.execute(stmt)
            sql_lines.append(stmt + ";")
        cur.execute(delete_sql)
        sql_lines.append(delete_sql + ";")
        for sql, params in inserts:
            cur.execute(sql, params)
            sql_lines.append(cur.mogrify(sql, params) + ";")
    conn.commit()
    sql_lines.append("COMMIT;")
    if sql_out:
        Path(sql_out).write_text("\n".join(sql_lines) + "\n")
    return len(alters), len(inserts)


def render_html(result, translations, db):
    css = """
    body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#222}
    h1{font-size:1.4rem} h2{margin-top:2rem;border-bottom:2px solid #ddd;padding-bottom:.3rem}
    table{border-collapse:collapse;width:100%;margin:.5rem 0} td,th{border:1px solid #ddd;padding:.4rem .6rem;vertical-align:top;text-align:left}
    th{background:#f6f8fa} .meta{color:#666;font-size:.85rem;white-space:nowrap}
    b{background:#fff3b0;font-weight:600} .old b{background:#ffd0d0} .new b{background:#c8f0c8}
    .summary td:last-child{text-align:right;font-variant-numeric:tabular-nums}
    code{background:#f0f0f0;padding:0 .2rem;border-radius:3px}
    """
    rows = []
    rows.append("<!doctype html><meta charset=utf-8><title>WRB descriptions diff</title>")
    rows.append(f"<style>{css}</style>")
    rows.append(
        "<h1>WRB soil descriptions — mobile i18n (source of truth) vs <code>wrb_fao90_desc</code></h1>"
    )
    rows.append(
        "<p class=meta>&lt;br&gt; tags and extra whitespace are normalized out before comparison, "
        "so only genuine wording/translation differences are shown.</p>"
    )

    # Summary
    rows.append("<h2>Summary</h2><table class=summary>")
    rows.append(
        f"<tr><td>Soils in JSON / DB / both</td><td>{len(translations.get('en', {}))} / {len(db)} / "
        f"{len(set(translations.get('en', {})) & set(db))}</td></tr>"
    )
    rows.append(f"<tr><td>Identical (after normalization)</td><td>{result['same']}</td></tr>")
    rows.append(f"<tr><td>Changed (wording drift)</td><td>{len(result['changed'])}</td></tr>")
    rows.append(
        f"<tr><td>DB blank, JSON has text (would fill)</td><td>{len(result['fill'])}</td></tr>"
    )
    rows.append(f"<tr><td>JSON blank, DB has text</td><td>{len(result['json_blank'])}</td></tr>")
    rows.append(
        f"<tr><td>New-language entries (ka/uk, no DB column)</td><td>{len(result['new_lang'])}</td></tr>"
    )
    rows.append(
        f"<tr><td>Soils only in DB (no JSON source)</td><td>{len(result['db_only_soils'])}</td></tr>"
    )
    rows.append(
        f"<tr><td>Soils only in JSON (not in DB)</td><td>{len(result['json_only_soils'])}</td></tr>"
    )
    rows.append("</table>")

    # Changed
    rows.append(f"<h2>Changed — wording drift ({len(result['changed'])})</h2>")
    if result["changed"]:
        rows.append(
            "<table><tr><th>Soil</th><th>Lang</th><th>Field</th><th>DB (current)</th><th>JSON (new)</th></tr>"
        )
        for soil_key, lang, field, db_val, json_val in result["changed"]:
            nd, nj = normalize(db_val), normalize(json_val)
            rows.append(
                f"<tr><td class=meta>{html.escape(soil_key)}</td><td class=meta>{lang}</td>"
                f"<td class=meta>{field}</td>"
                f"<td class=old>{word_diff_html(nd, nj, 'old')}</td>"
                f"<td class=new>{word_diff_html(nd, nj, 'new')}</td></tr>"
            )
        rows.append("</table>")

    # Fill
    rows.append(f"<h2>DB blank → would fill from JSON ({len(result['fill'])})</h2>")
    if result["fill"]:
        rows.append("<table><tr><th>Soil</th><th>Lang</th><th>Field</th><th>JSON value</th></tr>")
        for soil_key, lang, field, json_val in result["fill"]:
            rows.append(
                f"<tr><td class=meta>{html.escape(soil_key)}</td><td class=meta>{lang}</td>"
                f"<td class=meta>{field}</td><td>{html.escape(normalize(json_val))}</td></tr>"
            )
        rows.append("</table>")

    # JSON blank
    rows.append(f"<h2>JSON blank, DB has text ({len(result['json_blank'])})</h2>")
    if result["json_blank"]:
        rows.append("<table><tr><th>Soil</th><th>Lang</th><th>Field</th><th>DB value</th></tr>")
        for soil_key, lang, field, db_val in result["json_blank"]:
            rows.append(
                f"<tr><td class=meta>{html.escape(soil_key)}</td><td class=meta>{lang}</td>"
                f"<td class=meta>{field}</td><td>{html.escape(normalize(db_val))}</td></tr>"
            )
        rows.append("</table>")

    # Soil set differences
    rows.append(f"<h2>Soils only in DB — no JSON source ({len(result['db_only_soils'])})</h2>")
    rows.append("<p>" + ", ".join(html.escape(k) for k in result["db_only_soils"]) + "</p>")
    rows.append(f"<h2>Soils only in JSON — not in DB ({len(result['json_only_soils'])})</h2>")
    rows.append("<p>" + ", ".join(html.escape(k) for k in result["json_only_soils"]) + "</p>")

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--translations-dir",
        default="../terraso-mobile-client/dev-client/src/translations",
        help="Directory of mobile-client *.json translation files.",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("SOIL_ID_DATABASE_URL") or os.environ.get("DATABASE_URL"),
        help="Postgres URL for the soil-id DB (defaults to $SOIL_ID_DATABASE_URL).",
    )
    parser.add_argument(
        "--out", default="wrb_diff.html", help="HTML report output path (diff mode)."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rebuild wrb_fao90_desc from the JSON source of truth (delete + reinsert).",
    )
    parser.add_argument(
        "--sql-out",
        default=None,
        help="With --write, optionally also write the applied SQL to this path (for review).",
    )
    args = parser.parse_args()

    if not args.database_url:
        raise SystemExit("No database URL (set SOIL_ID_DATABASE_URL or pass --database-url).")

    translations = load_translations(args.translations_dir)
    if "en" not in translations:
        raise SystemExit(f"No en.json found under {args.translations_dir}")

    if args.write:
        with psycopg.connect(args.database_url, cursor_factory=psycopg.ClientCursor) as conn:
            db, _ = load_db(conn)
            json_keys = set(translations["en"])
            # Guard: rebuilding deletes every row, so refuse if the JSON doesn't cover
            # something the DB or the algorithm still needs (it would lose that description).
            missing = sorted((set(db) | load_algorithm_soils(conn)) - json_keys)
            if missing:
                raise SystemExit(
                    f"Refusing to rebuild: {len(missing)} soil(s) in the DB or algorithm output "
                    f"have no JSON entry and would lose their description: {', '.join(missing)}"
                )
            n_alter, n_insert = apply_write(conn, translations, args.sql_out)
        print(
            f"Rebuilt wrb_fao90_desc from JSON: {n_alter} column op(s), deleted all rows, "
            f"inserted {n_insert} soils across {len(translations)} languages "
            f"({', '.join(translations)})."
        )
        if args.sql_out:
            print(f"Wrote SQL to {args.sql_out}")
        print("Next: regenerate the dump (make dump_soil_id_db) and redistribute it.")
        return

    with psycopg.connect(args.database_url) as conn:
        db, existing = load_db(conn)

    result = compare(translations, db, existing)
    Path(args.out).write_text(render_html(result, translations, db))

    print(f"Languages compared: {', '.join(translations)}")
    print(
        f"Soils — JSON: {len(translations['en'])}, DB: {len(db)}, both: "
        f"{len(set(translations['en']) & set(db))}"
    )
    print(
        f"changed={len(result['changed'])} fill={len(result['fill'])} "
        f"json_blank={len(result['json_blank'])} new_lang={len(result['new_lang'])} "
        f"db_only={len(result['db_only_soils'])} json_only={len(result['json_only_soils'])} "
        f"same={result['same']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
