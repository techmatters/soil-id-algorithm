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
Render a Soil ID explain trace (JSON) into a self-contained HTML report.

The renderer itself lives in ``soil_id.render_explain.render_html``; this script
is a thin CLI wrapper that loads a trace and previews the report.

Two ways to invoke it:

    # A local file — either a bare explain trace or a full site export
    # ({"sites": [...]}); the wrapper is unwrapped automatically. HTML to stdout.
    python -m scripts.render_soil_explain trace.json > report.html

    # An export URL (without ?explain=true — it's added for you). The trace is
    # fetched, rendered to a temp .html file, and opened in your browser.
    python -m scripts.render_soil_explain \\
        https://api.staging.terraso.net/export/token/site/<id>/<name>.html
"""

import argparse
import json
import re
import sys
import tempfile
import urllib.parse
import urllib.request
import webbrowser

from soil_id.render_explain import render_html


def is_url(source):
    return source.startswith(("http://", "https://"))


def explain_url(url):
    """Normalize an export URL to the trace form: force a .json extension (so we
    get the explain trace rather than rendered HTML/CSV) and add ?explain=true."""
    parts = urllib.parse.urlsplit(url)
    head, _, last = parts.path.rpartition("/")
    stem = last.rsplit(".", 1)[0] if "." in last else last
    path = f"{head}/{stem}.json"
    query = dict(urllib.parse.parse_qsl(parts.query))
    query["explain"] = "true"
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, path, urllib.parse.urlencode(query), "")
    )


def extract_trace(data):
    """Accept either a bare explain trace or a full site export
    ({"sites": [...]}), returning (trace, label). For an export we take the first
    site's soilIdExplanation, mirroring what the app requests."""
    if isinstance(data, dict) and "candidates" not in data and "sites" in data:
        sites = data.get("sites") or []
        if not sites:
            sys.exit("export contains no sites")
        if len(sites) > 1:
            print(f"note: export has {len(sites)} sites; rendering the first", file=sys.stderr)
        site = sites[0]
        trace = site.get("soilIdExplanation")
        if not trace:
            sys.exit("site has no soilIdExplanation — was ?explain=true set?")
        return trace, site.get("name") or site.get("id") or "site"
    return data, None


def load_source(source):
    """Return (trace, label) from a URL or a local file path."""
    if is_url(source):
        url = explain_url(source)
        print(f"fetching {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as resp:
            data = json.load(resp)
    else:
        with open(source) as f:
            data = json.load(f)
    return extract_trace(data)


def open_in_browser(html_text, label):
    """Write the report to a temp .html file and open it. Uses the system temp
    dir (tempfile honors $TMPDIR — /tmp or /var/folders/... on macOS), with the
    site name in the filename so it's identifiable."""
    slug = re.sub(r"[^\w.-]+", "_", label or "soil-id").strip("_") or "soil-id"
    fd, path = tempfile.mkstemp(prefix=f"soil-explain-{slug}-", suffix=".html")
    with open(fd, "w") as f:
        f.write(html_text)
    print(f"wrote {path}", file=sys.stderr)
    webbrowser.open(f"file://{path}")
    return path


def main():
    ap = argparse.ArgumentParser("render_soil_explain")
    ap.add_argument(
        "source",
        help="explain-trace JSON file, a site-export JSON file, or an export URL",
    )
    args = ap.parse_args()
    trace, label = load_source(args.source)
    report = render_html(trace)
    if is_url(args.source):
        open_in_browser(report, label)
    else:
        print(report)


if __name__ == "__main__":
    main()
