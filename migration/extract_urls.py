#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract ground-truth post URLs from the committed Jekyll _site/ directory.

Output: migration/url_map.json  ->  { "<key>": "/blog/Y/M/D/<slug>/" }
where <key> is "YYYY-MM-DD|<jekyll-stem>" derived from each _site dir,
plus a reverse listing for auditing.

This MUST run before _site/ is deleted. _site is the only authoritative
source for Jekyll's slug transforms (space->-, fullwidth-colon->-, case kept,
CJK literal, comma literal).
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_BLOG = os.path.join(ROOT, "_site", "blog")

# Match _site/blog/YYYY/MM/DD/<slug>/index.html
INDEX_RE = re.compile(r"^(\d{4})[\\/](\d{2})[\\/](\d{2})[\\/](.+)$")


def main():
    if not os.path.isdir(SITE_BLOG):
        print("ERROR: _site/blog not found at", SITE_BLOG, file=sys.stderr)
        sys.exit(1)

    url_map = {}      # "Y-M-D|slug" -> url   (slug = jekyll dir name)
    by_date = {}      # "Y-M-D" -> [slug, ...]
    urls = []

    for dirpath, dirnames, filenames in os.walk(SITE_BLOG):
        if "index.html" not in filenames:
            continue
        rel = os.path.relpath(dirpath, SITE_BLOG)
        rel = rel.replace("\\", "/")
        if rel == ".":
            continue  # /blog/ itself
        m = INDEX_RE.match(rel)
        if not m:
            # e.g. blog/series/  -> skip non Y/M/D/slug paths
            continue
        y, mo, d, slugpath = m.group(1), m.group(2), m.group(3), m.group(4)
        # slugpath should be a single segment (the post slug)
        if "/" in slugpath:
            # nested deeper than expected; skip (none expected)
            print("WARN nested:", rel, file=sys.stderr)
            continue
        slug = slugpath
        url = "/blog/{}/{}/{}/{}/".format(y, mo, d, slug)
        date = "{}-{}-{}".format(y, mo, d)
        key = "{}|{}".format(date, slug)
        url_map[key] = url
        by_date.setdefault(date, []).append(slug)
        urls.append(url)

    out = {
        "count": len(urls),
        "url_map": url_map,
        "by_date": by_date,
        "urls": sorted(urls),
    }
    out_path = os.path.join(ROOT, "migration", "url_map.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Wrote", out_path)
    print("Total _site post URLs:", len(urls))
    # Show the non-ASCII ones for sanity
    print("\nNon-ASCII URLs:")
    for u in sorted(urls):
        if any(ord(c) > 127 for c in u):
            print("  ", u)


if __name__ == "__main__":
    main()
