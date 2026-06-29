#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pair every _posts/*.md to its exact Jekyll URL.

Strategy:
  1. Load migration/url_map.json (ground truth from _site).
  2. For each post: read front-matter date (Y-M-D) + compute the Jekyll slug
     from the filename stem (date prefix stripped).
  3. Jekyll's default slug transform (Stevenson/`jekyll-titlecase`? No — default
     `slugify: default`): we DO NOT reimplement transliteration. Instead we
     match by (date, transformed-stem) against the _site map. For ASCII/space/
     fullwidth-colon cases the transform is: spaces -> '-', fullwidth ':' -> '-'.
     Because _site is authoritative, we resolve ambiguity by:
       a) exact stem match in map, else
       b) apply minimal transform (space->-, '：'->-, collapse multiple '-'),
       c) if still unmatched and the date has exactly ONE _site slug, use it.
  4. Posts with an explicit front-matter `permalink:` already are honored.
  5. The 1 post newer than _site (no map entry) gets its permalink built
     directly from the (already ASCII) stem.

Output: migration/post_permalinks.json -> { "<filename>": "<permalink>" }
Also prints any UNMATCHED posts (must be 0 before migrating).
"""
import json
import os
import re
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTS = os.path.join(ROOT, "_posts")

DATE_PREFIX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.+)$")
FM_DATE = re.compile(r"^date:\s*(\d{4})-(\d{2})-(\d{2})", re.M)
FM_PERMALINK = re.compile(r"^permalink:\s*(.+?)\s*$", re.M)


def read_front_matter(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.startswith("---"):
        return text, ""
    end = text.find("\n---", 3)
    if end == -1:
        return text, ""
    fm = text[3:end]
    return text, fm


def jekyll_slugify_stem(stem):
    """Approximate Jekyll default slugify for matching against _site.
    Jekyll default mode replaces non-alphanumeric ASCII runs with '-',
    but KEEPS unicode (CJK) and lowercases ASCII? NO. Observed from _site:
    case is PRESERVED. So Jekyll here used `slugify: none`-ish for CJK but
    replaced spaces and fullwidth colon with '-'. We replicate observed:
      - space -> '-'
      - fullwidth colon '：' -> '-'
      - keep everything else literal (commas, CJK, ascii case)
      - collapse consecutive '-'
    """
    s = stem
    s = s.replace(" ", "-")
    s = s.replace("：", "-")  # fullwidth colon
    s = re.sub(r"-{2,}", "-", s)
    return s


def main():
    m = json.load(open(os.path.join(ROOT, "migration", "url_map.json"), encoding="utf-8"))
    url_map = m["url_map"]      # "date|slug" -> url
    by_date = m["by_date"]      # "date" -> [slug,...]

    result = {}
    unmatched = []
    files = sorted(glob.glob(os.path.join(POSTS, "*.md")))
    print("Total post files:", len(files))

    for path in files:
        fn = os.path.basename(path)
        text, fm = read_front_matter(path)

        # explicit permalink wins
        pm = FM_PERMALINK.search(fm)
        explicit = pm.group(1).strip().strip("'\"") if pm else None

        dm = DATE_PREFIX.match(fn[:-3])  # strip .md
        if not dm:
            unmatched.append((fn, "no date prefix in filename"))
            continue
        fy, fmo, fday, stem = dm.group(1), dm.group(2), dm.group(3), dm.group(4)

        # prefer front-matter date for the URL path (Jekyll uses post date)
        fmd = FM_DATE.search(fm)
        if fmd:
            y, mo, day = fmd.group(1), fmd.group(2), fmd.group(3)
        else:
            y, mo, day = fy, fmo, fday
        date = "{}-{}-{}".format(y, mo, day)

        if explicit:
            result[fn] = explicit
            continue

        # try exact stem, then transformed stem
        candidates = [stem, jekyll_slugify_stem(stem)]
        url = None
        for c in candidates:
            k = "{}|{}".format(date, c)
            if k in url_map:
                url = url_map[k]
                break
        # also try filename-date if front-matter date differs
        if url is None and (fy, fmo, fday) != (y, mo, day):
            fdate = "{}-{}-{}".format(fy, fmo, fday)
            for c in candidates:
                k = "{}|{}".format(fdate, c)
                if k in url_map:
                    url = url_map[k]
                    break
        # fallback: single slug on that date
        if url is None:
            slugs = by_date.get(date) or by_date.get("{}-{}-{}".format(fy, fmo, fday))
            if slugs and len(slugs) == 1:
                url = "/blog/{}/{}/{}/{}/".format(*date.split("-"), slugs[0])

        if url is None:
            # post newer than _site: build from stem directly (ASCII expected)
            built = "/blog/{}/{}/{}/{}/".format(y, mo, day, jekyll_slugify_stem(stem))
            result[fn] = built
            unmatched.append((fn, "NOT in _site -> built {}".format(built)))
            continue

        result[fn] = url

    out_path = os.path.join(ROOT, "migration", "post_permalinks.json")
    json.dump(result, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("Wrote", out_path, "with", len(result), "entries")

    print("\nNotes / not-in-_site (built directly):")
    for fn, why in unmatched:
        try:
            print("  ", fn, "->", why)
        except UnicodeEncodeError:
            print("  <unicode filename> ->", why)

    # Hard check: every post must have a permalink
    missing = [f for f in (os.path.basename(p) for p in files) if f not in result]
    print("\nPosts without resolved permalink:", len(missing))
    for f in missing:
        print("  MISSING:", f)


if __name__ == "__main__":
    main()
