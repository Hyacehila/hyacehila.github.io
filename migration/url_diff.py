#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""URL diff: compare generated public/blog/**/index.html against the 103
ground-truth Jekyll URLs (from _site, published posts only).

The 1 published-false post (world-models) is a draft now and must NOT appear.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_BLOG = os.path.join(ROOT, "public", "blog")
GROUND = json.load(open(os.path.join(ROOT, "migration", "url_map.json"), encoding="utf-8"))

ground_urls = set(GROUND["urls"])  # 103


def collect_public_urls():
    urls = set()
    for dirpath, _, filenames in os.walk(PUBLIC_BLOG):
        if "index.html" in filenames:
            rel = os.path.relpath(dirpath, os.path.join(ROOT, "public"))
            rel = "/" + rel.replace("\\", "/") + "/"
            urls.add(rel)
    return urls


def main():
    pub = collect_public_urls()
    print("Ground-truth URLs (_site, published): {}".format(len(ground_urls)))
    print("Generated public/blog URLs:           {}".format(len(pub)))

    missing = ground_urls - pub   # in Jekyll, not generated -> BROKEN LINK
    extra = pub - ground_urls     # generated, not in Jekyll -> new/unexpected

    print("\n[MISSING] in ground-truth but NOT generated (would break old links): {}".format(len(missing)))
    for u in sorted(missing):
        print("   ", u)

    print("\n[EXTRA] generated but not in _site (new posts / index pages): {}".format(len(extra)))
    for u in sorted(extra):
        print("   ", u)

    # Specifically verify the 5 non-ASCII
    print("\n[NON-ASCII] byte-exact check:")
    nonascii = [u for u in sorted(ground_urls) if any(ord(c) > 127 for c in u)]
    for u in nonascii:
        status = "OK " if u in pub else "MISSING"
        print("   {}  {}".format(status, u))

    ok = (len(missing) == 0)
    print("\n==> URL DIFF {}".format("PASS (no old URL broken)" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
