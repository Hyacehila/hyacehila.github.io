#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Migrate Jekyll _drafts -> Hexo source/_drafts.

Same front-matter cleanup as posts (drop layout/series/featured/published,
math->mathjax) and the same body Liquid fixes. NO permalink injected (drafts
are unpublished). post_url targets resolve against the published permalink map;
unknown targets fall back to the canonical /blog/Y/M/D/<stem>/ pattern derived
from the referenced filename.
"""
import json
import os
import re
import glob
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "_drafts")
OUT = os.path.join(ROOT, "source", "_drafts")

PERM = json.load(open(os.path.join(ROOT, "migration", "post_permalinks.json"), encoding="utf-8"))

DROP_KEYS = ("layout:", "series:", "featured:", "published:", "permalink:")
POST_URL_RE = re.compile(r"{%\s*post_url\s+([^\s%]+)\s*%}")
SITE_BASEURL_RE = re.compile(r"{{\s*site\.baseurl\s*}}")
RAW_OPEN = re.compile(r"{%\s*raw\s*%}")
RAW_CLOSE = re.compile(r"{%\s*endraw\s*%}")
DATE_PREFIX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.+)$")


def split_front_matter(text):
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, text
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None, text
    return lines[1:end], "\n".join(lines[end + 1:])


def transform_fm(fm_lines):
    out = []
    for ln in fm_lines:
        s = ln.strip()
        low = s.lower()
        if any(s.startswith(k) for k in DROP_KEYS):
            continue
        if s.startswith("math:"):
            if "true" in low:
                out.append("mathjax: true")
            continue
        out.append(ln)
    return out


def fallback_permalink(name):
    m = DATE_PREFIX.match(name)
    if not m:
        return None
    y, mo, d, stem = m.groups()
    stem = stem.replace(" ", "-").replace("：", "-")
    return "/blog/{}/{}/{}/{}/".format(y, mo, d, stem)


def sub_outside_raw(text, pattern, repl):
    parts, idx = [], 0
    while True:
        ro = RAW_OPEN.search(text, idx)
        if not ro:
            parts.append(pattern.sub(repl, text[idx:]))
            break
        parts.append(pattern.sub(repl, text[idx:ro.start()]))
        rc = RAW_CLOSE.search(text, ro.end())
        if not rc:
            parts.append(text[ro.start():])
            break
        parts.append(text[ro.start():rc.end()])
        idx = rc.end()
    return "".join(parts)


def transform_body(body, fn):
    body = SITE_BASEURL_RE.sub("", body)

    def repl(m):
        name = m.group(1)
        t = PERM.get(name + ".md") or fallback_permalink(name)
        if t is None:
            print("  WARN unresolved post_url:", name, "in", fn, file=sys.stderr)
            return m.group(0)
        return t
    return sub_outside_raw(body, POST_URL_RE, repl)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.md")))
    n = 0
    for path in files:
        fn = os.path.basename(path)
        text = open(path, encoding="utf-8").read()
        fm, body = split_front_matter(text)
        if fm is None:
            print("  ERROR no front matter:", fn, file=sys.stderr)
            continue
        new_fm = transform_fm(fm)
        new_body = transform_body(body, fn)
        out_text = "---\n" + "\n".join(new_fm) + "\n---\n" + new_body
        if not out_text.endswith("\n"):
            out_text += "\n"
        with open(os.path.join(OUT, fn), "w", encoding="utf-8", newline="\n") as f:
            f.write(out_text)
        n += 1
    print("Migrated {} drafts -> source/_drafts/".format(n))


if __name__ == "__main__":
    main()
