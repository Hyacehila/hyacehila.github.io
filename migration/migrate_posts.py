#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Migrate Jekyll _posts -> Hexo source/_posts (+ one draft).

For each _posts/*.md:
  Front matter (line-based, preserves original quoting/order of kept fields):
    - drop: layout, series, featured, published, (old) permalink
    - math: true  -> mathjax: true
    - math: false -> dropped
    - keep: title, title_en, date, categories, tags, author, excerpt, excerpt_en
    - inject canonical `permalink: '<from _site>'` (published posts only)
  Body:
    - {{ site.baseurl }}  -> '' (yields /assets/images/...)
    - {% post_url NAME %}  -> target post's permalink
    - {% raw %}..{% endraw %} kept verbatim (Hexo supports it)

Routing:
    - published: false  -> source/_drafts/  (matches Jekyll: not shown, kept in repo)
    - everything else   -> source/_posts/

After writing, scans each output for residual {{ / {% OUTSIDE raw blocks.
"""
import json
import os
import re
import glob
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_POSTS = os.path.join(ROOT, "_posts")
OUT_POSTS = os.path.join(ROOT, "source", "_posts")
OUT_DRAFTS = os.path.join(ROOT, "source", "_drafts")

PERM = json.load(open(os.path.join(ROOT, "migration", "post_permalinks.json"), encoding="utf-8"))

DROP_KEYS = ("layout:", "series:", "featured:", "published:", "permalink:")
POST_URL_RE = re.compile(r"{%\s*post_url\s+([^\s%]+)\s*%}")
SITE_BASEURL_RE = re.compile(r"{{\s*site\.baseurl\s*}}")
RAW_OPEN = re.compile(r"{%\s*raw\s*%}")
RAW_CLOSE = re.compile(r"{%\s*endraw\s*%}")


def split_front_matter(text):
    if not text.startswith("---"):
        return None, text
    # find the closing --- on its own line
    m = re.search(r"\n---\s*\n", text)
    if not m:
        return None, text
    fm = text[4:m.start() + 1]  # between first '---\n' and the closing
    # more robust: first line is '---'
    lines = text.split("\n")
    assert lines[0].strip() == "---"
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None, text
    fm_lines = lines[1:end]
    body = "\n".join(lines[end + 1:])
    return fm_lines, body


def transform_front_matter(fm_lines, permalink, is_draft):
    out = []
    published_false = False
    for ln in fm_lines:
        stripped = ln.strip()
        low = stripped.lower()
        if low.startswith("published:"):
            if "false" in low:
                published_false = True
            continue  # drop published key entirely
        if any(stripped.startswith(k) for k in DROP_KEYS if k != "published:"):
            continue
        if stripped.startswith("math:"):
            if "true" in low:
                out.append("mathjax: true")
            # math: false -> drop
            continue
        out.append(ln)
    # inject permalink for published posts (not drafts)
    if permalink and not is_draft:
        out.append("permalink: '{}'".format(permalink))
    return out, published_false


def transform_body(body, fn):
    # 1. site.baseurl -> ''
    body = SITE_BASEURL_RE.sub("", body)

    # 2. post_url -> target permalink (skip occurrences inside raw blocks)
    def repl_post_url(m):
        name = m.group(1)
        target = PERM.get(name + ".md")
        if target is None:
            print("  WARN: post_url target not found:", name, "in", fn, file=sys.stderr)
            return m.group(0)
        return target
    # We must not rewrite inside {% raw %}..{% endraw %}. Re0HF-02's raw block
    # contains no post_url, so a global sub is safe; but guard anyway by
    # splitting on raw regions.
    body = sub_outside_raw(body, POST_URL_RE, repl_post_url)
    return body


def sub_outside_raw(text, pattern, repl):
    """Apply regex sub only outside {% raw %}..{% endraw %} regions."""
    parts = []
    idx = 0
    while True:
        ro = RAW_OPEN.search(text, idx)
        if not ro:
            parts.append(pattern.sub(repl, text[idx:]))
            break
        # before raw: substitute
        parts.append(pattern.sub(repl, text[idx:ro.start()]))
        rc = RAW_CLOSE.search(text, ro.end())
        if not rc:
            # unbalanced; append rest verbatim
            parts.append(text[ro.start():])
            break
        # raw region verbatim
        parts.append(text[ro.start():rc.end()])
        idx = rc.end()
    return "".join(parts)


def scan_residual(text):
    """Return list of residual {{ or {% occurrences outside raw blocks."""
    hits = []
    idx = 0
    pos = 0
    # build list of raw region spans
    spans = []
    i = 0
    while True:
        ro = RAW_OPEN.search(text, i)
        if not ro:
            break
        rc = RAW_CLOSE.search(text, ro.end())
        if not rc:
            spans.append((ro.start(), len(text)))
            break
        spans.append((ro.start(), rc.end()))
        i = rc.end()

    def in_raw(p):
        return any(a <= p < b for a, b in spans)

    for m in re.finditer(r"{{|{%", text):
        if not in_raw(m.start()):
            # context
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.start())
            hits.append(text[line_start:line_end if line_end != -1 else None].strip())
    return hits


def main():
    os.makedirs(OUT_POSTS, exist_ok=True)
    os.makedirs(OUT_DRAFTS, exist_ok=True)

    files = sorted(glob.glob(os.path.join(SRC_POSTS, "*.md")))
    n_posts = n_drafts = 0
    residuals = {}

    for path in files:
        fn = os.path.basename(path)
        text = open(path, encoding="utf-8").read()
        fm_lines, body = split_front_matter(text)
        if fm_lines is None:
            print("  ERROR: no front matter:", fn, file=sys.stderr)
            continue

        # detect published:false BEFORE transform to decide routing
        is_draft = any(l.strip().lower().startswith("published:") and "false" in l.lower()
                       for l in fm_lines)
        permalink = PERM.get(fn)

        new_fm, _ = transform_front_matter(fm_lines, permalink, is_draft)
        new_body = transform_body(body, fn)

        out_text = "---\n" + "\n".join(new_fm) + "\n---\n" + new_body
        if not out_text.endswith("\n"):
            out_text += "\n"

        dest_dir = OUT_DRAFTS if is_draft else OUT_POSTS
        with open(os.path.join(dest_dir, fn), "w", encoding="utf-8", newline="\n") as f:
            f.write(out_text)

        if is_draft:
            n_drafts += 1
            print("  DRAFT:", fn)
        else:
            n_posts += 1

        res = scan_residual(new_body)
        if res:
            residuals[fn] = res

    print("\nWrote {} posts -> source/_posts/, {} draft(s) -> source/_drafts/".format(n_posts, n_drafts))
    print("\nResidual {{ / {% outside raw blocks (should be empty):")
    if not residuals:
        print("  (none)")
    for fn, hits in residuals.items():
        print("  ", fn)
        for h in hits[:5]:
            try:
                print("      ", h)
            except UnicodeEncodeError:
                print("       <unicode line>")


if __name__ == "__main__":
    main()
