#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strip the leading H1 (`# ...`) from post bodies.

Rule: the theme renders ONE title from front-matter `title:`. A leading level-1
heading at the very start of the body is therefore a redundant title-position
element (whether an exact dup, a reworded title, or a section like "序言"/"关于MCP"
that sits in the title slot before the first `##`). Remove ONLY the first body
line when it is a single `# ` heading (NOT `##`+). Everything after is kept,
including blockquotes and `##` sections.

--apply to write; default is dry-run.
"""
import glob, re, sys

APPLY = '--apply' in sys.argv
posts = sorted(glob.glob('source/_posts/*.md'))
changed = 0
samples = []

for p in posts:
    t = open(p, encoding='utf-8').read()
    m = re.match(r'^(---\n.*?\n---\n)', t, re.S)
    if not m:
        continue
    fm = m.group(1)
    body = t[len(fm):]
    # find first non-empty line index
    lines = body.split('\n')
    i = 0
    while i < len(lines) and lines[i].strip() == '':
        i += 1
    if i < len(lines) and re.match(r'^#\s+\S', lines[i]) and not lines[i].startswith('## '):
        removed = lines[i]
        # drop that line; also drop a single immediately-following blank line to avoid leading gap
        del lines[i]
        if i < len(lines) and lines[i].strip() == '':
            del lines[i]
        new_body = '\n'.join(lines)
        if APPLY:
            open(p, 'w', encoding='utf-8', newline='\n').write(fm + new_body)
        changed += 1
        if len(samples) < 6:
            samples.append((p.split('/')[-1], removed[:50]))

print(("APPLIED" if APPLY else "DRY-RUN") + f": would strip leading H1 from {changed} posts")
for fn, h1 in samples:
    print(f"  - {fn}: removed '{h1}'")
