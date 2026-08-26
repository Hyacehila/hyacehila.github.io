#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

const root = path.resolve(__dirname, '..');
const config = yaml.load(fs.readFileSync(path.join(root, '_config.yml'), 'utf8')) || {};
const aliases = config.seo?.tag_aliases || {};
const postDir = path.join(root, 'source', '_posts');
let changed = 0;

for (const name of fs.readdirSync(postDir).filter(file => file.endsWith('.md'))) {
  const file = path.join(postDir, name);
  const source = fs.readFileSync(file, 'utf8');
  const frontMatterEnd = source.indexOf('\n---', 4);
  if (!source.startsWith('---') || frontMatterEnd === -1) continue;

  const head = source.slice(0, frontMatterEnd);
  const match = head.match(/^tags:\s*(\[[^\r\n]*\])\s*$/m);
  if (!match) continue;

  let tags;
  try {
    tags = JSON.parse(match[1]);
  } catch {
    continue;
  }

  const normalized = Array.from(new Set(tags.map(tag => aliases[tag] || tag)));
  const serialized = JSON.stringify(normalized).replace(/","/g, '", "');
  const nextLine = `tags: ${serialized}`;
  const nextHead = head.replace(match[0], nextLine);
  if (nextHead === head) continue;

  fs.writeFileSync(file, nextHead + source.slice(frontMatterEnd), 'utf8');
  changed++;
}

console.log(`[tags] normalized ${changed} post front matters with ${Object.keys(aliases).length} aliases.`);
