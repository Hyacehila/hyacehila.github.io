/* eslint-disable no-console */
'use strict';

// One-time editorial pass for B3. Links are written into article prose so the
// generated site contains ordinary contextual anchors, not a recommendation
// widget or footer module.
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

const root = path.resolve(__dirname, '..');
const postsDir = path.join(root, 'source', '_posts');

function parseSource(file) {
  const source = fs.readFileSync(file, 'utf8');
  const match = source.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!match) return null;
  return { source, front: yaml.load(match[1]) || {}, body: source.slice(match[0].length) };
}

function asList(value) {
  if (Array.isArray(value)) return value.map(String);
  return value == null ? [] : [String(value)];
}

function internalLinkCount(body) {
  return (body.match(/\]\(\/(?:en\/)?blog\//g) || []).length;
}

function score(source, target) {
  if (source === target) return -1;
  const sourceCategories = asList(source.front.categories);
  const targetCategories = asList(target.front.categories);
  const sourceTags = new Set(asList(source.front.tags));
  const sharedTags = asList(target.front.tags).filter(tag => sourceTags.has(tag)).length;
  const sameLeaf = sourceCategories[1] && sourceCategories[1] === targetCategories[1];
  const sameRoot = sourceCategories[0] && sourceCategories[0] === targetCategories[0];
  return sharedTags * 4 + (sameLeaf ? 8 : 0) + (sameRoot ? 2 : 0);
}

function insertAfterOpeningBlock(body, paragraph) {
  const match = body.match(/^[\s\S]*?\n\n/);
  if (!match) return `${paragraph}\n\n${body}`;
  return `${match[0]}${paragraph}\n\n${body.slice(match[0].length)}`;
}

const files = fs.readdirSync(postsDir).filter(name => name.endsWith('.md')).sort();
const posts = files.map(name => {
  const parsed = parseSource(path.join(postsDir, name));
  if (!parsed || !parsed.front.permalink) throw new Error(`Missing front matter/permalink: ${name}`);
  return { ...parsed, name, url: String(parsed.front.permalink), title: String(parsed.front.title || name) };
});

let changed = 0;
let added = 0;
for (const post of posts) {
  const count = internalLinkCount(post.body);
  if (count >= 2) continue;
  const targets = posts
    .map(target => ({ target, value: score(post, target) }))
    .filter(item => item.value >= 0)
    .sort((a, b) => b.value - a.value || a.target.name.localeCompare(b.target.name))
    .slice(0, 2 - count)
    .map(item => item.target);
  if (targets.length < 2 - count) throw new Error(`Could not find contextual targets for ${post.name}`);
  const links = targets.map(target => `[${target.title}](${target.url})`);
  const sentence = links.length === 1
    ? `这篇文章中的问题也可以和${links[0]}放在一起阅读，以比较相近的概念如何在不同语境中展开。`
    : `这篇文章中的问题也可以和${links[0]}、${links[1]}放在一起阅读，以比较相近的概念如何在不同语境中展开。`;
  const nextBody = insertAfterOpeningBlock(post.body, sentence);
  const nextSource = post.source.slice(0, post.source.length - post.body.length) + nextBody;
  fs.writeFileSync(path.join(postsDir, post.name), nextSource, 'utf8');
  changed += 1;
  added += links.length;
  console.log(`[context-links] ${post.name}: +${links.length}`);
}

console.log(`[context-links] updated ${changed} posts; added ${added} contextual links.`);
