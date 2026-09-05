'use strict';

const fs = require('fs');
const path = require('path');
const assert = require('node:assert/strict');
const yaml = require('js-yaml');
const cheerio = require('cheerio');
const { frontMatter } = require('../scripts/validate-i18n');
const { taxonomyRedirects } = require('./taxonomy-redirects');

const root = path.resolve(__dirname, '..');
const publicDir = path.join(root, 'public');
const config = yaml.load(fs.readFileSync(path.join(root, '_config.yml'), 'utf8'));
const fileFor = route => path.join(publicDir, decodeURI(route).replace(/^\//, ''), 'index.html');

function walk(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const file = path.join(dir, entry.name);
    return entry.isDirectory() ? walk(file) : [file];
  });
}

const unpublished = new Set();
const categories = new Map();
for (const language of ['source', 'source_en']) {
  const posts = path.join(root, language, '_posts');
  for (const name of fs.readdirSync(posts).filter(name => name.endsWith('.md'))) {
    const fm = frontMatter(fs.readFileSync(path.join(posts, name), 'utf8'));
    if (language === 'source' && fm.categories?.[0] === 'Programming') {
      const category = fm.categories[1];
      if (!categories.has(category)) categories.set(category, new Set());
      categories.get(category).add(`/en${fm.permalink}`);
    }
  }
  // Directory membership controls publication; drafts may lack front matter.
  for (const file of walk(path.join(root, language, '_drafts')).filter(file => file.endsWith('.md'))) {
    const fm = frontMatter(fs.readFileSync(file, 'utf8'));
    if (!fm.permalink) continue;
    const route = `${language === 'source_en' ? '/en' : ''}${fm.permalink}`;
    assert.ok(!fs.existsSync(fileFor(route)), `Draft page exists: ${route}`);
    unpublished.add(route);
    if (fm.title) unpublished.add(fm.title);
  }
}

const output = walk(publicDir);
for (const file of output.filter(file => /\.(?:html|json|xml)$/.test(file))) {
  const source = fs.readFileSync(file, 'utf8');
  for (const value of unpublished) {
    assert.ok(!source.includes(value), `${path.relative(publicDir, file)} contains unpublished title/URL: ${value}`);
  }
}

for (const [category, expected] of categories) {
  const dir = path.join(publicDir, 'categories', 'programming', config.category_map[category]);
  assert.ok(fs.existsSync(path.join(dir, 'index.html')), `Missing category: ${category}`);
  const actual = new Set();
  for (const file of walk(dir).filter(file => file.endsWith('.html'))) {
    const $ = cheerio.load(fs.readFileSync(file, 'utf8'));
    $('.category-post-list .article-item a[href]').each((_, element) => {
      actual.add($(element).attr('href'));
    });
  }
  assert.deepEqual(actual, expected, `Category membership mismatch: ${category}`);
}

const sitemapText = output.filter(file => /sitemap[^/\\]*\.xml$/.test(file))
  .map(file => fs.readFileSync(file, 'utf8')).join('\n');
const base = config.url.replace(/\/$/, '');
for (const { from, to } of taxonomyRedirects(config)) {
  assert.ok(fs.existsSync(fileFor(to)), `Missing redirect target: ${to}`);
  const $ = cheerio.load(fs.readFileSync(fileFor(from), 'utf8'));
  assert.equal($('link[rel="canonical"]').attr('href'), `${base}${to}`, from);
  assert.equal($('meta[http-equiv="refresh"]').attr('content'), `0;url=${base}${to}`, from);
  assert.equal($('meta[name="robots"]').attr('content'), 'noindex,follow', from);
  assert.equal($('a').attr('href'), `${base}${to}`, from);
  assert.ok(!sitemapText.includes(`${base}${from}`), `Redirect is indexed: ${from}`);
}

console.log('Publication validation passed:', JSON.stringify({
  categories: Object.fromEntries(Array.from(categories, ([name, posts]) => [name, posts.size])),
  redirects: taxonomyRedirects(config).length
}));
