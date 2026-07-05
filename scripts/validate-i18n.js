/* eslint-disable no-console */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '..');
const errors = [];
const allowedCategoryPairs = new Set([
  'Foundation Models > Model Mechanics',
  'Foundation Models > Training & Alignment',
  'Agent Systems > Agent Architecture',
  'Agent Systems > Agent Evaluation & Governance',
  'Agent Systems > Agent Infrastructure',
  'Agent Systems > Agent Training',
  'Data Science & Statistics > Statistical Thinking',
  'Data Science & Statistics > Statistical Modeling & Inference',
  'Data Science & Statistics > Probability & Statistical Foundations',
  'Data Science & Statistics > Probabilistic Graphical Models',
  'Data Science & Statistics > Time Series & Spatial Data',
  'Data Science & Statistics > Data Practice',
  'Data Science & Statistics > Applied Machine Learning & AutoML',
  'Data Science & Statistics > Deep Learning',
  'Data Science & Statistics > Forecasting & Simulation',
  'Mathematics > Mathematical Analysis',
  'Mathematics > Algebra & Matrix Theory',
  'Mathematics > Geometry & Topology',
  'Mathematics > Optimization',
  'Programming > Computer Science Fundamentals',
  'Programming > Python',
  'Programming > Web Frontend',
  'Programming > Data & Databases',
  'Programming > Backend Engineering',
  'Programming > R',
  'Programming > C & C++',
  'Work & Society > AI Engineering Workflows',
  'Work & Society > Builder & Product Thinking',
  'Work & Society > Career & Learning',
  'Work & Society > AI & Society',
  'Creative Media & Games > Game AI & Production',
  'Creative Media & Games > Game Design',
  'Creative Media & Games > Generative Media Tools',
  'Fiction & Literature > Speculative Fiction',
  'Fiction & Literature > Science Fiction & Literary Criticism'
]);
const categoryNames = new Set(
  Array.from(allowedCategoryPairs).flatMap(pair => pair.split(' > '))
);
const retiredTags = new Set(['Agents', 'Methodology', 'Society', 'Fiction']);
const tagAliases = new Map([
  ['VLM', 'Vision-Language Models'],
  ['Long-Term Memory', 'Agent Memory'],
  ['Backend', 'Backend Engineering']
]);

function rel(file) {
  return path.relative(root, file).replace(/\\/g, '/');
}

function read(file) {
  return fs.readFileSync(path.join(root, file), 'utf8');
}

function walk(dir, out = []) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

function frontMatter(src) {
  const m = src.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  return m ? m[1] : '';
}

function fmValue(fm, key) {
  const re = new RegExp('^' + key + ':\\s*(.+?)\\s*$', 'm');
  const m = fm.match(re);
  if (!m) return '';
  return m[1].replace(/^['"]|['"]$/g, '').trim();
}

function stripQuotes(s) {
  return s.replace(/^['"]|['"]$/g, '').trim();
}

function fmArray(fm, key) {
  const re = new RegExp('^' + key + ':\\s*(.*?)\\s*$', 'm');
  const m = fm.match(re);
  if (!m) return [];
  const raw = m[1].trim();
  if (!raw || raw === '[]') return [];
  if (raw.startsWith('[') && raw.endsWith(']')) {
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.map(String) : [];
    } catch {
      return raw
        .slice(1, -1)
        .split(',')
        .map(stripQuotes)
        .filter(Boolean);
    }
  }
  return [stripQuotes(raw)];
}

function hasCjk(s) {
  return /[\u4e00-\u9fff]/.test(s);
}

function validateDictionary() {
  const sandbox = { window: {} };
  vm.runInNewContext(read('source/assets/js/i18n.js'), sandbox, { filename: 'i18n.js' });
  const dict = sandbox.window.I18N || {};
  const en = dict.en || {};
  const zh = dict.zh || {};

  const sourceFiles = walk(path.join(root, 'source'))
    .filter(file => /\.(md|html|ejs)$/.test(file))
    .filter(file => !rel(file).startsWith('source/_posts/'))
    .filter(file => !rel(file).startsWith('source/_drafts/'));
  const generatedTemplateFiles = [
    path.join(root, 'scripts/home-sidebar-identity.js')
  ].filter(file => fs.existsSync(file));

  const used = new Map();
  const re = /data-i18n=["']([^"']+)["']/g;
  sourceFiles.concat(generatedTemplateFiles).forEach(file => {
    const src = fs.readFileSync(file, 'utf8');
    let m;
    while ((m = re.exec(src))) {
      if (!used.has(m[1])) used.set(m[1], []);
      used.get(m[1]).push(rel(file));
    }
  });

  for (const [key, files] of used.entries()) {
    if (!Object.prototype.hasOwnProperty.call(en, key)) {
      errors.push(`Missing en i18n key "${key}" used in ${files.join(', ')}`);
    }
    if (!Object.prototype.hasOwnProperty.call(zh, key)) {
      errors.push(`Missing zh i18n key "${key}" used in ${files.join(', ')}`);
    }
  }
}

function validatePosts() {
  const postsDir = path.join(root, 'source/_posts');
  const posts = fs.readdirSync(postsDir).filter(name => name.endsWith('.md'));
  posts.forEach(name => {
    const fm = frontMatter(fs.readFileSync(path.join(postsDir, name), 'utf8'));
    if (!fmValue(fm, 'title_en')) errors.push(`${name} is missing title_en`);
    if (!fmValue(fm, 'excerpt_en')) errors.push(`${name} is missing excerpt_en`);

    const categories = fmArray(fm, 'categories');
    if (categories.length !== 2) {
      errors.push(`${name} must use exactly two category levels`);
    } else if (!allowedCategoryPairs.has(categories.join(' > '))) {
      errors.push(`${name} has unknown category path: ${categories.join(' > ')}`);
    }

    const tags = fmArray(fm, 'tags');
    if (!tags.length) errors.push(`${name} must have at least one tag`);
    const seenTags = new Set();
    tags.forEach(tag => {
      if (seenTags.has(tag)) errors.push(`${name} has duplicate tag "${tag}"`);
      seenTags.add(tag);
      if (retiredTags.has(tag)) errors.push(`${name} uses retired broad tag "${tag}"`);
      if (tagAliases.has(tag)) {
        errors.push(`${name} uses alias tag "${tag}"; use "${tagAliases.get(tag)}"`);
      }
    });
  });
}

function validateTaxonomyConfig() {
  const config = read('_config.yml');
  categoryNames.forEach(name => {
    if (!config.includes(`"${name}":`)) {
      errors.push(`_config.yml category_map is missing "${name}"`);
    }
  });

  const redefine = read('_config.redefine.yml');
  if (!/\n  tags:\r?\n    enable:\s*true\r?\n    limit:\s*0\b/.test(redefine)) {
    errors.push('_config.redefine.yml home.tags.limit must be 0 so tags are not truncated');
  }
}

function validateUiCjk() {
  const checked = [
    '_config.yml',
    '_config.redefine.yml',
    'source/me/index.md',
    'source/projects/index.md',
    'source/footprints/index.md',
    'source/cv/index.md',
    'source/friends/index.md',
    'source/categories/index.md',
    'source/tags/index.md'
  ];

  checked.forEach(file => {
    const src = read(file);
    if (hasCjk(src)) errors.push(`${file} contains CJK text in English-default UI source`);
  });
}

function validatePostI18nGenerator() {
  const src = read('scripts/archive-and-i18n.js');
  if (!src.includes('encodeURI(decoded)')) {
    errors.push('post_i18n_map must emit encoded URL variants for CJK permalinks');
  }
  if (!src.includes('title_zh')) {
    errors.push('post_i18n_map must emit title_zh so post titles can be restored in zh mode');
  }
  if (!src.includes('excerpt_zh')) {
    errors.push('post_i18n_map must emit excerpt_zh so post excerpts can be restored in zh mode');
  }
  if (!src.includes('before_post_render') || !src.includes('data.title = data.title_en')) {
    errors.push('posts must be rendered with title_en as the English-default article title');
  }
}

validateDictionary();
validatePosts();
validateTaxonomyConfig();
validateUiCjk();
validatePostI18nGenerator();

if (errors.length) {
  console.error('i18n validation failed:');
  errors.forEach(err => console.error(' - ' + err));
  process.exit(1);
}

console.log('i18n validation passed');
