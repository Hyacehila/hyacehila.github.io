/* eslint-disable no-console */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const yaml = require('js-yaml');

const root = path.resolve(__dirname, '..');
const errors = [];
const allowedCategoryPairs = new Set([
  'Foundation Models > Model Mechanics',
  'Foundation Models > Training & Alignment',
  'Agent Systems > Agent Architecture',
  'Agent Systems > Agent Evaluation & Governance',
  'Agent Systems > Agent Infrastructure',
  'Agent Systems > Agent Training',
  'Data Science > Statistical Thinking',
  'Data Science > Statistical Modeling & Inference',
  'Data Science > Probability & Statistical Foundations',
  'Data Science > Time Series & Spatial Data',
  'Data Science > Data Practice',
  'Economics & Finance > Economic Foundations',
  'Economics & Finance > Financial Markets',
  'Economics & Finance > Economics & Finance Notes',
  'Machine Learning > Classical Machine Learning',
  'Machine Learning > Deep Learning',
  'Machine Learning > Probabilistic Graphical Models',
  'Mathematics > Mathematical Analysis',
  'Mathematics > Algebra & Matrix Theory',
  'Mathematics > Geometry & Topology',
  'Mathematics > Optimization',
  'Programming > CS Foundations',
  'Programming > Programming Languages',
  'Programming > Full Stack Development',
  'Work & Society > AI Engineering Workflows',
  'Work & Society > Builder & Product Thinking',
  'Work & Society > Career & Learning',
  'Work & Society > AI & Society',
  'Work & Society > Research Practice',
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
  ['AI Products', 'AI Product'],
  ['Coding Agent', 'AI Coding'],
  ['Coding Agents', 'AI Coding'],
  ['Agent Evaluation', 'Evaluation'],
  ['Evals', 'Evaluation']
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
  return m ? yaml.load(m[1]) || {} : {};
}

function fmValue(fm, key) {
  return fm[key] == null ? '' : String(fm[key]).trim();
}

function fmArray(fm, key) {
  return Array.isArray(fm[key]) ? fm[key] : [];
}

function metadataErrors(source, english) {
  const result = [];
  for (const key of ['categories', 'tags']) {
    if (!Array.isArray(source[key]) || !Array.isArray(english[key]) ||
        JSON.stringify(source[key]) !== JSON.stringify(english[key])) {
      result.push(`English ${key} must match Chinese source in order`);
    }
  }
  for (const [language, fm] of [['Chinese', source], ['English', english]]) {
    if (Object.prototype.hasOwnProperty.call(fm, 'hidden') && typeof fm.hidden !== 'boolean') {
      result.push(`${language} hidden must be a boolean`);
    }
  }
  if ((source.hidden ?? false) !== (english.hidden ?? false)) result.push('English hidden must match Chinese source');
  return result;
}

function validateRetiredFrontMatter() {
  for (const folder of ['source/_posts', 'source/_drafts', 'source_en/_posts', 'source_en/_drafts']) {
    const dir = path.join(root, folder);
    if (!fs.existsSync(dir)) continue;
    walk(dir).filter(file => file.endsWith('.md')).forEach(file => {
      const fm = frontMatter(fs.readFileSync(file, 'utf8'));
      if (JSON.stringify(fm).includes('Backend Engineering')) {
        errors.push(`${rel(file)} uses retired Backend Engineering in front matter`);
      }
    });
  }
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
    if (!fmValue(fm, 'permalink')) errors.push(`${name} is missing permalink`);
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

function validateLanguageSetup() {
  const config = read('_config.yml');
  const englishConfig = read('_config.en.yml');
  const toggle = read('source/assets/js/lang-toggle.js');
  if (!/^language:\s*zh-CN\s*$/m.test(config)) errors.push('_config.yml must make Chinese the primary language');
  if (!/^root:\s*\/en\/\s*$/m.test(englishConfig)) errors.push('_config.en.yml must publish the English edition under /en/');
  if (!toggle.includes('var DEFAULT_LANG = "zh"')) errors.push('language toggle must default to Chinese');
  if (!toggle.includes('isEnglishRoute()')) errors.push('language toggle must recognize static /en/ routes');
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
}

function sourceDirectoryErrors(zhPosts, enPosts, zhDrafts, enDrafts) {
  const result = [];
  for (const name of zhPosts) {
    if (!enPosts.includes(name)) result.push(`English translation is missing from _posts: ${name}`);
  }
  for (const name of enPosts) {
    if (!zhPosts.includes(name)) result.push(`Orphan English translation in _posts: ${name}`);
  }
  for (const name of enDrafts) {
    if (!zhDrafts.includes(name)) result.push(`English draft has no Chinese draft: ${name}`);
  }
  for (const [language, posts, drafts] of [['Chinese', zhPosts, zhDrafts], ['English', enPosts, enDrafts]]) {
    for (const name of posts) {
      if (drafts.includes(name)) result.push(`${language} source exists in both _posts and _drafts: ${name}`);
    }
  }
  return result;
}

function validateEnglishSources() {
  const files = (language, folder) => {
    const dir = path.join(root, language, folder);
    return fs.existsSync(dir) ? fs.readdirSync(dir).filter(name => name.endsWith('.md')).sort() : [];
  };
  const zhPosts = files('source', '_posts');
  const enPosts = files('source_en', '_posts');
  const zhDrafts = files('source', '_drafts');
  const enDrafts = files('source_en', '_drafts');
  errors.push(...sourceDirectoryErrors(zhPosts, enPosts, zhDrafts, enDrafts));

  for (const [folder, chinese, english] of [['_posts', zhPosts, enPosts], ['_drafts', zhDrafts, enDrafts]]) {
    for (const name of english.filter(name => chinese.includes(name))) {
      const source = read(`source_en/${folder}/${name}`);
      const fm = frontMatter(source);
      const chineseFm = frontMatter(read(`source/${folder}/${name}`));
      metadataErrors(chineseFm, fm).forEach(message => errors.push(`${folder}/${name}: ${message}`));
      if (fmValue(fm, 'lang') !== 'en') errors.push(`${name} English source must set lang: en`);
      if (fmValue(fm, 'translation_status') !== 'machine') errors.push(`${name} must disclose translation_status: machine`);
      if (!source.includes('class="translation-notice"')) errors.push(`${name} is missing the visible machine-translation notice`);
      if (hasCjk(fmValue(fm, 'title'))) errors.push(`${name} has a CJK English title`);
    }
  }

  const forbiddenFixed = ['murmur'];
  forbiddenFixed.forEach(folder => {
    const candidate = path.join(root, 'source_en', folder);
    if (fs.existsSync(candidate) && walk(candidate).some(file => file.endsWith('.md'))) {
      errors.push(`English fixed-page source must not exist: source_en/${folder}/`);
    }
  });
}

function run() {
  validateDictionary();
  validatePosts();
  validateTaxonomyConfig();
  validateLanguageSetup();
  validatePostI18nGenerator();
  validateEnglishSources();
  validateRetiredFrontMatter();

  if (errors.length) {
    console.error('i18n validation failed:');
    errors.forEach(err => console.error(' - ' + err));
    process.exitCode = 1;
    return;
  }
  console.log('i18n validation passed');
}

if (require.main === module) run();
module.exports = { run, frontMatter, metadataErrors, sourceDirectoryErrors };
