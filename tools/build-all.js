/* eslint-disable no-console */
'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const { writeTaxonomyRedirects } = require('./taxonomy-redirects');

const root = path.resolve(__dirname, '..');
const npx = process.platform === 'win32' ? 'npx.cmd' : 'npx';

function run(command, args) {
  console.log(`[build] ${command} ${args.join(' ')}`);
  const needsShell = process.platform === 'win32' && /\.(?:cmd|bat)$/i.test(command);
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: 'inherit',
    // Only Windows command shims need shell resolution. Executables such as
    // process.execPath may contain spaces and must be spawned directly.
    shell: needsShell
  });
  if (result.status !== 0) process.exit(result.status || 1);
}

function walkFiles(dir, callback) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) walkFiles(file, callback);
    else callback(file);
  }
}

function normalizeEnglishImagePaths(dir) {
  walkFiles(dir, file => {
    if (!file.endsWith('.html')) return;
    const source = fs.readFileSync(file, 'utf8');
    const normalized = source.replace(/(["'(])\/en\/assets\/images\//g, '$1/assets/images/');
    if (normalized !== source) fs.writeFileSync(file, normalized, 'utf8');
  });
}

const FIXED_PAGE_DIRS = [
  'categories', 'tags', 'me', 'cv', 'projects',
  'footprints', 'friends', 'comments', 'photos'
];

function rewriteFixedRoutes(dir) {
  if (!fs.existsSync(dir)) return;
  walkFiles(dir, file => {
    if (!file.endsWith('.html')) return;
    const source = fs.readFileSync(file, 'utf8');
    const normalized = source
      .replace(/(["'(])\/en\/(archives|categories|tags|me|cv|projects|footprints|friends|comments|photos|murmur|masonry)(?=[\/#?"')])/g, '$1/$2')
      .replace(/(["'(])\/en\/(assets|css|js|images|fonts|webfonts)\//g, '$1/$2/')
      .replace(/<a(?![^>]*\bdata-no-swup)[^>]*\bhref="\/(archives|categories|tags|me|cv|projects|footprints|friends|comments|photos|murmur)(?:\/|["#?])/gi, match => match.replace('<a', '<a data-no-swup="true"'));
    const withoutSwup = normalized.replace(/<script>\s*window\.swup\s*=\s*new Swup\([\s\S]*?<\/script>/i, '');
    if (withoutSwup !== source) fs.writeFileSync(file, withoutSwup, 'utf8');
  });
}

function normalizeMasonryScriptOrder(dir) {
  if (!fs.existsSync(dir)) return;
  walkFiles(dir, file => {
    if (!file.endsWith('.html')) return;
    const source = fs.readFileSync(file, 'utf8');
    const mainRe = /(<script[^>]+src="[^"]*\/js\/build\/main\.js"[^>]*><\/script>)/i;
    const masonryRe = /(<script[^>]+src="[^"]*\/js\/build\/libs\/minimasonry\.min\.js"[^>]*><\/script>)/i;
    const main = source.match(mainRe);
    const masonry = source.match(masonryRe);
    if (!main || !masonry || main.index < masonry.index) return;
    const without = source.slice(0, masonry.index) + source.slice(masonry.index + masonry[0].length);
    const adjustedMainIndex = without.indexOf(main[0]);
    if (adjustedMainIndex < 0) return;
    const removedMain = without.slice(0, adjustedMainIndex) + without.slice(adjustedMainIndex + main[0].length);
    const insertAt = Math.max(0, adjustedMainIndex);
    const next = removedMain.slice(0, insertAt) + masonry[0] + '\n' + main[0] + removedMain.slice(insertAt);
    if (next !== source) fs.writeFileSync(file, next, 'utf8');
  });
}

function relocateFixedEnglishPages(englishOutput) {
  FIXED_PAGE_DIRS.forEach(name => {
    const source = path.join(englishOutput, name);
    if (!fs.existsSync(source)) return;
    const target = path.join(root, 'public', name);
    if (fs.existsSync(target)) fs.rmSync(target, { recursive: true, force: true });
    fs.cpSync(source, target, { recursive: true });
    fs.rmSync(source, { recursive: true, force: true });
    walkFiles(target, file => {
      if (!file.endsWith('.html')) return;
      const sourceHtml = fs.readFileSync(file, 'utf8');
      const rootConfig = sourceHtml.replace(/("root"\s*:\s*)"\/en\/"/g, '$1"/"');
      if (rootConfig !== sourceHtml) fs.writeFileSync(file, rootConfig, 'utf8');
    });
  });
  const murmur = path.join(englishOutput, 'murmur');
  if (fs.existsSync(murmur)) fs.rmSync(murmur, { recursive: true, force: true });
  // Archives are generated from the Chinese build and remain a fixed Chinese
  // page at /archives/. Never let the English generator replace it.
  const englishArchives = path.join(englishOutput, 'archives');
  if (fs.existsSync(englishArchives)) fs.rmSync(englishArchives, { recursive: true, force: true });
  const englishMasonry = path.join(englishOutput, 'masonry');
  if (fs.existsSync(englishMasonry)) fs.rmSync(englishMasonry, { recursive: true, force: true });
}

run(npx, ['hexo', 'clean']);
run(npx, ['hexo', 'generate', '--config', '_config.yml,_config.en.yml']);
const englishOutput = path.join(root, 'public', 'en');
const stagedEnglish = path.join(root, '.build-en-staging');
if (fs.existsSync(stagedEnglish)) fs.rmSync(stagedEnglish, { recursive: true, force: true });
if (fs.existsSync(englishOutput)) {
  // Copy instead of rename: Windows can keep generated files open briefly.
  fs.cpSync(englishOutput, stagedEnglish, { recursive: true });
  fs.rmSync(englishOutput, { recursive: true, force: true });
}
run(npx, ['hexo', 'clean']);
run(npx, ['hexo', 'generate']);
if (fs.existsSync(stagedEnglish)) {
  fs.mkdirSync(path.join(root, 'public'), { recursive: true });
  fs.cpSync(stagedEnglish, englishOutput, { recursive: true });
  fs.rmSync(stagedEnglish, { recursive: true, force: true });
  normalizeEnglishImagePaths(englishOutput);
  relocateFixedEnglishPages(englishOutput);
  rewriteFixedRoutes(englishOutput);
}
rewriteFixedRoutes(path.join(root, 'public'));
normalizeMasonryScriptOrder(path.join(root, 'public'));
writeTaxonomyRedirects(yaml.load(fs.readFileSync(path.join(root, '_config.yml'), 'utf8')), path.join(root, 'public'));
run(process.execPath, ['scripts/validate-i18n.js']);
run(process.execPath, ['scripts/validate-search.js']);
run(process.execPath, ['tools/validate-seo.js']);
run(process.execPath, ['tools/validate-publication.js']);
