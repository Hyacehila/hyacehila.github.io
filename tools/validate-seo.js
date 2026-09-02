/* eslint-disable no-console */
'use strict';

const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');

const root = path.resolve(__dirname, '..');
const publicDir = path.join(root, 'public');
const failures = [];
const warnings = [];
const pages = [];
const FIXED_ENGLISH_ROOTS = ['/me/', '/cv/', '/projects/', '/footprints/', '/friends/', '/comments/', '/photos/', '/categories/', '/tags/'];

function walk(dir, out = []) {
  if (!fs.existsSync(dir)) return out;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out); else out.push(full);
  }
  return out;
}

function rel(file) {
  return path.relative(publicDir, file).replace(/\\/g, '/');
}

function fail(file, message) {
  failures.push(`${rel(file)}: ${message}`);
}

function localRoute(file) {
  const relative = rel(file);
  if (relative === 'index.html') return '/';
  if (relative.endsWith('/index.html')) return `/${relative.slice(0, -10)}`;
  return `/${relative}`;
}

function isFixedEnglishRoute(route) {
  return FIXED_ENGLISH_ROOTS.some(root => route === root || route.startsWith(root));
}

function expectedFileForHref(href) {
  let pathname;
  try { pathname = new URL(href, 'https://hyacehila.github.io/').pathname; } catch (_) { return null; }
  try { pathname = decodeURIComponent(pathname); } catch (_) {}
  const trimmed = pathname.replace(/^\/+/, '');
  if (!trimmed) return path.join(publicDir, 'index.html');
  if (/\.[a-z0-9]{1,8}$/i.test(trimmed)) return path.join(publicDir, trimmed);
  return path.join(publicDir, trimmed, 'index.html');
}

function xmlLocs(file) {
  if (!fs.existsSync(file)) return [];
  return Array.from(fs.readFileSync(file, 'utf8').matchAll(/<loc>([^<]+)<\/loc>/g), match => match[1]);
}

for (const file of walk(publicDir).filter(item => item.endsWith('.html'))) {
  const html = fs.readFileSync(file, 'utf8');
  const $ = cheerio.load(html);
  const redirect = $('meta[http-equiv="refresh"]').length > 0;
  const canonicalNodes = $('link[rel="canonical"]');
  const canonical = canonicalNodes.first().attr('href') || '';
  const robotsNodes = $('meta[name="robots"]');
  const robots = robotsNodes.first().attr('content') || '';
  const noindex = /\bnoindex\b/i.test(robots);
  const route = localRoute(file);
  const isFixedEnglish = isFixedEnglishRoute(route);
  const isMurmur = route === '/murmur/' || route.startsWith('/murmur/');
  const isEnglish = rel(file).startsWith('en/') || isFixedEnglish;
  const isPost = $('.post-page-container .article-content').length > 0;
  const page = { file, canonical, noindex, isPost, isEnglish, isFixedEnglish, redirect, route };
  pages.push(page);

  if (canonicalNodes.length !== 1) fail(file, `expected one canonical, found ${canonicalNodes.length}`);
  if (!/^https:\/\/hyacehila\.github\.io\//.test(canonical)) fail(file, `invalid canonical "${canonical}"`);
  if (isEnglish && !isFixedEnglish && !/^https:\/\/hyacehila\.github\.io\/en\//.test(canonical)) fail(file, 'English canonical is missing /en/');
  if (isFixedEnglish && !new URL(canonical || 'https://hyacehila.github.io/').pathname.startsWith('/' + route.replace(/^\//, '').split('/')[0])) fail(file, 'fixed English page canonical must use the root route');
  if (!isEnglish && /^https:\/\/hyacehila\.github\.io\/en\//.test(canonical)) fail(file, 'Chinese page points canonical to /en/');
  const expectedLang = isMurmur ? 'zh-CN' : (isEnglish ? 'en' : 'zh-CN');
  if ($('html').attr('lang') !== expectedLang) fail(file, `incorrect html lang "${$('html').attr('lang')}"`);

  if (!redirect) {
    if (robotsNodes.length !== 1) fail(file, `expected one robots meta, found ${robotsNodes.length}`);
    if ($('meta[name="googlebot"]').length !== 1) fail(file, 'missing or duplicate googlebot meta');
    if (!$('title').text().trim()) fail(file, 'missing title');
    if (!noindex && !$('meta[name="description"]').attr('content')) fail(file, 'indexable page has no description');
    if (!noindex && $('h1').length !== 1) fail(file, `indexable page must have one H1, found ${$('h1').length}`);
    $('script[type="application/ld+json"]').each((_, element) => {
      try { JSON.parse($(element).text()); } catch (error) { fail(file, `invalid JSON-LD: ${error.message}`); }
    });
  }

  if (isPost) {
    if ($('h1').length !== 1) fail(file, `post must have one H1, found ${$('h1').length}`);
    if ($('.seo-breadcrumb').length !== 1) fail(file, 'missing visible breadcrumb');
    if ($('script[data-seo="breadcrumb"]').length !== 1) fail(file, 'missing BreadcrumbList JSON-LD');
    if ($('.seo-context-links').length) fail(file, 'forbidden automatic related-reading module is present');
    const image = $('meta[property="og:image"]').attr('content') || '';
    if (!/\/assets\/images\/og\/[^/]+\.png$/.test(image)) fail(file, `post has non-unique OG image "${image}"`);
    const scripts = $('script[type="application/ld+json"]:not([data-seo])').map((_, el) => $(el).text()).get();
    if (!scripts.some(source => source.includes('"@type":"BlogPosting"') && source.includes(canonical))) fail(file, 'BlogPosting schema does not match canonical');
    ['zh-CN', 'en', 'x-default'].forEach(lang => {
      if ($(`link[rel="alternate"][hreflang="${lang}"]`).length !== 1) fail(file, `missing ${lang} hreflang`);
    });
    if (isEnglish && $('.translation-notice').length !== 1) fail(file, 'English post is missing machine-translation disclosure');
    $('.article-content img').each((_, element) => {
      const imageNode = $(element);
      const actual = imageNode.attr('data-src') || imageNode.attr('src') || '';
      if (!imageNode.attr('alt')) fail(file, `content image has no alt: ${actual}`);
      if (actual.startsWith('/') && !actual.startsWith('//') && (!imageNode.attr('width') || !imageNode.attr('height'))) {
        fail(file, `local content image has no intrinsic dimensions: ${actual}`);
      }
    });
  }

  if (!isPost && !isMurmur && isFixedEnglish && $('.translation-notice').length) fail(file, 'fixed English page must not contain a translation notice');
  if (isFixedEnglish && $('link[rel="alternate"][hreflang]').length) fail(file, 'fixed English page must not emit hreflang alternates');

  const pathname = (() => { try { return new URL(canonical).pathname.replace(/^\/en(?=\/)/, ''); } catch (_) { return ''; } })();
  const hasMermaid = $('.mermaid').length > 0;
  $('script[src]').each((_, element) => {
    const src = $(element).attr('src') || '';
    if (/Typed\.min\.js/i.test(src) && pathname !== '/') fail(file, 'Typed.js loaded outside the homepage');
    if (/mermaid(?:\.min)?\.js/i.test(src) && !hasMermaid) fail(file, 'Mermaid loaded on a page without a diagram');
    if (/moment-with-locales\.min\.js/i.test(src) && !['/murmur/', '/essays/'].includes(pathname)) fail(file, 'Moment locales loaded outside date-heavy pages');
    if (/vercount\.one\/js/i.test(src)) fail(file, 'counter script still blocks page parsing');
  });
}

const indexable = pages.filter(page => !page.noindex && !page.canonical.endsWith('/404.html'));
for (const field of ['title', 'description']) {
  for (const language of [false, true]) {
    const groups = new Map();
    indexable.filter(page => page.isEnglish === language).forEach(page => {
      const $ = cheerio.load(fs.readFileSync(page.file, 'utf8'));
      const value = field === 'title' ? $('title').text().trim() : ($('meta[name="description"]').attr('content') || '').trim();
      if (!value) return;
      if (!groups.has(value)) groups.set(value, []);
      groups.get(value).push(page);
    });
    groups.forEach(group => {
      if (group.length > 1) failures.push(`duplicate ${field}: ${group.map(page => rel(page.file)).join(', ')}`);
    });
  }
}

const zhSitemap = xmlLocs(path.join(publicDir, 'sitemap-zh.xml'));
const enSitemap = xmlLocs(path.join(publicDir, 'sitemap-en.xml'));
if (!zhSitemap.length) failures.push('sitemap-zh.xml has no URLs');
if (!enSitemap.length) failures.push('sitemap-en.xml has no URLs');

['photos', 'archives', 'murmur', 'me', 'cv', 'projects', 'footprints', 'friends', 'comments', 'categories', 'tags', 'masonry']
  .forEach(name => {
    if (fs.existsSync(path.join(publicDir, 'en', name))) failures.push(`forbidden fixed-page output exists: public/en/${name}/`);
  });
try {
  const masonryData = JSON.parse(fs.readFileSync(path.join(publicDir, 'masonry', 'data.json'), 'utf8'));
  if (!Array.isArray(masonryData) || masonryData.length === 0) failures.push('masonry/data.json is empty');
  if (Array.isArray(masonryData) && masonryData.some(item => !item || !item.image || !item.width || !item.height)) {
    failures.push('masonry/data.json contains an item without image dimensions');
  }
} catch (error) {
  failures.push(`masonry/data.json is invalid: ${error.message}`);
}
const sitemapUrls = new Set([...zhSitemap, ...enSitemap]);
pages.filter(page => page.noindex && !page.redirect).forEach(page => {
  if (sitemapUrls.has(page.canonical)) fail(page.file, 'noindex URL is present in sitemap');
});
pages.filter(page => page.isPost && !page.noindex).forEach(page => {
  if (!sitemapUrls.has(page.canonical)) fail(page.file, 'indexable post is missing from sitemap');
});

const broken = new Map();
pages.filter(page => !page.noindex && !page.file.includes(`${path.sep}archives${path.sep}`)).forEach(page => {
  const $ = cheerio.load(fs.readFileSync(page.file, 'utf8'));
  $('a[href]').each((_, element) => {
    const href = $(element).attr('href') || '';
    if (!href.startsWith('/') || href.startsWith('//') || href.startsWith('/#')) return;
    const target = expectedFileForHref(href.split('#')[0].split('?')[0]);
    if (target && !fs.existsSync(target)) {
      const key = `${page.route} -> ${href}`;
      broken.set(key, true);
    }
  });
});
if (broken.size) {
  Array.from(broken.keys()).slice(0, 30).forEach(item => warnings.push(`unresolved internal link: ${item}`));
  if (broken.size > 30) warnings.push(`...and ${broken.size - 30} more unresolved internal links`);
}

const summary = {
  htmlPages: pages.length,
  posts: pages.filter(page => page.isPost).length,
  indexable: indexable.length,
  noindex: pages.filter(page => page.noindex).length,
  sitemap: { zh: zhSitemap.length, en: enSitemap.length },
  warnings: warnings.length,
  failures: failures.length
};
console.log(JSON.stringify(summary, null, 2));
if (warnings.length) {
  console.warn('SEO validation warnings:');
  warnings.forEach(item => console.warn(` - ${item}`));
}
if (failures.length) {
  console.error('SEO validation failed:');
  failures.slice(0, 100).forEach(item => console.error(` - ${item}`));
  if (failures.length > 100) console.error(` - ...and ${failures.length - 100} more`);
  process.exit(1);
}
console.log('SEO validation passed.');
