/* global hexo */
'use strict';

const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');

const postsByUrl = new Map();
let allPosts = [];
let categoryCounts = new Map();
let tagCounts = new Map();

// The English edition of these public pages is served at the original root
// route. Blog posts remain the only content with a /en/ counterpart.
const FIXED_ENGLISH_ROOTS = new Set([
  '/me/', '/cv/', '/projects/', '/footprints/', '/friends/', '/comments/',
  '/photos/', '/categories/', '/tags/'
]);

function isFixedEnglishPath(pathname) {
  return Array.from(FIXED_ENGLISH_ROOTS).some(root => pathname === root || pathname.startsWith(root));
}

function isMurmurPath(pathname) {
  return pathname === '/murmur/' || pathname.startsWith('/murmur/');
}

// English machine-translated sources may contain unmatched dollar signs from
// prose/code. The KaTeX filter assumes every `$` has a closing delimiter and
// can otherwise loop indefinitely; keep those pages renderable as plain text.
hexo.extend.filter.register('before_post_render', function (data) {
  if (String(this.config.language || '') === 'en' && typeof data.content === 'string') {
    data.content = data.content.replace(/\$/g, '&#36;');
  }
  return data;
}, 8);

function list(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  if (Array.isArray(value.data)) return value.data;
  if (typeof value.toArray === 'function') return value.toArray();
  return [value];
}

function named(value) {
  return list(value).map(item => ({
    name: String(item && (item.name || (item.data && item.data.name)) || item || ''),
    path: String(item && (item.path || (item.data && item.data.path)) || '')
  })).filter(item => item.name);
}

function siteBase(config) {
  return String(config.url || '').replace(/\/$/, '');
}

function canonicalFor(config, post) {
  let value = post && post.permalink
    ? String(post.permalink).replace(/\/index\.html$/, '/')
    : `${siteBase(config)}${config.root || '/'}${String(post.path || '').replace(/^\/+/, '')}`
    .replace(/([^:]\/)\/+/g, '$1')
    .replace(/\/index\.html$/, '/');
  const pathname = routePath(value);
  if (String(config.root || '/') === '/en/' && !pathname.startsWith('/en/') && !isFixedEnglishPath(pathname)) {
    value = `${siteBase(config)}/en${routePath(value)}`;
  }
  return value;
}

function routePath(url) {
  try { return new URL(url).pathname.replace(/\/index\.html$/, '/'); } catch (_) { return '/'; }
}

function languagePath(url) {
  const pathname = routePath(url);
  return pathname === '/en/' ? '/' : pathname.replace(/^\/en(?=\/)/, '') || '/';
}

function counterpart(url, language) {
  const base = siteBase(hexo.config);
  const raw = languagePath(url);
  return language === 'en' ? `${base}/en${raw === '/' ? '/' : raw}` : `${base}${raw}`;
}

function escapeXml(value) {
  return String(value || '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;');
}

function slugify(value) {
  return encodeURI(String(value).trim().replace(/[\s/]+/g, '-'));
}

function humanSlug(value) {
  try { value = decodeURIComponent(value); } catch (_) {}
  return String(value || '').replace(/[-_]+/g, ' ').trim();
}

function postImage(canonical) {
  const parts = languagePath(canonical).split('/').filter(Boolean);
  return `${siteBase(hexo.config)}/assets/images/og/${encodeURIComponent(parts[parts.length - 1] || 'home')}.png`;
}

function text(value) {
  return String(value || '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
}

function setMeta($, selector, attrs, content) {
  let node = $(selector).first();
  if (!node.length) {
    node = $('<meta>');
    Object.entries(attrs).forEach(([key, value]) => node.attr(key, value));
    $('head').append(node);
  }
  node.attr('content', content);
  $(selector).slice(1).remove();
}

function setLink($, selector, attrs) {
  $(selector).remove();
  const node = $('<link>');
  Object.entries(attrs).forEach(([key, value]) => node.attr(key, value));
  $('head').append(node);
}

function pageDescription(pathname, $, lang, noindex) {
  const title = text($('h1, .category-name, .tag-name, .page-title-header').first().text()) || text($('title').text().split('|')[0]);
  const pageMatch = pathname.match(/\/page\/(\d+)\/?$/);
  const pageSuffix = pageMatch ? (lang === 'en' ? `, page ${pageMatch[1]}` : `，第 ${pageMatch[1]} 页`) : '';
  if (/^\/categories\/.+/.test(pathname)) {
    const name = title || humanSlug(pathname.split('/').filter(Boolean).pop());
    return lang === 'en'
      ? `Articles in the ${name} category${pageSuffix}, covering AI, data science, statistics, and software engineering.`
      : `「${name}」分类下的技术文章${pageSuffix}，涵盖 AI、数据科学、统计学与软件工程。`;
  }
  if (/^\/tags\/.+/.test(pathname)) {
    const name = title || humanSlug(pathname.split('/').filter(Boolean).pop());
    return lang === 'en'
      ? `Articles tagged ${name}${pageSuffix} on Hyacehila's Blog.`
      : `Hyacehila 的「${name}」标签文章${pageSuffix}。`;
  }
  if (/^\/archives\//.test(pathname)) {
    return lang === 'en' ? `Archive of published articles${pageSuffix}.` : `博客文章归档${pageSuffix}。`;
  }
  const known = {
    '/': ['关于 AI 智能体、大语言模型、数据科学、统计学与软件工程的个人技术博客。', 'A personal technical blog about AI agents, large language models, data science, statistics, and software engineering.'],
    '/categories/': ['按主题浏览博客文章分类。', 'Browse blog articles by topic and category.'],
    '/tags/': ['浏览博客文章标签索引。', 'Browse the blog tag index.'],
    '/me/': ['了解 Hyacehila 的研究兴趣、技术背景与联系方式。', 'About Hyacehila, research interests, technical background, and contact details.'],
    '/cv/': ['Hyacehila 的教育、研究与工作经历。', "Hyacehila's education, research, and professional experience."],
    '/projects/': ['Hyacehila 的软件、数据科学与 AI 项目。', "Hyacehila's software, data science, and AI projects."],
    '/photos/': ['Hyacehila 的摄影作品与视觉记录。', "Hyacehila's photography and visual journal."],
    '/footprints/': ['Hyacehila 的旅行足迹与地点记录。', "Hyacehila's travel footprints and place journal."],
    '/murmur/': ['简短想法、近况与随笔记录。', 'Short notes, updates, and passing thoughts.'],
    '/comments/': ['访客留言页面。', 'Visitor comments page.'],
    '/friends/': ['友情链接与朋友站点。', 'Friends and recommended websites.'],
    '/404.html': ['页面未找到。', 'Page not found.']
  };
  const pair = known[pathname];
  if (pair) return pair[lang === 'en' ? 1 : 0];
  if (noindex) return lang === 'en' ? `${title || 'Page'}${pageSuffix} on Hyacehila's Blog.` : `${title || '页面'}${pageSuffix}｜Hyacehila 的博客。`;
  return '';
}

function shouldNoindex(pathname, $) {
  const seo = hexo.config.seo || {};
  if (list(seo.noindex_pages).includes(pathname)) return true;
  if (/\/page\/\d+\/?$/.test(pathname)) return true;
  if (/^\/archives(?:\/|$)/.test(pathname)) return true;
  if (/^\/tags\/[^/]+\/?$/.test(pathname)) {
    const name = text($('.tag-name').first().text()) || humanSlug(pathname.split('/').filter(Boolean).pop());
    return !new Set(list(seo.index_tags).map(String)).has(name);
  }
  if (/^\/categories\/.+\/?$/.test(pathname)) {
    const name = text($('.category-name').first().text());
    const count = categoryCounts.get(name) || $('.category-post-list li, .post-list-item').length;
    return count < Number(seo.min_category_posts || 3);
  }
  return false;
}

function breadcrumbItems(record, canonical, lang, $, pathname) {
  const home = { name: lang === 'en' ? 'Home' : '首页', url: lang === 'en' && !isFixedEnglishPath(pathname) && !isMurmurPath(pathname) ? `${siteBase(hexo.config)}/en/` : `${siteBase(hexo.config)}/` };
  if (record) {
    const items = [home];
    record.categories.forEach(category => items.push({
      name: category.name,
      url: `${siteBase(hexo.config)}/${String(category.path || `categories/${slugify(category.name)}/`).replace(/^\/+/, '')}`
    }));
    items.push({ name: record.title, url: canonical });
    return items;
  }
  if (/^\/(?:categories|tags)\/.+/.test(pathname)) {
    const isTag = pathname.startsWith('/tags/');
    return [home, {
      name: lang === 'en' ? (isTag ? 'Tags' : 'Categories') : (isTag ? '标签' : '分类'),
      url: `${siteBase(hexo.config)}/${isTag ? 'tags' : 'categories'}/`
    }, { name: text($(isTag ? '.tag-name' : '.category-name').first().text()), url: canonical }];
  }
  return [];
}

function addBreadcrumb($, record, canonical, lang, pathname) {
  const items = breadcrumbItems(record, canonical, lang, $, pathname);
  if (!items.length) return;
  const nav = $('<nav class="seo-breadcrumb px-2 sm:px-6 md:px-8" aria-label="Breadcrumb"><ol></ol></nav>');
  items.forEach((item, index) => {
    const li = $('<li></li>');
    if (index === items.length - 1) li.attr('aria-current', 'page').text(item.name);
    else li.append($('<a></a>').attr('href', item.url).text(item.name));
    nav.find('ol').append(li);
  });
  if (record) $('.article-title').first().after(nav);
  else $('.category-name, .tag-name').first().before(nav);
  $('head').append($('<script type="application/ld+json" data-seo="breadcrumb"></script>').text(JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: items.map((item, index) => ({ '@type': 'ListItem', position: index + 1, name: item.name, item: item.url }))
  })));
}

function imageSize(file) {
  try {
    const buffer = fs.readFileSync(file);
    if (buffer.length > 24 && buffer.toString('ascii', 1, 4) === 'PNG') return [buffer.readUInt32BE(16), buffer.readUInt32BE(20)];
    if (buffer.length > 10 && buffer.toString('ascii', 0, 3) === 'GIF') return [buffer.readUInt16LE(6), buffer.readUInt16LE(8)];
    if (buffer.length > 12 && buffer.toString('ascii', 0, 4) === 'RIFF' && buffer.toString('ascii', 8, 12) === 'WEBP') {
      const kind = buffer.toString('ascii', 12, 16);
      if (kind === 'VP8X') return [1 + buffer.readUIntLE(24, 3), 1 + buffer.readUIntLE(27, 3)];
    }
    if (buffer.length > 4 && buffer[0] === 0xff && buffer[1] === 0xd8) {
      let offset = 2;
      while (offset + 9 < buffer.length) {
        if (buffer[offset] !== 0xff) { offset += 1; continue; }
        const marker = buffer[offset + 1];
        const length = buffer.readUInt16BE(offset + 2);
        if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(marker)) {
          return [buffer.readUInt16BE(offset + 7), buffer.readUInt16BE(offset + 5)];
        }
        offset += Math.max(2, length + 2);
      }
    }
    if (/\.svg$/i.test(file)) {
      const source = buffer.toString('utf8', 0, Math.min(buffer.length, 4096));
      const width = source.match(/\bwidth=["']([\d.]+)/i);
      const height = source.match(/\bheight=["']([\d.]+)/i);
      const viewBox = source.match(/\bviewBox=["'][^"']*?([\d.]+)\s+([\d.]+)["']/i);
      if (width && height) return [Math.round(Number(width[1])), Math.round(Number(height[1]))];
      if (viewBox) return [Math.round(Number(viewBox[1])), Math.round(Number(viewBox[2]))];
    }
  } catch (_) {}
  return null;
}

function improveImages($, lang) {
  $('.article-content img').each((_, element) => {
    const image = $(element);
    const actual = image.attr('data-src') || image.attr('src') || '';
    if (!image.attr('alt')) {
      let label = humanSlug(path.basename(actual.split(/[?#]/)[0], path.extname(actual)));
      if (!label || label === 'loading') label = lang === 'en' ? 'Article illustration' : '文章插图';
      image.attr('alt', lang === 'en' ? `Illustration: ${label}` : `插图：${label}`);
    }
    if (image.attr('width') && image.attr('height')) return;
    if (!actual.startsWith('/') || actual.startsWith('//')) return;
    const decoded = (() => { try { return decodeURIComponent(actual.split(/[?#]/)[0]); } catch (_) { return actual.split(/[?#]/)[0]; } })();
    const sourceFile = path.join(hexo.base_dir, hexo.config.source_dir || 'source', decoded.replace(/^\/+/, ''));
    const size = imageSize(sourceFile) || imageSize(path.join(hexo.base_dir, 'source', decoded.replace(/^\/+/, '')));
    if (size && size[0] && size[1]) image.attr({ width: size[0], height: size[1] });
  });
}

function conditionAssets($, pathname) {
  const hasMermaid = $('.mermaid').length > 0;
  const isHome = pathname === '/';
  const isMomentPage = pathname === '/murmur/' || pathname === '/essays/';
  const isMasonry = $('.masonry-container, .photo-wall, [data-masonry]').length > 0 || pathname === '/photos/';
  $('script[src]').each((_, element) => {
    const node = $(element);
    const src = node.attr('src') || '';
    if (/Typed\.min\.js/i.test(src) && !isHome) node.remove();
    if (/mermaid(?:\.min)?\.js/i.test(src) && !hasMermaid) node.remove();
    if (/moment-with-locales\.min\.js/i.test(src) && !isMomentPage) node.remove();
    if (/minimasonry\.min\.js/i.test(src) && !isMasonry) node.remove();
    if (/vercount\.one\/js/i.test(src)) {
      node.replaceWith('<script data-seo="deferred-counter">window.addEventListener("load",function(){var f=function(){var s=document.createElement("script");s.src="https://vercount.one/js";s.async=true;document.body.appendChild(s)};if("requestIdleCallback" in window)requestIdleCallback(f,{timeout:2500});else setTimeout(f,1200)},{once:true});</script>');
    }
  });
}

function updateStructuredData($, canonical, lang, record) {
  $('script[type="application/ld+json"]:not([data-seo])').each((_, element) => {
    let value;
    try { value = JSON.parse($(element).text()); } catch (_) { return; }
    const nodes = value && Array.isArray(value['@graph']) ? value['@graph'] : [value];
    nodes.forEach(node => {
      if (!node || typeof node !== 'object') return;
      node.inLanguage = lang === 'en' ? 'en' : 'zh-CN';
      if (node.url) node.url = canonical;
      const description = $('meta[name="description"]').attr('content');
      if (description && node.description) node.description = description;
      if (node['@type'] === 'BlogPosting' && record) {
        node.url = canonical;
        node.mainEntityOfPage = { '@type': 'WebPage', '@id': canonical };
        node.image = [postImage(canonical)];
      }
    });
    $(element).text(JSON.stringify(value));
  });
}

hexo.extend.filter.register('before_generate', function () {
  postsByUrl.clear();
  allPosts = [];
  categoryCounts = new Map();
  tagCounts = new Map();
  const posts = this.locals && this.locals.get('posts');
  if (!posts) return;
  posts.forEach(post => {
    const categories = named(post.categories);
    const tags = named(post.tags);
    const record = {
      url: canonicalFor(this.config, post),
      title: text(post.title),
      date: new Date(post.date || 0).getTime(),
      categories,
      tags
    };
    postsByUrl.set(record.url, record);
    allPosts.push(record);
    categories.forEach(item => categoryCounts.set(item.name, (categoryCounts.get(item.name) || 0) + 1));
    tags.forEach(item => tagCounts.set(item.name, (tagCounts.get(item.name) || 0) + 1));
  });
});

hexo.extend.filter.register('after_render:html', function (html) {
  if (!html || !/<head[\s>]/i.test(html)) return html;
  const $ = cheerio.load(html, { decodeEntities: false });
  let canonical = $('link[rel="canonical"]').first().attr('href') || $('meta[property="og:url"]').first().attr('content') || '';
  if (!canonical) return html;
  const originalPath = routePath(canonical);
  const logicalPath = languagePath(canonical);
  const fixedEnglish = isFixedEnglishPath(logicalPath);
  const lang = isMurmurPath(logicalPath) ? 'zh' : (routePath(canonical).startsWith('/en/') || this.config.language === 'en' ? 'en' : 'zh');
  if (lang === 'en' && !routePath(canonical).startsWith('/en/') && !fixedEnglish) {
    canonical = `${siteBase(this.config)}/en${routePath(canonical)}`;
    $('link[rel="canonical"]').attr('href', canonical);
    setMeta($, 'meta[property="og:url"]', { property: 'og:url' }, canonical);
  }
  if (lang === 'en' && fixedEnglish) {
    canonical = `${siteBase(this.config)}${logicalPath}`;
    $('link[rel="canonical"]').attr('href', canonical);
    setMeta($, 'meta[property="og:url"]', { property: 'og:url' }, canonical);
  }
  const pathname = languagePath(canonical);
  const record = postsByUrl.get(canonical) || allPosts.find(item => languagePath(item.url).toLowerCase() === pathname.toLowerCase());
  if (record && canonical !== record.url) {
    canonical = record.url;
    $('link[rel="canonical"]').attr('href', canonical);
    setMeta($, 'meta[property="og:url"]', { property: 'og:url' }, canonical);
  }
  const noindex = shouldNoindex(pathname, $) || $('meta[name="robots"]').attr('content') === 'noindex';

  const pageNumber = pathname.match(/\/page\/(\d+)\/?$/);
  if (pageNumber) {
    const suffix = lang === 'en' ? `Page ${pageNumber[1]}` : `第 ${pageNumber[1]} 页`;
    const currentTitle = text($('title').text());
    if (currentTitle && !currentTitle.includes(suffix)) {
      const parts = currentTitle.split(/\s+\|\s+/);
      parts[0] = `${parts[0]} - ${suffix}`;
      const updatedTitle = parts.join(' | ');
      $('title').text(updatedTitle);
      setMeta($, 'meta[property="og:title"]', { property: 'og:title' }, updatedTitle);
      setMeta($, 'meta[name="twitter:title"]', { name: 'twitter:title' }, updatedTitle);
    }
  }

  $('html').attr('lang', lang === 'en' ? 'en' : 'zh-CN');
  setMeta($, 'meta[name="robots"]', { name: 'robots' }, noindex ? 'noindex,follow' : 'index,follow,max-image-preview:large');
  setMeta($, 'meta[name="googlebot"]', { name: 'googlebot' }, noindex ? 'noindex,follow' : 'index,follow,max-image-preview:large');
  $('meta[http-equiv="revisit-after"], meta[name="revisit-after"]').remove();

  const existingDescription = $('meta[name="description"]').attr('content');
  const genericDescription = String(this.config.description || '').trim();
  const wrongLanguageDescription = lang === 'en' ? /[\u3400-\u9fff]/.test(existingDescription || '') : !/[\u3400-\u9fff]/.test(existingDescription || '');
  const description = record
    ? ((!existingDescription || existingDescription.trim() === genericDescription || wrongLanguageDescription) ? (lang === 'en'
      ? `${record.title} — a technical essay from Hyacehila's Blog.`
      : `《${record.title}》：Hyacehila 博客中的技术文章。`) : existingDescription)
    : pageDescription(pathname, $, lang, noindex);
  if (description) {
    setMeta($, 'meta[name="description"]', { name: 'description' }, description);
    setMeta($, 'meta[property="og:description"]', { property: 'og:description' }, description);
    setMeta($, 'meta[name="twitter:description"]', { name: 'twitter:description' }, description);
  }
  setMeta($, 'meta[property="og:locale"]', { property: 'og:locale' }, lang === 'en' ? 'en_US' : 'zh_CN');

  if (record) {
    const image = postImage(canonical);
    setMeta($, 'meta[property="og:image"]', { property: 'og:image' }, image);
    setMeta($, 'meta[property="og:image:width"]', { property: 'og:image:width' }, '1200');
    setMeta($, 'meta[property="og:image:height"]', { property: 'og:image:height' }, '630');
    setMeta($, 'meta[property="og:image:type"]', { property: 'og:image:type' }, 'image/png');
    setMeta($, 'meta[name="twitter:image"]', { name: 'twitter:image' }, image);
    setMeta($, 'meta[name="twitter:card"]', { name: 'twitter:card' }, 'summary_large_image');
    $('.article-content h1').each((_, element) => {
      element.tagName = 'h2';
      $(element).addClass('seo-demoted-h1');
    });
  }
  if ($('.category-name').length) {
    const node = $('.category-name').first();
    node.replaceWith($('<h1 class="category-name"></h1>').html(node.html()));
  }
  if ($('.tag-name').length) {
    const node = $('.tag-name').first();
    node.replaceWith($('<h1 class="tag-name"></h1>').html(node.html()));
  }

  const paired = Boolean(record) || pathname === '/';
  if (paired) {
    setLink($, 'link[rel="alternate"][hreflang="zh-CN"]', { rel: 'alternate', hreflang: 'zh-CN', href: counterpart(canonical, 'zh') });
    setLink($, 'link[rel="alternate"][hreflang="en"]', { rel: 'alternate', hreflang: 'en', href: counterpart(canonical, 'en') });
    setLink($, 'link[rel="alternate"][hreflang="x-default"]', { rel: 'alternate', hreflang: 'x-default', href: counterpart(canonical, 'zh') });
  }
  if (lang === 'en') $('link[type="application/atom+xml"]').attr('href', '/en/feed.xml');

  addBreadcrumb($, record, canonical, lang, pathname);
  improveImages($, lang);
  updateStructuredData($, canonical, lang, record);
  conditionAssets($, pathname);
  return $.html();
}, 40);

function sitemapUrl(loc, alternate = true) {
  const zh = counterpart(loc, 'zh');
  const en = counterpart(loc, 'en');
  return `<url><loc>${escapeXml(loc)}</loc>${alternate ? `<xhtml:link rel="alternate" hreflang="zh-CN" href="${escapeXml(zh)}"/><xhtml:link rel="alternate" hreflang="en" href="${escapeXml(en)}"/><xhtml:link rel="alternate" hreflang="x-default" href="${escapeXml(zh)}"/>` : ''}</url>`;
}

function urlset(urls) {
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">\n${urls.join('\n')}\n</urlset>\n`;
}

hexo.extend.generator.register('seo-sitemaps', function (locals) {
  if (String(this.config.root || '/') === '/en/') return [];
  const base = siteBase(this.config);
  const seo = this.config.seo || {};
  const zhUrls = new Set(['/','/murmur/'].map(item => `${base}${item}`));
  const enUrls = new Set(['/en/']);
  const postPaths = [];
  locals.posts.forEach(post => {
    const url = canonicalFor(this.config, post);
    zhUrls.add(url);
    enUrls.add(counterpart(url, 'en'));
    postPaths.push(routePath(url));
  });
  locals.categories.forEach(category => {
    if (Number(category.length || 0) >= Number(seo.min_category_posts || 3)) enUrls.add(`${base}/${String(category.path).replace(/^\/+/, '')}`);
  });
  const indexedTags = new Set(list(seo.index_tags).map(String));
  locals.tags.forEach(tag => {
    if (indexedTags.has(String(tag.name))) enUrls.add(`${base}/${String(tag.path).replace(/^\/+/, '')}`);
  });
  ['/me/','/cv/','/projects/','/photos/','/footprints/','/categories/'].forEach(item => enUrls.add(`${base}${item}`));
  const index = `<?xml version="1.0" encoding="UTF-8"?>\n<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap><loc>${base}/sitemap-zh.xml</loc></sitemap><sitemap><loc>${base}/sitemap-en.xml</loc></sitemap></sitemapindex>\n`;
  const zhList = Array.from(zhUrls).filter(url => !Array.from(enUrls).includes(url));
  const enList = Array.from(enUrls);
  return [
    { path: 'sitemap.xml', data: index },
    { path: 'sitemap-zh.xml', data: urlset(zhList.sort().map(url => sitemapUrl(url, routePath(url) === '/' || routePath(url).startsWith('/blog/')))) },
    { path: 'sitemap-en.xml', data: urlset(enList.sort().map(url => sitemapUrl(url, routePath(url) === '/en/' || routePath(url).startsWith('/en/blog/')))) }
  ];
});
