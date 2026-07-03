/* global hexo */
'use strict';

const HTML_ENTITIES = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
  '#39': "'"
};
const seoByCanonical = new Map();

function decodeHtml(value) {
  return String(value || '').replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (match, entity) => {
    const key = entity.toLowerCase();
    if (key[0] === '#') {
      const code = key[1] === 'x' ? parseInt(key.slice(2), 16) : parseInt(key.slice(1), 10);
      return Number.isFinite(code) ? String.fromCodePoint(code) : match;
    }
    return HTML_ENTITIES[key] || match;
  });
}

function escapeHtmlAttr(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function getTagAttr(tag, name) {
  const pattern = new RegExp(`${name}\\s*=\\s*("([^"]*)"|'([^']*)'|([^\\s>]+))`, 'i');
  const match = tag.match(pattern);
  return match ? (match[2] || match[3] || match[4] || '') : '';
}

function replaceTagAttr(tag, name, value) {
  const encoded = escapeHtmlAttr(value);
  const pattern = new RegExp(`${name}\\s*=\\s*("([^"]*)"|'([^']*)'|([^\\s>]+))`, 'i');
  if (pattern.test(tag)) return tag.replace(pattern, `${name}="${encoded}"`);
  return tag.replace(/\s*\/?>$/, ` ${name}="${encoded}">`);
}

function getMetaTags(html) {
  return html.match(/<meta\b[^>]*>/gi) || [];
}

function getMetaContent(html, keyAttr, keyValue) {
  const key = String(keyValue).toLowerCase();
  for (const tag of getMetaTags(html)) {
    if (String(getTagAttr(tag, keyAttr)).toLowerCase() === key) {
      return getTagAttr(tag, 'content');
    }
  }
  return '';
}

function getMetaContents(html, keyAttr, keyValue) {
  const key = String(keyValue).toLowerCase();
  return getMetaTags(html)
    .filter(tag => String(getTagAttr(tag, keyAttr)).toLowerCase() === key)
    .map(tag => getTagAttr(tag, 'content'))
    .filter(Boolean);
}

function getCanonical(html) {
  const links = html.match(/<link\b[^>]*>/gi) || [];
  for (const tag of links) {
    if (String(getTagAttr(tag, 'rel')).toLowerCase() === 'canonical') {
      return getTagAttr(tag, 'href');
    }
  }
  return '';
}

function getTitle(html) {
  const ogTitle = getMetaContent(html, 'property', 'og:title');
  if (ogTitle) return decodeHtml(ogTitle).trim();
  const match = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  return match ? decodeHtml(match[1]).replace(/\s+/g, ' ').trim() : '';
}

function getDescription(html) {
  return decodeHtml(
    getMetaContent(html, 'name', 'description') ||
    getMetaContent(html, 'property', 'og:description')
  ).trim();
}

function normalizeUrl(url) {
  return String(url || '').replace(/\/index\.html$/, '/');
}

function normalizeSiteUrl(url) {
  const normalized = normalizeUrl(url);
  return normalized.endsWith('/') ? normalized : `${normalized}/`;
}

function isHomeUrl(url, siteUrl) {
  return normalizeUrl(url) === normalizeSiteUrl(siteUrl);
}

function compactObject(value) {
  if (Array.isArray(value)) {
    const list = value.map(compactObject).filter(item => item !== undefined);
    return list.length ? list : undefined;
  }
  if (value && typeof value === 'object') {
    const result = {};
    for (const [key, item] of Object.entries(value)) {
      const compacted = compactObject(item);
      if (compacted !== undefined) result[key] = compacted;
    }
    return Object.keys(result).length ? result : undefined;
  }
  if (value === '' || value === null || value === undefined) return undefined;
  return value;
}

function safeJson(value) {
  return JSON.stringify(value).replace(/<\/script/gi, '<\\/script');
}

function replaceOgUrl(html, canonical) {
  if (!canonical) return html;
  let didReplace = false;
  const updated = html.replace(/<meta\b[^>]*>/gi, tag => {
    if (String(getTagAttr(tag, 'property')).toLowerCase() !== 'og:url') return tag;
    didReplace = true;
    return replaceTagAttr(tag, 'content', canonical);
  });
  if (didReplace) return updated;
  return updated.replace('</head>', `<meta property="og:url" content="${escapeHtmlAttr(canonical)}">\n</head>`);
}

function stripHtml(value) {
  return decodeHtml(String(value || '').replace(/<[^>]*>/g, ' '))
    .replace(/\s+/g, ' ')
    .trim();
}

function toList(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  if (Array.isArray(value.data)) return value.data;
  if (typeof value.toArray === 'function') return value.toArray();
  return [value];
}

function toKeywordList(value) {
  return toList(value)
    .map(item => {
      if (!item) return '';
      if (typeof item === 'string') return item;
      if (item.name) return item.name;
      if (item.data && item.data.name) return item.data.name;
      return String(item);
    })
    .map(stripHtml)
    .filter(Boolean);
}

function buildPostCanonical(config, post) {
  if (!post) return '';
  if (post.permalink) return normalizeUrl(post.permalink);
  if (!post.path) return '';
  return normalizeUrl(normalizeSiteUrl(config.url) + String(post.path).replace(/^\/+/, ''));
}

function replaceMetaContent(html, keyAttr, keyValue, content) {
  if (!content) return html;
  let didReplace = false;
  const key = String(keyValue).toLowerCase();
  const updated = html.replace(/<meta\b[^>]*>/gi, tag => {
    if (String(getTagAttr(tag, keyAttr)).toLowerCase() !== key) return tag;
    didReplace = true;
    return replaceTagAttr(tag, 'content', content);
  });
  if (didReplace) return updated;
  return updated.replace(
    '</head>',
    `<meta ${keyAttr}="${escapeHtmlAttr(keyValue)}" content="${escapeHtmlAttr(content)}">\n</head>`
  );
}

function buildJsonLd(html, siteConfig) {
  const siteUrl = normalizeSiteUrl(siteConfig.url);
  const canonical = normalizeUrl(getCanonical(html) || getMetaContent(html, 'property', 'og:url'));
  if (!canonical) return null;

  const title = getTitle(html);
  const description = getDescription(html);
  const siteName = decodeHtml(getMetaContent(html, 'property', 'og:site_name') || siteConfig.title).trim();
  const author = decodeHtml(
    getMetaContent(html, 'property', 'article:author') ||
    getMetaContent(html, 'name', 'author') ||
    siteConfig.author
  ).trim();
  const language = Array.isArray(siteConfig.language) ? siteConfig.language[0] : siteConfig.language;
  const image = getMetaContent(html, 'property', 'og:image');
  const ogType = String(getMetaContent(html, 'property', 'og:type')).toLowerCase();

  if (ogType === 'article') {
    const tags = getMetaContents(html, 'property', 'article:tag').map(decodeHtml);
    return compactObject({
      '@context': 'https://schema.org',
      '@type': 'BlogPosting',
      mainEntityOfPage: {
        '@type': 'WebPage',
        '@id': canonical
      },
      headline: title,
      description,
      url: canonical,
      image: image ? [image] : undefined,
      datePublished: getMetaContent(html, 'property', 'article:published_time'),
      dateModified: getMetaContent(html, 'property', 'article:modified_time'),
      author: author ? { '@type': 'Person', name: author } : undefined,
      publisher: author ? { '@type': 'Person', name: author } : undefined,
      keywords: tags.length ? tags.join(', ') : getMetaContent(html, 'name', 'keywords'),
      inLanguage: language
    });
  }

  if (isHomeUrl(canonical, siteUrl)) {
    return compactObject({
      '@context': 'https://schema.org',
      '@graph': [
        {
          '@type': 'WebSite',
          name: siteName,
          url: siteUrl,
          description,
          publisher: author ? { '@type': 'Person', name: author } : undefined,
          inLanguage: language
        },
        {
          '@type': 'Blog',
          name: siteName,
          url: siteUrl,
          description,
          author: author ? { '@type': 'Person', name: author } : undefined,
          inLanguage: language
        }
      ]
    });
  }

  return compactObject({
    '@context': 'https://schema.org',
    '@type': 'WebPage',
    name: title,
    description,
    url: canonical,
    isPartOf: {
      '@type': 'WebSite',
      name: siteName,
      url: siteUrl
    },
    inLanguage: language
  });
}

hexo.extend.filter.register('before_generate', function (locals) {
  seoByCanonical.clear();
  if (!locals || !locals.posts) return;

  locals.posts.forEach(post => {
    const canonical = buildPostCanonical(this.config, post);
    if (!canonical) return;

    const description = stripHtml(
      post.description_en ||
      post.excerpt_en ||
      post.description ||
      post.excerpt
    );
    const keywords = Array.from(new Set([
      ...toKeywordList(post.tags),
      ...toKeywordList(post.categories)
    ])).join(', ');

    seoByCanonical.set(canonical, { description, keywords });
  });
});

hexo.extend.filter.register('after_render:html', function (html) {
  if (!html || !/<head[\s>]/i.test(html)) return html;

  const canonical = normalizeUrl(getCanonical(html) || getMetaContent(html, 'property', 'og:url'));
  const postSeo = seoByCanonical.get(canonical);
  let result = html;

  if (postSeo) {
    const keywords = postSeo.keywords || getMetaContents(result, 'property', 'article:tag')
      .map(decodeHtml)
      .join(', ');
    result = replaceMetaContent(result, 'name', 'description', postSeo.description);
    result = replaceMetaContent(result, 'property', 'og:description', postSeo.description);
    result = replaceMetaContent(result, 'name', 'keywords', keywords);
  }

  result = replaceOgUrl(result, canonical);

  if (/<script\b[^>]*type=["']application\/ld\+json["'][^>]*>/i.test(result)) {
    return result;
  }

  const jsonLd = buildJsonLd(result, this.config);
  if (!jsonLd) return result;

  return result.replace('</head>', `<script type="application/ld+json">${safeJson(jsonLd)}</script>\n</head>`);
}, 20);
