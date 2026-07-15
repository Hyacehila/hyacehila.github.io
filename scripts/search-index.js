/* global hexo */
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const cheerio = require('cheerio');
const MiniSearch = require('minisearch');
const SearchTokenizer = require('../source/assets/js/search-tokenizer.js');
const SearchEngine = require('../source/assets/js/search-engine.js');

const MAX_SECTION_LENGTH = 3400;
const SECTION_OVERLAP = 120;
const BLOCK_TAGS = new Set([
  'address', 'article', 'aside', 'blockquote', 'br', 'dd', 'div', 'dl', 'dt',
  'figcaption', 'figure', 'footer', 'header', 'hr', 'li', 'main', 'nav', 'ol',
  'p', 'section', 'table', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr', 'ul'
]);
const SKIP_TAGS = new Set(['canvas', 'math', 'noscript', 'script', 'style', 'svg']);
const HEADING_TAG = /^h[1-6]$/;

const cleanText = value => {
  return SearchTokenizer.normalizeText(value)
    .replace(/[\s\u00a0]+/g, ' ')
    .trim();
};

const countControlCharacters = value => {
  const matches = String(value || '').match(SearchTokenizer.CONTROL_CHARACTERS);
  return matches ? matches.length : 0;
};

const toList = value => {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  if (typeof value.toArray === 'function') return value.toArray();
  if (Array.isArray(value.data)) return value.data;
  return [value];
};

const toNameList = value => {
  return Array.from(new Set(toList(value)
    .map(item => {
      if (!item) return '';
      if (typeof item === 'string') return item;
      if (item.name) return item.name;
      if (item.data && item.data.name) return item.data.name;
      return String(item);
    })
    .map(cleanText)
    .filter(Boolean)));
};

const joinRoot = (root, targetPath) => {
  const normalizedRoot = `/${String(root || '/').replace(/^\/+|\/+$/g, '')}/`.replace(/\/{2,}/g, '/');
  const normalizedPath = String(targetPath || '').replace(/^\/+/, '');
  return `${normalizedRoot}${normalizedPath}`.replace(/\/{2,}/g, '/');
};

const hashValue = value => crypto.createHash('sha256').update(value).digest('hex');

const extractCodeTerms = value => {
  const matches = SearchTokenizer.normalizeText(value).match(
    /c\+\+|c#|node\.js|[A-Za-z_$][A-Za-z0-9_$]*(?:[.:/#-][A-Za-z0-9_$+#-]+)*/g
  ) || [];
  return Array.from(new Set(matches)).slice(0, 180).join(' ');
};

const splitSection = value => {
  const text = cleanText(value);
  if (!text) return [];
  if (text.length <= MAX_SECTION_LENGTH) return [text];

  const chunks = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(text.length, start + MAX_SECTION_LENGTH);
    if (end < text.length) {
      const candidate = text.slice(start + Math.floor(MAX_SECTION_LENGTH * 0.62), end);
      const boundary = Math.max(
        candidate.lastIndexOf('。'), candidate.lastIndexOf('！'), candidate.lastIndexOf('？'),
        candidate.lastIndexOf('. '), candidate.lastIndexOf('! '), candidate.lastIndexOf('? '),
        candidate.lastIndexOf('; '), candidate.lastIndexOf('；')
      );
      if (boundary >= 0) end = start + Math.floor(MAX_SECTION_LENGTH * 0.62) + boundary + 1;
    }
    const chunk = cleanText(text.slice(start, end));
    if (chunk) chunks.push(chunk);
    if (end >= text.length) break;
    start = Math.max(start + 1, end - SECTION_OVERLAP);
  }
  return chunks;
};

const mergeSectionChunks = chunks => {
  const merged = [];
  chunks.forEach(chunk => {
    const previous = merged[merged.length - 1];
    const canMerge = previous && previous.body.length + chunk.body.length + 1 <= MAX_SECTION_LENGTH;
    if (!canMerge) {
      merged.push({ ...chunk });
      return;
    }

    previous.body = cleanText(`${previous.body} ${chunk.body}`);
    if (chunk.heading && !previous.heading.includes(chunk.heading)) {
      previous.heading = cleanText([previous.heading, chunk.heading].filter(Boolean).join(' · '));
    }
    previous.codeTerms = cleanText(`${previous.codeTerms} ${chunk.codeTerms}`);
    if (!previous.anchor && chunk.anchor) previous.anchor = chunk.anchor;
  });
  return merged;
};

const extractSections = html => {
  const $ = cheerio.load(String(html || ''), null, false);
  $('.katex, .katex-mathml, .copy-button, .highlight-tools, .code-info-button, annotation').remove();
  const sections = [{ heading: '', anchor: '', parts: [], codeParts: [] }];
  let current = sections[0];

  const appendBoundary = () => {
    if (current.parts.length && current.parts[current.parts.length - 1] !== '\n') current.parts.push('\n');
  };

  const walk = node => {
    if (!node) return;
    if (node.type === 'text') {
      if (node.data) current.parts.push(node.data);
      return;
    }
    if (node.type !== 'tag') {
      (node.children || []).forEach(walk);
      return;
    }

    const tagName = String(node.name || '').toLowerCase();
    if (SKIP_TAGS.has(tagName)) return;
    if (HEADING_TAG.test(tagName)) {
      const heading = cleanText($(node).text());
      current = {
        heading,
        anchor: cleanText($(node).attr('id') || ''),
        parts: [],
        codeParts: []
      };
      sections.push(current);
      return;
    }
    if (tagName === 'pre') {
      current.codeParts.push($(node).text());
      appendBoundary();
      return;
    }
    if (tagName === 'code') {
      const code = $(node).text();
      current.codeParts.push(code);
      current.parts.push(` ${code} `);
      return;
    }

    if (BLOCK_TAGS.has(tagName)) appendBoundary();
    (node.children || []).forEach(walk);
    if (BLOCK_TAGS.has(tagName)) appendBoundary();
  };

  $.root().contents().each((index, node) => walk(node));
  const chunks = sections.flatMap(section => {
    const codeTerms = extractCodeTerms(section.codeParts.join('\n'));
    return splitSection(section.parts.join(' ')).map(body => ({
      heading: section.heading,
      anchor: section.anchor,
      body,
      codeTerms
    }));
  });
  return mergeSectionChunks(chunks);
};

const stripHtml = html => {
  const $ = cheerio.load(String(html || ''), null, false);
  $('.katex, .katex-mathml, script, style, svg, canvas, math, noscript').remove();
  return cleanText($.root().text());
};

const buildAliases = rawData => {
  const aliases = {};
  const groups = rawData && Array.isArray(rawData.groups) ? rawData.groups : [];
  groups.forEach(group => {
    const terms = Array.from(new Set((Array.isArray(group) ? group : [])
      .map(SearchTokenizer.normalizeForMatch)
      .filter(Boolean)));
    terms.forEach(term => {
      aliases[term] = terms.filter(candidate => candidate !== term);
    });
  });
  return aliases;
};

const getAliasData = locals => {
  const data = locals && locals.data;
  if (!data) return {};
  return data['search-aliases'] || data.searchAliases || {};
};

const getPostDate = post => {
  const value = post.updated || post.date;
  if (!value) return '';
  const date = value.toDate ? value.toDate() : new Date(value);
  return Number.isNaN(date.getTime()) ? '' : date.toISOString();
};

const createPostRecords = (post, root) => {
  const url = joinRoot(root, post.path);
  const postId = `p-${hashValue(url).slice(0, 16)}`;
  const titleEn = cleanText(post.title_en || post.title || '');
  const titleZh = cleanText(post.title_zh || post.title || titleEn);
  const tags = toNameList(post.tags);
  const categories = toNameList(post.categories);
  const excerptEn = stripHtml(post.excerpt_en || post.description_en || post.excerpt || post.description || '');
  const excerptZh = stripHtml(post.excerpt_zh || '');
  const excerpt = cleanText([excerptEn, excerptZh].filter(Boolean).join(' '));
  const date = getPostDate(post);
  const rawContent = String(post.content || post._content || '');
  const sections = extractSections(rawContent);
  const postData = {
    postId,
    url,
    titleEn,
    titleZh,
    excerpt,
    date,
    tags,
    categories,
    hidden: post.hidden === true
  };
  const records = [{
    id: `${postId}:meta`,
    postId,
    recordType: 'meta',
    titleEn,
    titleZh,
    heading: '',
    tags,
    categories,
    excerpt,
    body: '',
    codeTerms: '',
    anchorUrl: url
  }];

  sections.forEach((section, index) => {
    const anchorUrl = section.anchor ? `${url}#${encodeURIComponent(section.anchor)}` : url;
    records.push({
      id: `${postId}:section:${index}`,
      postId,
      recordType: 'section',
      titleEn: '',
      titleZh: '',
      heading: section.heading,
      tags: [],
      categories: [],
      excerpt: '',
      body: section.body,
      codeTerms: section.codeTerms,
      anchorUrl
    });
  });

  return {
    post: postData,
    records,
    sourceControlCharacters: countControlCharacters(rawContent)
  };
};

const resolveMiniSearchUmd = () => {
  const cjsEntry = require.resolve('minisearch');
  return path.resolve(path.dirname(cjsEntry), '..', 'umd', 'index.js');
};

const getMiniSearchVersion = () => {
  const cjsEntry = require.resolve('minisearch');
  const packagePath = path.resolve(path.dirname(cjsEntry), '..', '..', 'package.json');
  return JSON.parse(fs.readFileSync(packagePath, 'utf8')).version;
};

hexo.extend.generator.register('bm25_search_index', function (locals) {
  const startedAt = Date.now();
  const records = [];
  const posts = {};
  let sourceControlCharacters = 0;

  locals.posts.forEach(post => {
    if (post.published === false || post.search === false) return;
    const built = createPostRecords(post, this.config.root);
    posts[built.post.postId] = built.post;
    records.push(...built.records);
    sourceControlCharacters += built.sourceControlCharacters;
  });

  const coreRecords = records.filter(record => {
    return record.recordType === 'meta' || !posts[record.postId].hidden;
  });
  const archiveRecords = records.filter(record => {
    return record.recordType !== 'meta' && posts[record.postId].hidden;
  });
  const buildIndex = documents => {
    const miniSearch = new MiniSearch(SearchEngine.createMiniSearchOptions());
    miniSearch.addAll(documents);
    return JSON.stringify(miniSearch);
  };
  const serializedCoreIndex = buildIndex(coreRecords);
  const serializedArchiveIndex = buildIndex(archiveRecords);
  const coreRecordMap = Object.fromEntries(coreRecords.map(record => [record.id, record]));
  const archiveRecordMap = Object.fromEntries(archiveRecords.map(record => [record.id, record]));
  const dates = Object.values(posts).map(post => post.date).filter(Boolean).sort();
  const corePayload = {
    schemaVersion: SearchEngine.INDEX_SCHEMA_VERSION,
    engine: 'minisearch',
    engineVersion: getMiniSearchVersion(),
    generatedAt: dates.length ? dates[dates.length - 1] : null,
    posts,
    records: coreRecordMap,
    aliases: buildAliases(getAliasData(locals))
  };
  const archivePayload = {
    schemaVersion: SearchEngine.INDEX_SCHEMA_VERSION,
    records: archiveRecordMap
  };
  const serializedCoreData = JSON.stringify(corePayload);
  const serializedArchiveData = JSON.stringify(archivePayload);
  const contentHash = hashValue([
    serializedCoreIndex,
    serializedArchiveIndex,
    serializedCoreData,
    serializedArchiveData
  ].join('\0')).slice(0, 16);
  const coreIndexFile = `index-core-${contentHash}.json`;
  const archiveIndexFile = `index-archive-${contentHash}.json`;
  const coreDataFile = `documents-core-${contentHash}.json`;
  const archiveDataFile = `documents-archive-${contentHash}.json`;
  const shardStats = (indexData, documentData, shardRecords) => ({
    recordCount: shardRecords.length,
    indexBytes: Buffer.byteLength(indexData),
    indexGzipBytes: zlib.gzipSync(indexData).length,
    documentsBytes: Buffer.byteLength(documentData),
    documentsGzipBytes: zlib.gzipSync(documentData).length
  });
  const coreStats = shardStats(serializedCoreIndex, serializedCoreData, coreRecords);
  const archiveStats = shardStats(serializedArchiveIndex, serializedArchiveData, archiveRecords);
  const manifest = {
    schemaVersion: SearchEngine.INDEX_SCHEMA_VERSION,
    engine: 'MiniSearch',
    engineVersion: corePayload.engineVersion,
    hash: contentHash,
    postCount: Object.keys(posts).length,
    recordCount: records.length,
    sourceControlCharactersCleaned: sourceControlCharacters,
    shards: {
      core: { index: coreIndexFile, documents: coreDataFile, ...coreStats },
      archive: { index: archiveIndexFile, documents: archiveDataFile, ...archiveStats }
    },
    totalBytes: coreStats.indexBytes + coreStats.documentsBytes + archiveStats.indexBytes + archiveStats.documentsBytes,
    totalGzipBytes: coreStats.indexGzipBytes + coreStats.documentsGzipBytes + archiveStats.indexGzipBytes + archiveStats.documentsGzipBytes,
    generatedAt: corePayload.generatedAt
  };

  this.log.info(
    `[search] MiniSearch index: ${manifest.postCount} posts, ${manifest.recordCount} records, ` +
    `${Math.round(coreStats.indexGzipBytes / 1024)}+${Math.round(archiveStats.indexGzipBytes / 1024)} KiB index gzip, ` +
    `${Math.round(manifest.totalGzipBytes / 1024)} KiB total gzip, ${Date.now() - startedAt} ms`
  );

  return [
    { path: `assets/search/${coreIndexFile}`, data: serializedCoreIndex },
    { path: `assets/search/${archiveIndexFile}`, data: serializedArchiveIndex },
    { path: `assets/search/${coreDataFile}`, data: serializedCoreData },
    { path: `assets/search/${archiveDataFile}`, data: serializedArchiveData },
    { path: 'assets/search/manifest.json', data: JSON.stringify(manifest) },
    { path: `assets/vendor/minisearch-${corePayload.engineVersion}.js`, data: fs.readFileSync(resolveMiniSearchUmd(), 'utf8') }
  ];
});
