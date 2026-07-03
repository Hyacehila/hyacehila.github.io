/* global hexo */
'use strict';

const pagination = require('hexo-pagination');

const isHidden = post => post.hidden === true;
const isEnglishDefault = config => {
  const lang = config && config.language;
  if (Array.isArray(lang)) return lang[0] === 'en';
  return String(lang || '').split(/[-_]/)[0] === 'en';
};
const toList = value => {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  if (Array.isArray(value.data)) return value.data;
  if (typeof value.toArray === 'function') return value.toArray();
  return [value];
};
const toKeywordList = value => {
  return toList(value)
    .map(item => {
      if (!item) return '';
      if (typeof item === 'string') return item;
      if (item.name) return item.name;
      if (item.data && item.data.name) return item.data.name;
      return String(item);
    })
    .map(item => item.trim())
    .filter(Boolean);
};

// Keep post bodies as authored, but make article chrome English-first at build
// time. The original zh title/excerpt stay on the post model for the runtime
// language switcher.
hexo.extend.filter.register('before_post_render', function (data) {
  if (!isEnglishDefault(this.config)) return data;

  if (data.title_en) {
    if (!data.title_zh && data.title) data.title_zh = data.title;
    data.title = data.title_en;
  }

  if (data.excerpt_en) {
    if (!data.excerpt_zh && data.excerpt) data.excerpt_zh = data.excerpt;
    data.excerpt = data.excerpt_en;
    data.description = data.description_en || data.excerpt_en;
    data.og_description = data.og_description || data.description;
  }

  const keywords = Array.from(new Set([
    ...toKeywordList(data.tags),
    ...toKeywordList(data.categories)
  ]));
  if (keywords.length && !data.keywords) data.keywords = keywords.join(', ');

  return data;
});

// 1) Hide selected posts from the Home feed only. Archives, Categories, and
//    Tags keep their default all-post behavior from Hexo/Redefine.
hexo.extend.generator.register('index', function (locals) {
  const config = this.config;

  let posts = locals.posts.filter(p => !isHidden(p));
  posts = posts.sort(config.index_generator.order_by || '-date');
  posts.data = posts.data || [];
  posts.data.sort((a, b) => (b.sticky || 0) - (a.sticky || 0));

  const paginationDir = config.index_generator.pagination_dir || config.pagination_dir || 'page';
  const path = config.index_generator.path || '';

  return pagination(path, posts, {
    perPage: config.index_generator.per_page,
    layout: config.index_generator.layout || ['index', 'archive'],
    format: paginationDir + '/%d/',
    data: { __index: true }
  });
});

// 2) Emit a JSON map { "<post.path>": { title_en, excerpt_en } } so the
//    client-side EN/ZH toggle can swap blog titles/excerpts in lists
//    (Home / archives / categories / tags) without theme-template edits.
hexo.extend.generator.register('post_i18n_map', function (locals) {
  const map = {};

  function addRecord(key, rec) {
    if (!key) return;
    key = '/' + String(key).replace(/^\/+/, '').replace(/index\.html$/, '');
    if (key[key.length - 1] !== '/') key += '/';
    map[key] = rec;
  }

  locals.posts.forEach(p => {
    if (p.title_en || p.excerpt_en) {
      // Normalize to a single leading slash, no trailing index.html; this
      // matches the <a href> the theme renders for each post. CJK permalinks
      // may appear encoded in DOM hrefs, so emit both raw and encoded forms.
      let key = String(p.path).replace(/index\.html$/, '');
      key = '/' + key.replace(/^\/+/, '');
      const rec = {};
      if (p.title_zh) rec.title_zh = String(p.title_zh);
      else if (p.title && p.title !== p.title_en) rec.title_zh = String(p.title);
      if (p.title_en) rec.title_en = String(p.title_en);
      if (p.excerpt_zh) rec.excerpt_zh = String(p.excerpt_zh);
      if (p.excerpt_en) rec.excerpt_en = String(p.excerpt_en);
      addRecord(key, rec);
      try {
        const decoded = decodeURIComponent(key);
        addRecord(decoded, rec);
        addRecord(encodeURI(decoded), rec);
      } catch (e) {
        addRecord(encodeURI(key), rec);
      }
    }
  });
  return {
    path: 'assets/data/post-i18n.json',
    data: JSON.stringify(map)
  };
});
