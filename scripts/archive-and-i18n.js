/* global hexo */
'use strict';

// 1) Override the Home (index) generator so posts marked `archived: true`
//    are excluded from the Home article stream — but they remain in the
//    archives / categories / tags generators (which use the full post set).
const pagination = require('hexo-pagination');

hexo.extend.generator.register('index', function (locals) {
  const config = this.config;
  // filter out archived posts for the Home feed only
  let posts = locals.posts.filter(p => p.archived !== true);
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
  locals.posts.forEach(p => {
    if (p.title_en || p.excerpt_en) {
      // Normalize to a single leading slash, no trailing index.html — matches
      // the <a href> the theme renders for each post.
      let key = String(p.path).replace(/index\.html$/, '');
      key = '/' + key.replace(/^\/+/, '');
      map[key] = {};
      if (p.title_en) map[key].title_en = String(p.title_en);
      if (p.excerpt_en) map[key].excerpt_en = String(p.excerpt_en);
    }
  });
  return {
    path: 'assets/data/post-i18n.json',
    data: JSON.stringify(map)
  };
});
