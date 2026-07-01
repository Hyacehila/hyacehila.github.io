/* global hexo */
'use strict';

const pagination = require('hexo-pagination');

const fmtNum = num => num.toString().padStart(2, '0');
const isArchived = post => post.archived === true;

// 1) Home and Archives intentionally form a split:
//    Home shows non-archived posts; Archives shows archived posts only.
//    Categories / Tags keep using the full post set from their own generators.
hexo.extend.generator.register('index', function (locals) {
  const config = this.config;

  let posts = locals.posts.filter(p => !isArchived(p));
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

hexo.extend.generator.register('archive', function (locals) {
  const { config } = this;
  const archiveGenerator = Object.assign({
    per_page: typeof config.per_page === 'undefined' ? 10 : config.per_page,
    yearly: true,
    monthly: true,
    daily: false
  }, config.archive_generator);

  let archiveDir = config.archive_dir || 'archives';
  const paginationDir = config.pagination_dir || 'page';
  const perPage = archiveGenerator.per_page;
  const allPosts = locals.posts
    .filter(isArchived)
    .sort(archiveGenerator.order_by || '-date');
  const result = [];

  if (archiveDir[archiveDir.length - 1] !== '/') archiveDir += '/';

  function generate(path, posts, options = {}) {
    options.archive = true;

    result.push(...pagination(path, posts, {
      perPage,
      layout: ['archive', 'index'],
      format: paginationDir + '/%d/',
      data: options
    }));
  }

  function generateEmptyArchive() {
    result.push({
      path: archiveDir,
      layout: ['archive', 'index'],
      data: {
        base: archiveDir,
        total: 1,
        current: 1,
        current_url: archiveDir,
        posts: allPosts,
        prev: 0,
        prev_link: '',
        next: 0,
        next_link: '',
        archive: true
      }
    });
  }

  if (!allPosts.length) {
    generateEmptyArchive();
    return result;
  }

  generate(archiveDir, allPosts);

  if (!archiveGenerator.yearly) return result;

  const posts = {};

  allPosts.forEach(post => {
    const date = post.date;
    const year = date.year();
    const month = date.month() + 1;

    if (!Object.prototype.hasOwnProperty.call(posts, year)) {
      posts[year] = [[], [], [], [], [], [], [], [], [], [], [], [], []];
    }

    posts[year][0].push(post);
    posts[year][month].push(post);

    if (archiveGenerator.daily) {
      const day = date.date();
      if (!Object.prototype.hasOwnProperty.call(posts[year][month], 'day')) {
        posts[year][month].day = {};
      }

      (posts[year][month].day[day] || (posts[year][month].day[day] = [])).push(post);
    }
  });

  const { Query } = this.model('Post');
  const years = Object.keys(posts);

  for (let i = 0, len = years.length; i < len; i++) {
    const year = +years[i];
    const data = posts[year];
    const url = archiveDir + year + '/';

    if (!data[0].length) continue;

    generate(url, new Query(data[0]), { year });

    if (!archiveGenerator.monthly && !archiveGenerator.daily) continue;

    for (let month = 1; month <= 12; month++) {
      const monthData = data[month];

      if (!monthData.length) continue;

      if (archiveGenerator.monthly) {
        generate(url + fmtNum(month) + '/', new Query(monthData), {
          year,
          month
        });
      }

      if (!archiveGenerator.daily) continue;

      for (let day = 1; day <= 31; day++) {
        const dayData = monthData.day[day];
        if (!dayData || !dayData.length) continue;

        generate(url + fmtNum(month) + '/' + fmtNum(day) + '/', new Query(dayData), {
          year,
          month,
          day
        });
      }
    }
  }

  return result;
});

hexo.extend.filter.register('before_generate', function () {
  this.theme.setView('pages/archive/archive.ejs', `
<div class="archive-container shadow-none hover:shadow-none sm:shadow-redefine sm:hover:shadow-redefine-hover">
  <% const archivePosts = page.posts || []; %>
  <% if (archivePosts.length) { %>
    <%- partial('utils/posts-list', {posts: archivePosts}) %>
  <% } else { %>
    <div class="archive-empty px-6 py-10 text-third-text-color text-center">&#26242;&#26080;&#24402;&#26723;&#25991;&#31456;</div>
  <% } %>
</div>
`);
});

// 2) Emit a JSON map { "<post.path>": { title_en, excerpt_en } } so the
//    client-side EN/ZH toggle can swap blog titles/excerpts in lists
//    (Home / archives / categories / tags) without theme-template edits.
hexo.extend.generator.register('post_i18n_map', function (locals) {
  const map = {};
  locals.posts.forEach(p => {
    if (p.title_en || p.excerpt_en) {
      // Normalize to a single leading slash, no trailing index.html; this
      // matches the <a href> the theme renders for each post.
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
