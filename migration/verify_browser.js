// Browser verification of the migrated Hexo site (http://localhost:4000).
// Drives Chromium through the acceptance checklist and prints PASS/FAIL lines.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

const BASE = 'http://localhost:4000';
const results = [];
function check(name, ok, extra) {
  results.push({ name, ok: !!ok, extra: extra || '' });
  console.log((ok ? 'PASS ' : 'FAIL ') + name + (extra ? '  :: ' + extra : ''));
}

function normalizeHref(href) {
  try {
    const url = new URL(href);
    url.hash = '';
    url.search = '';
    return url.href.replace(/\/$/, '');
  } catch (e) {
    return String(href || '').replace(/\/$/, '');
  }
}

function loadFriendLinks() {
  const linksPath = path.join(__dirname, '..', 'source', '_data', 'links.yml');
  const groups = yaml.load(fs.readFileSync(linksPath, 'utf8')) || [];
  return groups.flatMap(group => Array.isArray(group.list) ? group.list : [])
    .map(friend => friend && friend.link)
    .filter(Boolean)
    .map(normalizeHref);
}

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ ignoreHTTPSErrors: true });
  const page = await ctx.newPage();
  const consoleErrors = [];
  const bad404 = [];
  page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', e => consoleErrors.push('PAGEERROR: ' + (e && e.message ? e.message : (e && e.stack ? e.stack : 'unknown'))));
  page.on('response', r => {
    if (r.status() === 404) {
      var u = r.url();
      // cv.pdf is the intended placeholder; everything else is a real 404
      if (!/cv\.pdf/i.test(u)) bad404.push(u);
    }
  });

  // 1. HOME stream
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  const articleCount = await page.locator('article, .home-article-item, .recent-post-item, .article-item').count();
  check('home loads', (await page.title()).length > 0, 'title="' + (await page.title()) + '"');
  check('home shows article list', articleCount > 0, articleCount + ' article nodes');
  const navLinks = await page.locator('header a, nav a, .navbar a, #navbar a').evaluateAll(els =>
    els.map(a => a.getAttribute('href')).filter(Boolean));
  const hrefs = navLinks.join('|');
  check('navbar links to /archives/ /me/ /projects/ and About children',
    hrefs.includes('/archives/') && hrefs.includes('/me/') && hrefs.includes('/projects/') &&
      hrefs.includes('/footprints/') && hrefs.includes('/friends/') && hrefs.includes('/cv/'),
    [...new Set(navLinks)].join(', ').slice(0, 160));
  check('home identity sidebar present', (await page.locator('.home-identity-sidebar .home-identity-card').count()) === 1);
  check('home identity focus items', (await page.locator('.home-identity-focus-item').count()) === 5);
  check('home sidebar nav links restored', (await page.locator('.home-sidebar-container .sidebar-links .links').count()) === 4);
  check('home statistics are static',
    (await page.locator('.home-identity-statistics a').count()) === 0 &&
      (await page.locator('.home-identity-statistics .item').count()) === 3);
  const homeToggle = page.locator('#language-toggle');
  const focusBefore = await page.locator('.home-identity-focus-item [data-i18n="service-data-science-title"]').first().textContent();
  if (await homeToggle.count() > 0) {
    await homeToggle.click();
    await page.waitForTimeout(400);
    const focusAfter = await page.locator('.home-identity-focus-item [data-i18n="service-data-science-title"]').first().textContent();
    check('Home identity focus toggles language', focusBefore && focusAfter && focusBefore.trim() !== focusAfter.trim(),
      'before="' + (focusBefore || '').trim() + '" after="' + (focusAfter || '').trim() + '"');
  }

  // 2. ARTICLE with CJK permalink + math + mermaid
  const cjkUrl = BASE + '/blog/2026/01/22/' + encodeURIComponent('表格数据上仍旧是SOTA-基于Tree的模型') + '/';
  const resp = await page.goto(cjkUrl, { waitUntil: 'networkidle' });
  check('CJK-permalink article 200', resp && resp.status() === 200, 'status=' + (resp && resp.status()));
  check('article has KaTeX rendered', (await page.locator('.katex').count()) > 0, (await page.locator('.katex').count()) + ' katex nodes');

  // math post (Re0HF-04, previously failing)
  await page.goto(BASE + '/blog/2025/12/30/Re0HF-04/', { waitUntil: 'networkidle' });
  check('Re0HF-04 renders + katex', (await page.locator('.katex').count()) > 0, (await page.locator('.katex').count()) + ' katex');

  // mermaid post (VLM)
  await page.goto(BASE + '/blog/2026/01/20/' + encodeURIComponent('从-LLM-到-VLM,语言模型如何实现视觉理解') + '/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  const mermaidSvg = await page.locator('.mermaid svg, svg[id^="mermaid"]').count();
  check('VLM post mermaid rendered to SVG', mermaidSvg > 0, mermaidSvg + ' mermaid svg');

  // images post (shapley, chinese filenames) — Redefine lazyloads via data-src
  await page.goto(BASE + '/blog/2026/02/28/shapley-and-shap/', { waitUntil: 'networkidle' });
  const shapSrcs = await page.locator('img').evaluateAll(els =>
    els.map(im => im.getAttribute('data-src') || im.getAttribute('src') || '')
       .filter(s => s.indexOf('/assets/images/shap/') !== -1));
  // fetch each to confirm the CJK-named file is actually served (200)
  let shapOk = shapSrcs.length > 0;
  for (const s of shapSrcs) {
    const r = await page.request.get(s.startsWith('http') ? s : BASE + s);
    if (r.status() !== 200) shapOk = false;
  }
  check('shapley local CJK-named images served', shapOk, shapSrcs.length + ' shap imgs, all 200=' + shapOk);

  // 3. ARCHIVES
  await page.goto(BASE + '/archives/', { waitUntil: 'networkidle' });
  check('archives 200 + has entries', (await page.locator('a[href*="/blog/"]').count()) > 5,
    (await page.locator('a[href*="/blog/"]').count()) + ' post links');

  // 4. CATEGORIES + TAGS
  const catResp = await page.goto(BASE + '/categories/', { waitUntil: 'networkidle' });
  check('categories 200', catResp && catResp.status() === 200, 'status=' + (catResp && catResp.status()));
  const tagResp = await page.goto(BASE + '/tags/', { waitUntil: 'networkidle' });
  check('tags 200', tagResp && tagResp.status() === 200, 'status=' + (tagResp && tagResp.status()));

  // 5. ME page
  await page.goto(BASE + '/me/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(600);
  check('Me page focus cards removed', (await page.locator('.focus-card').count()) === 0);
  check('Me page starts with Education',
    (await page.locator('.me-page .page-section h2').first().getAttribute('data-i18n')) === 'education-title');
  const intro = await page.locator('[data-i18n="about-text-1"]').first().textContent();
  check('Me page i18n intro applied', intro && intro.trim().length > 10);

  // 6. PROJECTS
  await page.goto(BASE + '/projects/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(500);
  const projCards = await page.locator('.project-card').count();
  check('Projects page has 3 case cards', projCards === 3, projCards + ' cards');

  // 7. ABOUT: murmur + globe + cv + friends
  await page.goto(BASE + '/about/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2500); // allow fetch + globe lib
  const murmurItems = await page.locator('#murmur-timeline .murmur-item').count();
  check('About murmur timeline rendered', murmurItems > 0, murmurItems + ' murmur items');
  const globeCanvas = await page.locator('#globe-container canvas').count();
  check('About globe canvas present', globeCanvas > 0, globeCanvas + ' canvas');
  const friendCards = await page.locator('.friend-card').count();
  check('About friends cards render', friendCards > 0, friendCards + ' friends');
  const cvFrame = await page.locator('iframe.cv-frame').count();
  check('About CV iframe present', cvFrame === 1);
  // CV placeholder shows when pdf missing (expected now)
  await page.waitForTimeout(800);
  const phVisible = await page.locator('#cv-placeholder').isVisible().catch(() => false);
  check('CV placeholder shown for missing pdf', phVisible, 'placeholder visible=' + phVisible);

  // 7b. SWUP navigation: globe + toggle must survive in-site nav
  await page.goto(BASE + '/me/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(400);
  // click the About nav link (swup navigation, no full reload)
  const aboutNav = page.locator('a[href="/about/"]').first();
  if (await aboutNav.count() > 0) {
    await aboutNav.click();
    await page.waitForTimeout(3000);
    const swupUrl = page.url();
    const globeAfterNav = await page.locator('#globe-container canvas').count();
    const murmurAfterNav = await page.locator('#murmur-timeline .murmur-item').count();
    check('swup nav to About re-inits globe', /\/about\/?$/.test(swupUrl) && globeAfterNav > 0,
      'url=' + swupUrl + ' canvas=' + globeAfterNav);
    check('swup nav to About re-renders murmur', murmurAfterNav > 0, murmurAfterNav + ' items');
  }

  // 8. SEARCH data
  const searchResp = await page.goto(BASE + '/search.xml', { waitUntil: 'load' });
  check('search.xml 200', searchResp && searchResp.status() === 200);

  // Friends page is data-driven: every link in source/_data/links.yml should render.
  await page.goto(BASE + '/friends/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(500);
  const expectedFriendLinks = loadFriendLinks();
  const renderedFriendLinks = (await page.locator('a[href]').evaluateAll(els =>
    els.map(a => a.href))).map(normalizeHref);
  const missingFriendLinks = expectedFriendLinks.filter(link => !renderedFriendLinks.includes(link));
  check('Friends page links match data', expectedFriendLinks.length > 0 && missingFriendLinks.length === 0,
    expectedFriendLinks.length + ' expected, missing: ' + (missingFriendLinks.join(', ') || 'none'));

  // console errors summary. Excluded as known-benign:
  //  - cv.pdf 404 (intended placeholder until a real CV is added)
  //  - theme/Mermaid internal unhandled rejection "firstChild" (present even with
  //    our scripts disabled; diagrams still render — not ours to patch)
  const realErrors = consoleErrors.filter(e =>
    !/favicon|net::ERR|vercount|busuanzi|cdn|ZStatic|CDNJS|wallhaven|redefine-og|redefine-avatar|cv\.pdf|loading\.svg/i.test(e) &&
    !/PAGEERROR:\s*(Object|unknown)$/.test(e) &&
    !/firstChild/i.test(e) &&
    !/Failed to load resource.*404/i.test(e));  // URL-level 404s checked separately via bad404
  check('no fatal console/page errors (known-benign excluded)', realErrors.length === 0, realErrors.slice(0, 5).join(' | '));
  check('no unexpected 404s (cv.pdf placeholder excluded)', bad404.length === 0, [...new Set(bad404)].slice(0, 5).join(' | '));

  await browser.close();
  const failed = results.filter(r => !r.ok);
  console.log('\n==== ' + (results.length - failed.length) + '/' + results.length + ' checks passed ====');
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
