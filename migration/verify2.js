const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const BASE = 'http://localhost:4000';
const results = [];
function check(name, ok, extra) { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (extra ? '  :: ' + extra : '')); }

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
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errs = [];
  page.on('pageerror', e => errs.push('PE:' + (e.message || '')));
  page.on('console', m => { if (m.type() === 'error') errs.push('CE:' + m.text()); });
  const bad404 = [];
  page.on('response', r => { if (r.status() === 404 && !/cv\.pdf/i.test(r.url())) bad404.push(r.url()); });

  // Home banner
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(800);
  check('home banner present', await page.locator('.home-banner, [class*=banner]').count() > 0);
  check('banner title = Hyacehila', (await page.locator('body').innerText()).includes('Hyacehila'));
  check('home identity sidebar present', await page.locator('.home-identity-sidebar .home-identity-card').count() === 1);
  check('home identity focus items (5)', await page.locator('.home-identity-focus-item').count() === 5);
  check('home sidebar nav links restored', await page.locator('.home-sidebar-container .sidebar-links .links').count() === 4);
  check('home statistics are static', await page.locator('.home-identity-statistics a').count() === 0 && await page.locator('.home-identity-statistics .item').count() === 3);

  // toggle works on Home (navbar 首页 -> Home)
  const navBefore = (await page.locator('header').innerText());
  await page.locator('#language-toggle').click();
  await page.waitForTimeout(500);
  const navAfter = (await page.locator('header').innerText());
  check('toggle changes Home chrome (首页->Home)', /首页|归档/.test(navBefore) && /Home|Archives/.test(navAfter), 'after has Home/Archives');
  await page.locator('#language-toggle').click(); // back to zh
  await page.waitForTimeout(300);

  // About dropdown items exist as links
  const subs = ['/murmur/', '/footprints/', '/friends/', '/cv/', '/contact/'];
  for (const s of subs) {
    const r = await page.request.get(BASE + s);
    check('sub-page 200 ' + s, r.status() === 200);
  }

  // murmur (说说) native renders 13 entries
  await page.goto(BASE + '/murmur/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(500);
  check('murmur 13 entries', await page.locator('#shuoshuo-content').count() === 13, await page.locator('#shuoshuo-content').count() + ' entries');

  // friends native renders all links from source/_data/links.yml
  await page.goto(BASE + '/friends/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(500);
  const expectedFriendLinks = loadFriendLinks();
  const renderedFriendLinks = (await page.locator('a[href]').evaluateAll(els =>
    els.map(a => a.href))).map(normalizeHref);
  const missingFriendLinks = expectedFriendLinks.filter(link => !renderedFriendLinks.includes(link));
  check('friends links match data', expectedFriendLinks.length > 0 && missingFriendLinks.length === 0,
    expectedFriendLinks.length + ' expected, missing: ' + (missingFriendLinks.join(', ') || 'none'));

  // footprints globe
  await page.goto(BASE + '/footprints/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  check('footprints globe canvas', await page.locator('#globe-container canvas').count() > 0);

  // me page layouts
  await page.goto(BASE + '/me/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(500);
  check('me: focus cards removed', await page.locator('.focus-card').count() === 0);
  check('me: education is first section', await page.locator('.me-page .page-section h2').first().getAttribute('data-i18n') === 'education-title');
  check('me: timeline (2 lists)', await page.locator('.timeline').count() === 2, await page.locator('.timeline-item').count() + ' items');
  check('me: research enumerated', await page.locator('.research-list > li').count() === 2);
  check('me: no comment heading', await page.locator('.comments-container').count() === 0);
  // me intro non-empty in zh default
  const intro = await page.locator('[data-i18n="about-text-1"]').first().textContent();
  check('me intro non-empty (zh)', (intro || '').trim().length > 10);

  // projects
  await page.goto(BASE + '/projects/', { waitUntil: 'networkidle' });
  check('projects 3 cards', await page.locator('.project-card').count() === 3);

  // a post: katex + mermaid + no leading H1 dup
  await page.goto(BASE + '/blog/2025/12/30/Re0HF-04/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  check('post katex', await page.locator('.katex').count() > 0);
  check('post mermaid svg', await page.locator('.mermaid svg, svg[id^=mermaid]').count() > 0);
  // first content heading should NOT be an <h1> duplicating title (theme renders title separately)
  const firstH = await page.locator('.markdown-body h1, .post-content h1, article h1').count();
  check('post body has no duplicate H1', firstH === 0, firstH + ' h1 in body');

  const realErrs = errs.filter(e => !/firstChild|PE:\s*$|PE:Object|cv\.pdf|favicon|vercount|busuanzi|ZStatic|CDNJS|wallhaven|redefine-og|redefine-avatar|loading\.svg|404/i.test(e));
  check('no fatal console errors', realErrs.length === 0, realErrs.slice(0,4).join(' | '));
  check('no unexpected 404s', bad404.length === 0, [...new Set(bad404)].slice(0,4).join(' | '));

  await browser.close();
  const passed = results.filter(Boolean).length;
  console.log('\n==== ' + passed + '/' + results.length + ' checks passed ====');
  process.exit(passed === results.length ? 0 : 1);
})().catch(e => { console.error('ERR', e); process.exit(2); });
