// Capture full-page screenshots of key pages at desktop + mobile widths,
// plus the EN/ZH toggle states and the globe after it renders.
const { chromium } = require('playwright');
const fs = require('fs');

const BASE = 'http://localhost:4002';
const OUT = 'migration/shots';
fs.mkdirSync(OUT, { recursive: true });

const pages = [
  { name: 'home', path: '/' },
  { name: 'archives', path: '/archives/' },
  { name: 'categories', path: '/categories/' },
  { name: 'tags', path: '/tags/' },
  { name: 'me', path: '/me/' },
  { name: 'projects', path: '/projects/' },
  { name: 'about', path: '/about/' },
  { name: 'article-cjk', path: '/blog/2026/01/22/' + encodeURIComponent('表格数据上仍旧是SOTA-基于Tree的模型') + '/' },
  { name: 'article-math', path: '/blog/2025/12/30/Re0HF-04/' },
  { name: 'article-mermaid', path: '/blog/2026/01/20/' + encodeURIComponent('从-LLM-到-VLM,语言模型如何实现视觉理解') + '/' },
];

(async () => {
  const browser = await chromium.launch();

  // Desktop
  const dctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const dp = await dctx.newPage();
  for (const pg of pages) {
    await dp.goto(BASE + pg.path, { waitUntil: 'networkidle' });
    await dp.waitForTimeout(pg.name === 'about' ? 3500 : 1200);
    await dp.screenshot({ path: `${OUT}/desktop-${pg.name}.png`, fullPage: true });
    console.log('shot desktop-' + pg.name);
  }
  // toggle EN on Me + About
  for (const nm of ['me', 'about']) {
    await dp.goto(BASE + '/' + nm + '/', { waitUntil: 'networkidle' });
    await dp.waitForTimeout(nm === 'about' ? 3000 : 800);
    await dp.locator('#language-toggle').click();
    await dp.waitForTimeout(1200);
    await dp.screenshot({ path: `${OUT}/desktop-${nm}-EN.png`, fullPage: true });
    console.log('shot desktop-' + nm + '-EN');
  }
  await dctx.close();

  // Mobile
  const mctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true });
  const mp = await mctx.newPage();
  for (const nm of ['home', 'me', 'about']) {
    await mp.goto(BASE + '/' + nm + '/', { waitUntil: 'networkidle' });
    await mp.waitForTimeout(nm === 'about' ? 3500 : 1000);
    await mp.screenshot({ path: `${OUT}/mobile-${nm}.png`, fullPage: true });
    console.log('shot mobile-' + nm);
  }
  await mctx.close();

  await browser.close();
  console.log('DONE');
})().catch(e => { console.error('ERR', e); process.exit(1); });
