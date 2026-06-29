const { chromium } = require('playwright');
const fs = require('fs');
const BASE = 'http://localhost:4000';
const OUT = 'migration/shots2';
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const p = await ctx.newPage();

  // Home with banner — top of page (not full, to see the cover)
  await p.goto(BASE + '/', { waitUntil: 'networkidle' });
  await p.waitForTimeout(1500);
  await p.screenshot({ path: `${OUT}/home-banner-top.png` });           // viewport = cover
  // scroll down a bit to see feed over fixed banner
  await p.evaluate(() => window.scrollTo(0, 700));
  await p.waitForTimeout(600);
  await p.screenshot({ path: `${OUT}/home-scrolled.png` });

  // Language toggle on Home: click and screenshot navbar/sidebar in EN
  await p.evaluate(() => window.scrollTo(0, 0));
  await p.waitForTimeout(300);
  await p.locator('#language-toggle').click();
  await p.waitForTimeout(600);
  await p.screenshot({ path: `${OUT}/home-EN.png` });
  // toggle back
  await p.locator('#language-toggle').click();
  await p.waitForTimeout(400);

  // Me page (redesigned: cards + timeline + enumeration)
  await p.goto(BASE + '/me/', { waitUntil: 'networkidle' });
  await p.waitForTimeout(800);
  await p.screenshot({ path: `${OUT}/me-full.png`, fullPage: true });

  // Murmur (说说 native)
  await p.goto(BASE + '/murmur/', { waitUntil: 'networkidle' });
  await p.waitForTimeout(800);
  await p.screenshot({ path: `${OUT}/murmur.png`, fullPage: true });

  // Friends (native)
  await p.goto(BASE + '/friends/', { waitUntil: 'networkidle' });
  await p.waitForTimeout(800);
  await p.screenshot({ path: `${OUT}/friends.png`, fullPage: true });

  // Footprints (globe)
  await p.goto(BASE + '/footprints/', { waitUntil: 'networkidle' });
  await p.waitForTimeout(3500);
  await p.screenshot({ path: `${OUT}/footprints.png`, fullPage: true });

  // CV
  await p.goto(BASE + '/cv/', { waitUntil: 'networkidle' });
  await p.waitForTimeout(1200);
  await p.screenshot({ path: `${OUT}/cv.png` });

  // About dropdown open: hover the About nav item
  await p.goto(BASE + '/', { waitUntil: 'networkidle' });
  await p.waitForTimeout(800);
  // find the About/关于 navbar item and hover
  const about = p.locator('.navbar-item', { hasText: '关于' }).first();
  try {
    await about.hover();
    await p.waitForTimeout(600);
    await p.screenshot({ path: `${OUT}/about-dropdown.png` });
  } catch (e) { console.log('dropdown hover failed:', e.message); }

  await ctx.close();
  await browser.close();
  console.log('DONE shots2');
})().catch(e => { console.error('ERR', e); process.exit(1); });
