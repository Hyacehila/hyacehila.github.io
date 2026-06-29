const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push('PAGEERROR: ' + (e && e.message ? e.message : JSON.stringify(e))));
  page.on('console', m => { if (m.type() === 'error') errs.push('CONSOLE: ' + m.text()); });
  for (const path of ['/', '/me/', '/about/', '/projects/']) {
    errs.length = 0;
    await page.goto('http://localhost:4000' + path, { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    console.log('--- ' + path + ' ---');
    const real = errs.filter(e => !/favicon|vercount|busuanzi|ZStatic|CDNJS|wallhaven|redefine-og|redefine-avatar|loading\.svg/i.test(e));
    if (real.length === 0) console.log('  (clean)');
    real.slice(0, 8).forEach(e => console.log('  ' + e.slice(0, 200)));
  }
  await browser.close();
})();
