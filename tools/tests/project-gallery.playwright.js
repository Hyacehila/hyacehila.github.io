async (page) => {
 const results=[];
 const check=(ok,label)=>{if(!ok)throw Error(label);results.push(label);};
 await page.goto('http://localhost:4000/projects/');
 await page.locator('.gallery-card').first().waitFor();
 for(const width of [1440,1024,768,390,320]) {
  await page.setViewportSize({width,height:1000});
  await page.locator('.gallery-card').last().scrollIntoViewIfNeeded();
  await page.evaluate(()=>document.fonts.ready);
  const state=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth,columns:getComputedStyle(document.querySelector('.gallery-grid')).gridTemplateColumns.split(' ').length,images:[...document.querySelectorAll('.gallery-cover img')].every(i=>i.complete&&i.naturalWidth>0)}));
  check(!state.overflow,`${width}: no horizontal overflow`);check(state.images,`${width}: all five covers loaded`);
  check(state.columns===(width>=1440?3:width>=768?2:1),`${width}: expected column count (${state.columns})`);
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:`output/playwright/projects-${width}.png`,fullPage:true});
 }
 await page.setViewportSize({width:1440,height:1000});
 for(let i=0;i<5;i++) {
  const details=page.locator('.gallery-details').nth(i);const summary=details.locator('summary');
  await summary.click();check(await details.getAttribute('open')!==null,`card ${i+1}: expands`);
  check(await details.locator('.gallery-description').evaluate(e=>e.scrollHeight<=e.clientHeight+1),`card ${i+1}: full description visible`);
  await summary.focus();await page.keyboard.press('Enter');check(await details.getAttribute('open')===null,`card ${i+1}: keyboard collapses`);
 }
 check(page.url()==='http://localhost:4000/projects/','expand controls never navigate');
 const link=page.locator('.gallery-link').first();
 check(await link.getAttribute('href')==='https://github.com/Hyacehila/AnalysisPosts','correct external project destination');
 await page.context().route('https://github.com/**',route=>route.fulfill({status:200,body:'External navigation verified'}));
 await page.locator('.gallery-card').nth(1).scrollIntoViewIfNeeded();
 const popupPromise=page.waitForEvent('popup');
 const cover=await page.locator('.gallery-card').nth(1).locator('.gallery-cover').boundingBox(); await page.mouse.click(cover.x+cover.width/2,cover.y+cover.height/2);
 const popup=await popupPromise;await popup.waitForLoadState();check(popup.url()==='https://github.com/Hyacehila/AnalysisPosts','cover opens correct external tab');await popup.close();
 check(await page.locator('.gallery-link').count()===2,'no invented links for unpublished projects');
 await page.evaluate(()=>{window.scrollTo(0,0);document.documentElement.classList.add('dark');});
 await page.mouse.move(0,0);
 await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
 await page.screenshot({path:'output/playwright/projects-dark.png',fullPage:true});
 await page.evaluate(()=>{document.documentElement.classList.remove('dark');document.querySelectorAll('[data-i18n]').forEach(e=>{const v=window.I18N.en[e.dataset.i18n];if(v)e.innerHTML=v;});});
 await page.screenshot({path:'output/playwright/projects-english.png',fullPage:true});
 await page.setViewportSize({width:640,height:1000});
 check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'English narrow layout does not overflow');
 await page.screenshot({path:'output/playwright/projects-english-narrow.png',fullPage:true});
 await page.emulateMedia({reducedMotion:'reduce'});
 check(await page.locator('.gallery-card').first().evaluate(e=>getComputedStyle(e).animationName)==='none','reduced motion disables entrance animation');
 const context=await page.context().browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
 const plain=await context.newPage();await plain.goto('http://localhost:4000/projects/');
 check(await plain.locator('.gallery-title').first().innerText()==='Lenny Compass','no-JS content remains readable');
 await plain.locator('.gallery-details summary').first().click();check(await plain.locator('.gallery-details').first().getAttribute('open')!==null,'no-JS expansion works');await context.close();
 return results;
}




