async (page) => {
 const results=[];
 const check=(ok,label)=>{if(!ok)throw Error(label);results.push(label);};
 await page.emulateMedia({reducedMotion:'no-preference'});
 await page.goto('http://localhost:4000/projects/');
 await page.locator('.gallery-card').first().waitFor();
 check(await page.locator('.gallery-card').count()===5,'all five projects are shown');
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
 check(await page.locator('.gallery-category,.gallery-index,.gallery-footer,.gallery-count').count()===0,'keywords, numbering and footer actions removed');
 check(await page.locator('.gallery-title a').count()===0,'titles do not navigate');
 for(let i=0;i<5;i++) {
  const paragraph=page.locator('.gallery-description').nth(i),button=paragraph.locator('button');
  const narrative=await paragraph.locator('.gallery-full').textContent();
  const preview=await paragraph.locator('.gallery-preview').textContent();
  check(Array.from(narrative).length>=200&&Array.from(narrative).length<=300,`card ${i+1}: full narrative is 200–300 characters`);
  check(preview.length>0&&preview.length<narrative.length&&preview.endsWith('…'),`card ${i+1}: preview fits the rendered space with ellipsis`);
  check(narrative.startsWith(preview.slice(0,-1))&&!/[。！？.!?]$/.test(preview.slice(0,-1)),`card ${i+1}: preview truncates the original narrative mid-sentence`);

  await paragraph.locator('.gallery-preview').click();
  check(await button.getAttribute('aria-expanded')==='false',`card ${i+1}: description text does not expand`);
  await button.click();check(await button.getAttribute('aria-expanded')==='true',`card ${i+1}: button expands`);
  check(await paragraph.locator('.gallery-full').isVisible(),`card ${i+1}: full description visible`);
  await button.focus();await page.keyboard.press('Enter');check(await button.getAttribute('aria-expanded')==='false',`card ${i+1}: keyboard collapses`);
 }
 await page.locator('.gallery-card').first().hover();
 await page.waitForFunction(()=>Number(getComputedStyle(document.querySelector('.gallery-card')).transform.split(',')[5]?.replace(')','')) < -4);
 check(true,'card hover lifts the whole card');
 await page.waitForFunction(()=>Number(getComputedStyle(document.querySelector('.gallery-cover img')).transform.split('(')[1]?.split(',')[0])>1.024);
 check(true,'cover hover reaches the expected image scale');
 await page.screenshot({path:'output/playwright/projects-hover.png',fullPage:true});
 await page.mouse.move(0,0);
 await page.waitForFunction(()=>getComputedStyle(document.querySelector('.gallery-card')).transform==='none');
 check(true,'card settles back after pointer leaves');
 check(page.url()==='http://localhost:4000/projects/','expand controls never navigate');
 const link=page.locator('.gallery-link').nth(1);
 check(await page.locator('.gallery-link').nth(3).getAttribute('href')==='https://github.com/Hyacehila/LaterOn','LaterOn repository link');
 check(await page.locator('.gallery-link').nth(4).getAttribute('href')==='https://github.com/Miraclexc/AutoTabGroups','AutoTabGroups repository link');
 check(await page.locator('.gallery-link').first().getAttribute('href')==='https://github.com/Hyacehila/lenny-compass','Lenny Compass links to its repository');
 check(await link.getAttribute('href')==='https://github.com/Hyacehila/Tweets2Insights','correct external project destination');
 await page.context().route('https://github.com/**',route=>route.fulfill({status:200,body:'External navigation verified'}));
 await page.locator('.gallery-card').first().scrollIntoViewIfNeeded();
 const popupPromise=page.waitForEvent('popup');
 const cover=await page.locator('.gallery-card').first().locator('.gallery-cover').boundingBox(); await page.mouse.click(cover.x+cover.width/2,cover.y+cover.height/2);
 const popup=await popupPromise;await popup.waitForLoadState();check(popup.url()==='https://github.com/Hyacehila/lenny-compass','cover opens correct external tab');await popup.close();
 check(await page.locator('a.gallery-cover').count()===5,'all five project images link to their repositories');
 // Dismiss the theme's existing image viewer before capturing page themes.
 await page.keyboard.press('Escape');
 await page.evaluate(()=>{window.scrollTo(0,0);document.documentElement.classList.add('dark');});
 await page.mouse.move(0,0);
 await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
 await page.screenshot({path:'output/playwright/projects-dark.png',fullPage:true});
 await page.evaluate(()=>{document.documentElement.classList.remove('dark');document.querySelectorAll('[data-i18n]').forEach(e=>{const v=window.I18N.en[e.dataset.i18n];if(v)e.innerHTML=v;});document.dispatchEvent(new CustomEvent('i18n:applied'));});
 await page.waitForFunction(()=>document.querySelector('.gallery-preview').textContent.startsWith('Lenny Compass helps'));
 check((await page.locator('.gallery-full').first().textContent()).startsWith((await page.locator('.gallery-preview').first().textContent()).slice(0,-1)),'language update regenerates preview from translated full text');
 await page.screenshot({path:'output/playwright/projects-english.png',fullPage:true});
 await page.setViewportSize({width:640,height:1000});
 check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'English narrow layout does not overflow');
 await page.screenshot({path:'output/playwright/projects-english-narrow.png',fullPage:true});
 await page.emulateMedia({reducedMotion:'reduce'});
 check(await page.locator('.gallery-card').first().evaluate(e=>getComputedStyle(e).transitionDuration)==='0s','reduced motion disables transitions');
 const context=await page.context().browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
 const plain=await context.newPage();await plain.goto('http://localhost:4000/projects/');
 check(await plain.locator('.gallery-title').first().innerText()==='Lenny Compass','no-JS content remains readable');
 check(await plain.locator('.gallery-full').first().isVisible(),'no-JS full description remains readable');check(await plain.locator('.gallery-toggle').first().isHidden(),'no-JS inactive toggle hidden');await context.close();
 return results;
}




