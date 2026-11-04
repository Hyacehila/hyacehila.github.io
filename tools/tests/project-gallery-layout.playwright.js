async (page) => {
 const results=[];
 await page.goto('http://localhost:4000/projects/');
 await page.emulateMedia({reducedMotion:'reduce'});
 const equal=async()=>{await page.evaluate(()=>new Promise(resolve=>{
 let last='',since=performance.now();
 const tick=()=>{const size=[...document.querySelectorAll('.gallery-card')].map(e=>{const r=e.getBoundingClientRect();return [r.width,r.height];}).join();if(size!==last){last=size;since=performance.now();}if(performance.now()-since>220)resolve();else requestAnimationFrame(tick);};tick();
 }));return page.waitForFunction(()=>{
  const cards=[...document.querySelectorAll('.gallery-card')].filter(e=>e.querySelector('.gallery-toggle').getAttribute('aria-expanded')==='false');
  const heights=cards.map(e=>e.getBoundingClientRect().height);
  return heights.length>0&&Math.max(...heights)-Math.min(...heights)<1;
 });};
 for(const lang of ['zh','en']) {
  await page.evaluate(lang=>{document.querySelectorAll('[data-i18n]').forEach(e=>{const v=window.I18N[lang][e.dataset.i18n];if(v)e.textContent=v;});document.dispatchEvent(new CustomEvent('i18n:applied'));},lang);
  for(const dark of [false,true]) {
   await page.evaluate(d=>document.documentElement.classList.toggle('dark',d),dark);
   for(const width of [1440,1024,768,390,320]) {
    await page.setViewportSize({width,height:1000});await equal();
    const geometry=await page.locator('.gallery-description').evaluateAll(es=>es.map(p=>{
     const b=p.querySelector('button').getBoundingClientRect(), box=p.getBoundingClientRect();
     const range=document.createRange();range.selectNodeContents(p.querySelector('.gallery-preview'));
     const rects=[...range.getClientRects()];const last=rects[rects.length-1];
     return {right:Math.abs(box.right-b.right),bottom:Math.abs(box.bottom-b.bottom),lines:box.height/parseFloat(getComputedStyle(p).lineHeight),overlap:last.right>b.left+.5,lastLine:Math.abs((last.top+last.bottom)/2-(b.top+b.bottom)/2)};
    }));
    if(geometry.some(g=>g.right>1||g.bottom>1||Math.abs(g.lines-4)>.1||g.overlap||g.lastLine>5))throw Error(JSON.stringify({lang,dark,width,geometry}));
    const before=await page.locator('.gallery-card').evaluateAll(es=>es.map(e=>e.getBoundingClientRect().height));
    await page.locator('.gallery-toggle').first().click();
    const opened=await page.locator('.gallery-card').evaluateAll(es=>es.map(e=>e.getBoundingClientRect().height));
    if(opened[0]<=before[0]||opened.slice(1).some((h,i)=>Math.abs(h-before[i+1])>1))throw Error(JSON.stringify({lang,dark,width,before,opened}));
    if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Horizontal overflow');
    await page.locator('.gallery-toggle').first().click();await equal();
    results.push({lang,mode:dark?'dark':'light',width,collapsedHeight:Math.round(before[0]),expandedHeight:Math.round(opened[0])});
   }
  }
 }
 await page.locator('.gallery-toggle').first().click();
 await page.setViewportSize({width:1440,height:1000});await equal();
 await page.locator('.gallery-toggle').first().click();await equal();
 await page.evaluate(()=>{document.documentElement.classList.remove('dark');document.querySelectorAll('[data-i18n]').forEach(e=>{const v=window.I18N.zh[e.dataset.i18n];if(v)e.textContent=v;});document.dispatchEvent(new CustomEvent('i18n:applied'));window.scrollTo(0,0);document.activeElement.blur();});await equal();
 await page.locator('.gallery-card').last().scrollIntoViewIfNeeded();
 await page.waitForFunction(()=>[...document.querySelectorAll('.gallery-cover img')].every(i=>i.complete&&i.naturalWidth>0));
 await page.evaluate(()=>window.scrollTo(0,0));await page.mouse.move(0,0);
 await page.screenshot({path:'output/playwright/projects-equal-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:1000});await equal();
 await page.screenshot({path:'output/playwright/projects-equal-mobile.png',fullPage:true});
 await page.setViewportSize({width:1440,height:1000});await equal();
 await page.evaluate(()=>{document.documentElement.style.zoom='2';window.dispatchEvent(new Event('resize'));});await equal();
 const zoom=await page.locator('.gallery-description').evaluateAll(es=>es.map(p=>{
  const b=p.querySelector('button').getBoundingClientRect(),box=p.getBoundingClientRect();
  const range=document.createRange();range.selectNodeContents(p.querySelector('.gallery-preview'));
  const rects=[...range.getClientRects()],last=rects[rects.length-1];
  return Math.abs(box.right-b.right)<1&&Math.abs(box.bottom-b.bottom)<1&&last.right<=b.left+.5&&Math.abs((last.top+last.bottom-b.top-b.bottom)/2)<10;
 }));
 if(zoom.some(v=>!v)||await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('200% zoom alignment or overflow');
 await page.screenshot({path:'output/playwright/projects-zoom-200.png',fullPage:true});
 await page.locator('.gallery-toggle').first().click();
 if(!await page.locator('.gallery-full').first().isVisible())throw Error('200% zoom expansion');
 await page.screenshot({path:'output/playwright/projects-expanded-zoom.png',fullPage:true});
 await page.locator('.gallery-toggle').first().click();
 await page.evaluate(()=>{document.documentElement.style.zoom='';window.dispatchEvent(new Event('resize'));});await equal();
 results.push({zoom:'200%',alignment:true,expansion:true});
 return results;
}


