// Requires Playwright + Chromium. Run from the repository root:
// node tools/quant_analysis/test_report.cjs [path/to/report.html]
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

(async () => {
  const report = path.resolve(process.argv[2] || 'debug_alexnet/report.html');
  const html = fs.readFileSync(report, 'utf8');
  const payloadPattern = /(<script id="analysis-data" type="application\/json">)([\s\S]*?)(<\/script>)/;
  const payload = JSON.parse(html.match(payloadPattern)[2]);
  const groups = [...new Map(payload.records.map(r => [JSON.stringify([r.module_name,r.role]),r])).values()];
  const names = ['weight','bias','activation'];
  const isCategory = (r,type) => type === 'activation' ? r.role.startsWith('activation') : r.role === type;
  const counts = Object.fromEntries(names.map(type => [type,groups.filter(r => isCategory(r,type)).length]));
  const reference = payload.summary.config?.float_label || 'float';
  const temp = fs.mkdtempSync(path.join(os.tmpdir(),'quant-report-'));
  const browser = await chromium.launch({headless:true,executablePath:chromium.executablePath(),args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100},acceptDownloads:true});
    const errors = [], requests = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => { if (/^https?:/.test(r.url())) requests.push(r.url()); });
    await page.goto(pathToFileURL(report).href);
    await page.waitForSelector('.series-outline');
    assert.equal(await page.locator('.distribution-card').count(), groups.length);
    const switchMode = async (type,mode) => page.locator(`[data-category="${type}"] [data-mode="${mode}"]`).click();
    const seriesLabels = type => page.locator(`#${type}-gallery path[data-series]`).evaluateAll(nodes => [...new Set(nodes.map(n => n.dataset.series))]);
    const layout = type => page.locator(`#${type}-window`).evaluate(window => {
      const cards = [...window.querySelectorAll('.distribution-card')];
      const rows = [...new Set(cards.map(c => c.offsetTop))].sort((a,b)=>a-b);
      const secondBottom = Math.max(...cards.filter(c => c.offsetTop <= (rows[1] ?? rows[0])).map(c => c.offsetTop+c.offsetHeight));
      const style = getComputedStyle(window);
      return {rows:rows.length, height:window.clientHeight, total:window.scrollHeight,
        expected:secondBottom+parseFloat(style.paddingTop)+parseFloat(style.paddingBottom)};
    });
    for (const type of names) {
      assert.equal(await page.locator(`#${type}-gallery .plot`).count(),counts[type]);
      const box = await layout(type);
      if (box.rows > 2) { assert(Math.abs(box.height-box.expected)<=2,`${type}: two rows`); assert(box.total>box.height); }
    }
    // Scrolling a gallery leaves the page in place and can reach its last layer.
    await page.locator('#activation-window').scrollIntoViewIfNeeded();
    const pageY = await page.evaluate(()=>scrollY);
    await page.locator('#activation-window').evaluate(e=>{e.scrollTop=e.scrollHeight;});
    assert.equal(await page.evaluate(()=>scrollY),pageY);
    assert(await page.locator('#activation-window').evaluate(e=>e.scrollTop>0));
    const activationScroll = await page.locator('#activation-window').evaluate(e=>e.scrollTop);
    const axes = await page.locator('#weight-gallery .plot').first().locator('text').allTextContents();
    await switchMode('weight','float');
    assert.deepEqual(await seriesLabels('weight'),[reference]);
    assert.deepEqual(await page.locator('#weight-gallery .plot').first().locator('text').allTextContents(),axes);
    assert.equal(await page.locator('#activation-window').evaluate(e=>e.scrollTop),activationScroll);
    assert((await seriesLabels('bias')).length===2);
    assert(!(await page.locator('#weight-gallery .card-evidence').first().textContent()).includes('quantized'));
    const first = page.locator('#weight-gallery .plot').first();
    await first.focus(); await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('#chart-tooltip .tooltip-row').count(),1);
    await page.keyboard.press('Escape');
    const [download] = await Promise.all([page.waitForEvent('download'),page.locator('#weight-gallery .chart-export').first().click()]);
    const svgPath = path.join(temp,'float.svg'); await download.saveAs(svgPath);
    const svg = fs.readFileSync(svgPath,'utf8');
    assert.equal((svg.match(/data-series=/g)||[]).length,1); assert(!svg.includes('var(--'));
    await switchMode('weight','quant');
    assert(!(await seriesLabels('weight')).includes(reference));
    await switchMode('weight','all');
    for (const type of ['bias','activation']) {
      await switchMode(type,'float'); assert.deepEqual(await seriesLabels(type),[reference]);
      await switchMode(type,'quant'); assert(!(await seriesLabels(type)).includes(reference));
      await switchMode(type,'all');
    }
    // Case-insensitive literal substring search applies to all three galleries.
    await page.locator('#layer-search').fill('  FEATURES.0  ');
    for (const type of names) assert.equal(await page.locator(`#${type}-gallery .plot`).count(),groups.filter(r=>isCategory(r,type)&&r.module_name.toLowerCase().includes('features.0')).length);
    await page.locator('#layer-search').fill('<script>[no such layer]');
    assert.equal(await page.locator('.plot').count(),0); assert.equal(await page.locator('.empty').count(),3);
    await page.locator('#clear-search').click();
    assert.equal(await page.locator('.plot').count(),groups.length);
    // Each breakpoint keeps two actual rows, with no page overflow or distorted SVGs.
    for (const width of [1024,736,390,320,1440]) {
      await page.setViewportSize({width,height:1100}); await page.waitForTimeout(100);
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`overflow at ${width}`);
      assert(await page.locator('.plot').evaluateAll(nodes=>nodes.every(s=>Math.abs(s.getBoundingClientRect().width-s.viewBox.baseVal.width)<1)));
      for (const type of names) { const box=await layout(type); if(box.rows>2) assert(Math.abs(box.height-box.expected)<=2,`${type} at ${width}`); }
    }
    // Reuse the offline shell with controlled records for missing-run and anomaly cases.
    const writeFixture = records => {
      const fixture = path.join(temp,'fixture.html');
      const data = JSON.stringify({records,summary:payload.summary}).replace(/</g,'\\u003c');
      fs.writeFileSync(fixture,html.replace(payloadPattern,(_,start,old,end)=>start+data+end));
      return pathToFileURL(fixture).href;
    };
    const q = payload.records.find(r=>r.label!==reference), f=payload.records.find(r=>r.label===reference);
    const fixtureRecords=[];
    for(let i=0;i<30;i++) for(const original of [q,f]) {
      const r=structuredClone(original);r.module_name=`layer.${String(i).padStart(2,'0')}`;r.role='weight';
      r.anomaly={flag:i===29&&r.label!==reference,severity:'warning',reasons:['test drift']};fixtureRecords.push(r);
    }
    const single=structuredClone(q);single.module_name='quant-only';single.role='bias';single.anomaly={flag:false};fixtureRecords.push(single);
    const empty=structuredClone(q);empty.module_name='no-finite';empty.role='activation_output';empty.statistics.histogram={edges:[],counts:[]};empty.anomaly={flag:true,severity:'critical',reasons:['NaN']};fixtureRecords.push(empty);
    await page.goto(writeFixture(fixtureRecords));
    assert.equal(await page.locator('[data-category="bias"] [data-mode="float"]').isDisabled(),true);
    await switchMode('weight','float');await page.locator('#layer-search').fill('layer.00');
    await page.locator('.issue-link').filter({hasText:'layer.29'}).click();
    assert.equal(await page.locator('#layer-search').inputValue(),'');
    assert.equal(await page.locator('[data-category="weight"] [data-mode="all"]').getAttribute('aria-pressed'),'true');
    const target=page.locator(await page.evaluate(()=>location.hash));assert.match(await target.textContent(),/layer.29/);
    assert.match(await target.getAttribute('class'),/is-located/);
    assert.equal(await target.locator('path[data-series]').count(),2);
    assert(await page.locator('#weight-window').evaluate(e=>e.scrollTop>0));
    const inside=await target.evaluate(e=>{const r=e.getBoundingClientRect(),w=e.closest('.gallery-window').getBoundingClientRect();return {top:r.top,bottom:r.bottom,windowTop:w.top,windowBottom:w.bottom};});
    assert(inside.top>=inside.windowTop-1&&inside.bottom<=inside.windowBottom+1,JSON.stringify(inside));
    assert.match(await page.locator('#activation-gallery').textContent(),/没有有限采样值/);
    await page.goto(writeFixture([]));assert.equal(await page.locator('.empty').count(),3);
    assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
    console.log('PASS: two-row scrolling at 320–1440px, independent model switches, stable axes, search/clear/empty results, SVG export, missing runs, anomaly navigation, offline');
  } finally { await browser.close(); fs.rmSync(temp,{recursive:true,force:true}); }
})().catch(error=>{console.error(error);process.exitCode=1;});
