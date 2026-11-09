(() => {
  'use strict';
  const {records = [], summary = {}} = JSON.parse(document.getElementById('analysis-data').textContent);
  const el = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = value => value == null ? '—' : Number(value).toLocaleString('en', {maximumSignificantDigits:3});
  const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
  const referenceLabel = summary.config?.float_label || 'float';
  const quantLabel = summary.config?.model_label || 'quantized';
  const labels = [...new Set(records.map(r => r.label))];
  const severity = r => r.anomaly?.severity === 'critical' ? 2 : r.anomaly?.flag ? 1 : 0;
  const key = r => JSON.stringify([r.module_name, r.role]);
  const seriesColor = r => r.label === referenceLabel ? 'var(--float)' : 'var(--quant)';
  const tooltip = el('chart-tooltip');
  const groupMap = new Map();
  const roleCaption = role => role === 'weight' ? '权重 · Weight' : role === 'bias' ? '偏置 · Bias' : role.startsWith('activation_input') ? '输入激活 · ' + role : '输出激活 · ' + role;
  for (const r of records) {
    const k = key(r);
    if (!groupMap.has(k)) groupMap.set(k, {id:`distribution-${groupMap.size}`, name:r.module_name, role:r.role, series:[]});
    groupMap.get(k).series.push(r);
  }
  const groups = [...groupMap.values()];
  const categories = {
    weight: groups.filter(g => g.role === 'weight'),
    bias: groups.filter(g => g.role === 'bias'),
    activation: groups.filter(g => g.role.startsWith('activation')),
  };
  const requiredGroups = [...categories.weight, ...categories.bias, ...categories.activation];
  const modes = {weight:'all', bias:'all', activation:'all'};
  let searchQuery = '';
  for (const [type, list] of Object.entries(categories)) list.forEach(g => { g.category = type; });
  const visibleSeries = group => group.series.filter(r => modes[group.category] === 'all' ||
    (modes[group.category] === 'float' ? r.label === referenceLabel : r.label !== referenceLabel));
  const matchesSearch = group => group.name.toLocaleLowerCase().includes(searchQuery);
  const legend = series => series.map(r => `<span class="legend-item"><i class="swatch ${r.label === referenceLabel ? 'reference' : ''}"></i>${esc(r.label)}</span>`).join('');
  el('legend').innerHTML = legend(labels.map(label => ({label}))) + '<span class="legend-note">实线：量化 · 虚线：浮点</span>';
  el('distribution-count').textContent = requiredGroups.length;
  el('model-label').textContent = /alexnet/i.test(summary.model_factory || '') ? 'AlexNet' : '模型分析';
  el('run-context').textContent = `${new Set(records.map(r => r.module_name)).size} 个模块 · ${labels.length} 组模型数据${summary.batches != null ? ` · ${summary.batches} 个 Batch` : ''}`;
  el('rules').textContent = summary.anomaly_rules?.description || '有配对浮点数据时检查分布漂移；NaN/Inf 标记为数值错误。';
  el('record-count').textContent = `${records.length} 条`;

  function createCard(group) {
    const series = visibleSeries(group);
    const flags = group.series.filter(r => severity(r));
    const level = flags.some(r => severity(r) === 2) ? 'critical' : flags.length ? 'warning' : '';
    const target = series.find(r => r.label === quantLabel) || series[0];
    const reference = series.find(r => r.label === referenceLabel);
    const s = target.statistics;
    const meanDelta = reference && reference !== target && s.mean != null && reference.statistics.mean != null ? s.mean - reference.statistics.mean : null;
    const statsText = `均值 ${fmt(s.mean)} · 标准差 ${fmt(s.std)} · 零值 ${percent(s.zero_ratio)}`;
    return `<article class="glass distribution-card ${level} ${location.hash === '#' + group.id ? 'is-located' : ''}" id="${group.id}" data-role="${esc(group.role)}">
      <div class="card-heading"><div class="card-name"><h3>${esc(group.name)}</h3><span class="role-caption">${esc(roleCaption(group.role))}</span></div>${level ? `<span class="status-badge ${level}" aria-label="${level === 'critical' ? '数值错误' : '待复核'}">${level === 'critical' ? '!' : '▲'}</span>` : ''}<button class="chart-export" type="button" data-export="${group.id}-plot" aria-label="导出 ${esc(group.name)} ${esc(group.role)} SVG">SVG ↗</button></div>
      <svg class="plot" id="${group.id}-plot" role="img" tabindex="0" aria-label="${esc(group.name)} ${esc(roleCaption(group.role))}，${esc(statsText)}；方向键查看分箱数据"></svg>
      <div class="card-stats"><div class="mini-legend">${legend(series)}</div><span class="stat-delta">${meanDelta == null ? '采样分布' : `Δ均值 ${fmt(meanDelta)}`}</span></div>
      <p class="card-evidence">${esc(target.label)} · ${statsText}</p>
      ${flags.length ? `<div class="status-line">${flags.map(r => `<strong>${esc(r.label)} · ${severity(r) === 2 ? '数值错误' : '待复核'}</strong><ul>${r.anomaly.reasons.map(reason => `<li>${esc(reason)}</li>`).join('')}</ul>`).join('')}</div>` : ''}
    </article>`;
  }
  function matchingGroups(type) {
    return categories[type].filter(g => matchesSearch(g) && visibleSeries(g).length);
  }

  // Measure real grid rows rather than assuming a fixed card height: long layer
  // names and anomaly reasons may make individual rows taller.
  function sizeWindow(type) {
    const grid = el(`${type}-gallery`), viewport = el(`${type}-window`);
    const cards = [...grid.querySelectorAll('.distribution-card')];
    const tops = [...new Set(cards.map(card => card.offsetTop))].sort((a,b) => a-b);
    const firstRows = cards.filter(card => card.offsetTop <= (tops[1] ?? tops[0]));
    const bottom = Math.max(0, ...firstRows.map(card => card.offsetTop + card.offsetHeight));
    const style = getComputedStyle(viewport), padding = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
    viewport.style.maxHeight = cards.length && tops.length > 2 ? `${bottom + padding}px` : 'none';
  }

  function renderCategory(type) {
    const all = categories[type], list = matchingGroups(type);
    el(`${type}-count`).textContent = list.length;
    el(`${type}-summary`).textContent = `${list.length} / ${all.length} 张分布图`;
    const empty = !all.length ? '未采集到此类张量数据。' :
      !all.some(matchesSearch) ? '没有匹配的层，请尝试其他关键词或清空搜索。' : '匹配层未采集到所选模型的数据，请切换为“全部”。';
    el(`${type}-gallery`).innerHTML = list.length ? list.map(createCard).join('') : `<div class="glass empty">${empty}</div>`;
    document.querySelectorAll(`[data-category="${type}"] button`).forEach(button => {
      const available = button.dataset.mode === 'all' || all.some(g => g.series.some(r => button.dataset.mode === 'float' ? r.label === referenceLabel : r.label !== referenceLabel));
      button.disabled = !available;
      button.title = available ? '' : '未采集到此类模型数据';
      button.setAttribute('aria-pressed', String(button.dataset.mode === modes[type]));
    });
    list.forEach(draw);
    sizeWindow(type);
    el(`${type}-window`).scrollTop = 0;
  }

  function updateSearchStatus() {
    const count = Object.keys(categories).reduce((sum,type) => sum + matchingGroups(type).length, 0);
    el('search-status').textContent = `当前匹配 ${count} / ${requiredGroups.length} 张分布图${searchQuery ? ` · 层名包含“${el('layer-search').value.trim()}”` : ' · 可在各区域独立切换浮点 / 量化'}`;
    el('clear-search').disabled = !el('layer-search').value;
  }

  function renderGalleries() {
    tooltip.hidden = true;
    Object.keys(categories).forEach(renderCategory);
    updateSearchStatus();
  }

  function renderDiagnostics() {
    const flags = records.filter(r => severity(r)).sort((a,b) => severity(b) - severity(a));
    el('issue-count').textContent = flags.length;
    el('diagnostics').innerHTML = `<div class="diagnostic-summary"><span class="diagnostic-dot">${flags.length ? '!' : '✓'}</span><strong>${flags.length ? `${flags.length} 条记录需要关注` : '当前规则未检出分布异常'}</strong><span>${flags.length ? '点击问题可直达对应分布图。' : '可搜索层名，或在各区域滚动查看分布。'}</span><span class="diagnostic-detail">${summary.output_comparison ? `输出 MAE ${fmt(summary.output_comparison.mae)}` : '尚无配对输出误差'}</span></div>${flags.length ? `<div class="issue-list">${flags.map(r => {
      const group = groupMap.get(key(r));
      return `<a class="issue-link ${severity(r) === 2 ? 'critical' : 'warning'}" href="#${requiredGroups.includes(group) ? group.id : 'records-table'}"><strong>${severity(r) === 2 ? '●' : '▲'} ${esc(r.module_name)} / ${esc(r.role)}</strong><small>${esc(r.label)} · ${r.anomaly.reasons.map(esc).join('；')}</small></a>`;
    }).join('')}</div>` : ''}`;
  }

  // Reconstruct both sampled histograms on the SAME bins. Distribute a source
  // bin's probability by overlap (uniform within that bin); preserve point mass.
  // This never compares unrelated per-run bin indexes or invents raw samples.
  const commonHistogram = globalThis.quantCommonHistogram;
  requiredGroups.forEach(group => { group.histogram = commonHistogram(group); });
  const text = (x,y,value,extra='') => `<text x="${x}" y="${y}" ${extra}>${esc(value)}</text>`;

  function draw(group) {
    const svg = el(`${group.id}-plot`), width = Math.max(1, Math.round(svg.getBoundingClientRect().width)), height = 216;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('height', height);
    const title = `${group.name} / ${group.role}`;
    let content = `<title>${esc(title)}</title><desc>浮点与量化分布共享数值坐标，纵轴是统一分箱后的估计采样占比。</desc>`;
    const full = group.histogram;
    const selected = visibleSeries(group);
    const h = full && {...full, series:full.series.filter(s => selected.includes(s.r))};
    if (!h || !h.series.length) { svg.innerHTML = content + text(15,100,'所选模型没有有限采样值'); return; }
    const left = 47, right = width - 12, top = 39, bottom = height - 36;
    const x = v => left + (v - h.low) / (h.high - h.low) * (right - left);
    // Preserve both axes when switching models, so a mode change cannot make a
    // distribution appear to shrink or expand just because the plot rescaled.
    const maxP = Math.min(1, Math.max(...full.series.flatMap(s => s.probabilities)) * 1.18) || 1;
    const y = p => bottom - p / maxP * (bottom - top);
    for (let i = 0; i <= 2; i++) {
      const p = maxP * i / 2, yy = y(p);
      content += `<line x1="${left}" y1="${yy}" x2="${right}" y2="${yy}" stroke="var(--grid)"/>`;
      content += text(left - 7,yy + 4,percent(p),'text-anchor="end"');
    }
    content += text(left,21,'采样占比 · 统一分箱估计','class="axis-caption"');
    const nTicks = width < 330 ? 2 : 3;
    for (let i = 0; i <= nTicks; i++) {
      const v = h.low + (h.high - h.low) * i / nTicks;
      content += text(x(v),bottom + 20,fmt(v),`text-anchor="${i === 0 ? 'start' : i === nTicks ? 'end' : 'middle'}"`);
    }
    for (const [index, s] of h.series.entries()) {
      const ref = s.r.label === referenceLabel, color = seriesColor(s.r), gradient = `${group.id}-gradient-${index}`;
      content += `<defs><linearGradient id="${gradient}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${color}" stop-opacity="${ref ? '.19' : '.38'}"/><stop offset="100%" stop-color="${color}" stop-opacity=".025"/></linearGradient></defs>`;
      let outline = `M ${x(h.edges[0])} ${y(s.probabilities[0])}`;
      for (let i = 0; i < h.n; i++) {
        outline += ` H ${x(h.edges[i + 1])}`;
        if (i + 1 < h.n) outline += ` V ${y(s.probabilities[i + 1])}`;
      }
      const area = `${outline} L ${right} ${bottom} L ${left} ${bottom} Z`;
      content += `<path data-series="${esc(s.r.label)}" d="${area}" fill="url(#${gradient})"/>`;
      content += `<path class="series-outline" d="${outline}" fill="none" stroke="${color}" stroke-width="${ref ? '1.6' : '1.9'}" ${ref ? 'stroke-dasharray="4 3"' : ''}/>`;
      const peak = Math.max(...s.probabilities), peakIndex = s.probabilities.indexOf(peak), peakX = x((h.edges[peakIndex] + h.edges[peakIndex + 1]) / 2);
      content += `<circle cx="${peakX}" cy="${y(peak)}" r="7" fill="${color}" opacity=".09"/><circle cx="${peakX}" cy="${y(peak)}" r="2.6" fill="${color}" opacity=".7"/>`;
    }
    content += `<g class="hover-guide" hidden><line class="cursor-line" y1="${top}" y2="${bottom}" stroke="var(--muted)" stroke-opacity=".4" stroke-dasharray="3 3"/>${h.series.map(s => `<circle class="cursor-marker" r="4" fill="${seriesColor(s.r)}" stroke="#fff" stroke-width="1.5"/>`).join('')}</g><rect class="chart-hit" x="${left}" y="${top}" width="${right-left}" height="${bottom-top}" fill="transparent"/>`;
    svg.innerHTML = content;
    let keyboardIndex = 0, pinned = false;
    const hide = () => { svg.querySelector('.hover-guide').setAttribute('hidden',''); tooltip.hidden = true; };
    const show = (index, clientX, clientY, cursorX) => {
      const guide = svg.querySelector('.hover-guide'); guide.removeAttribute('hidden');
      const xx = cursorX ?? x((h.edges[index] + h.edges[index+1]) / 2);
      const line = guide.querySelector('line'); line.setAttribute('x1',xx); line.setAttribute('x2',xx);
      guide.querySelectorAll('circle').forEach((circle,i) => { circle.setAttribute('cx',xx); circle.setAttribute('cy',y(h.series[i].probabilities[index])); });
      tooltip.innerHTML = `<strong>${esc(group.name)} · ${esc(group.role)}</strong><div>[${fmt(h.edges[index])}, ${fmt(h.edges[index+1])}]</div>${h.series.map(s => `<div class="tooltip-row"><span>${esc(s.r.label)}</span><span>${percent(s.probabilities[index])}</span></div>`).join('')}`;
      tooltip.hidden = false;
      tooltip.style.left = `${Math.max(8,Math.min(innerWidth-tooltip.offsetWidth-10,clientX+14))}px`;
      tooltip.style.top = `${Math.max(8,Math.min(innerHeight-tooltip.offsetHeight-10,clientY+14))}px`;
      keyboardIndex = index;
    };
    const atPointer = event => {
      const bounds = svg.getBoundingClientRect(), cursorX = Math.max(left,Math.min(right,event.clientX-bounds.left));
      const index = Math.max(0,Math.min(h.n-1,Math.floor((cursorX-left)/(right-left)*h.n)));
      show(index,event.clientX,event.clientY,cursorX);
    };
    const hit = svg.querySelector('.chart-hit');
    hit.addEventListener('pointermove', event => { if (!pinned) atPointer(event); });
    hit.addEventListener('pointerleave', () => { if (!pinned) hide(); });
    hit.addEventListener('click', event => { pinned = !pinned; if (pinned) atPointer(event); else hide(); });
    svg.onkeydown = event => {
      if (event.key === 'Escape') { pinned = false; hide(); return; }
      if (!['ArrowLeft','ArrowRight'].includes(event.key)) return;
      event.preventDefault(); keyboardIndex = Math.max(0,Math.min(h.n-1,keyboardIndex+(event.key==='ArrowRight'?1:-1)));
      const bounds = svg.getBoundingClientRect(); show(keyboardIndex,bounds.left+width/2,bounds.top+height/2);
    };
    svg.onblur = hide;
  }

  function exportSvg(id) {
    const source = el(id), copy = source.cloneNode(true), style = getComputedStyle(document.documentElement);
    copy.querySelectorAll('.chart-hit,.hover-guide').forEach(node => node.remove());
    copy.removeAttribute('tabindex');
    copy.setAttribute('xmlns','http://www.w3.org/2000/svg');
    copy.setAttribute('width',source.viewBox.baseVal.width);
    copy.setAttribute('height',source.viewBox.baseVal.height + 25);
    copy.setAttribute('viewBox',`0 0 ${source.viewBox.baseVal.width} ${source.viewBox.baseVal.height+25}`);
    const group = requiredGroups.find(g => `${g.id}-plot` === id);
    const legendY = source.viewBox.baseVal.height + 7;
    let legendX = 47;
    for (const r of visibleSeries(group)) {
      copy.insertAdjacentHTML('beforeend',`<line x1="${legendX}" y1="${legendY-3}" x2="${legendX+15}" y2="${legendY-3}" stroke="${seriesColor(r)}" stroke-width="2" ${r.label===referenceLabel?'stroke-dasharray="4 3"':''}/>${text(legendX+21,legendY,r.label)}`);
      legendX += 110;
    }
    const xml = new XMLSerializer().serializeToString(copy).replace(/var\((--[\w-]+)\)/g,(_,name)=>style.getPropertyValue(name).trim());
    const svgStyle = `<style>text{fill:${style.getPropertyValue('--muted')};font:11px system-ui,sans-serif}.axis-caption{font-size:10px}</style>`;
    const start = xml.indexOf('>')+1;
    const blob = new Blob([xml.slice(0,start)+svgStyle+xml.slice(start)],{type:'image/svg+xml;charset=utf-8'});
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href=url; a.download=`${group.name}-${group.role}.svg`.replace(/[^A-Za-z0-9_.-]/g,'_'); a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  el('rows').innerHTML = records.map(r => {
    const s=r.statistics,g=groupMap.get(key(r)),sev=severity(r);
    const link = requiredGroups.includes(g) ? `<a href="#${g.id}">分布 ↗</a>` : '—';
    return `<tr><td>${link}</td><td>${esc(r.label)}</td><td>${esc(r.module_name)}</td><td>${esc(r.role)}</td><td class="reason ${sev===2?'critical':sev?'warning':''}">${sev===2?'● 数值错误':sev?'▲ 待复核':'未检出'}${r.anomaly?.reasons?.length?'<br>'+r.anomaly.reasons.map(esc).join('；'):''}</td>${['min','max','mean','std','p99'].map(k=>`<td>${fmt(s[k])}</td>`).join('')}<td>${percent(s.zero_ratio)}</td><td>${s.nan_count||0} / ${s.inf_count||0}</td></tr>`;
  }).join('');
  document.querySelector('.table-wrap').id='records-table';
  document.addEventListener('click', event => {
    const button = event.target.closest('.chart-export');
    if (button) exportSvg(button.dataset.export);
  });
  el('layer-search').addEventListener('input', () => {
    searchQuery = el('layer-search').value.trim().toLocaleLowerCase();
    renderGalleries();
  });
  el('clear-search').addEventListener('click', () => {
    el('layer-search').value = ''; searchQuery = ''; renderGalleries(); el('layer-search').focus();
  });
  document.querySelectorAll('.series-switch').forEach(control => control.addEventListener('click', event => {
    const button = event.target.closest('button');
    if (!button || button.disabled) return;
    modes[control.dataset.category] = button.dataset.mode;
    tooltip.hidden = true; renderCategory(control.dataset.category); updateSearchStatus();
  }));

  function revealDistribution(id) {
    const group = requiredGroups.find(g => g.id === id);
    if (!group) return;
    // Anomaly/table links remain usable even when their layer or run is hidden.
    if (!matchesSearch(group)) { el('layer-search').value = ''; searchQuery = ''; }
    modes[group.category] = 'all';
    renderGalleries();
    el(id).scrollIntoView({block:'nearest', behavior:'instant'});
  }
  document.addEventListener('click', event => {
    const link = event.target.closest('a[href^="#distribution-"]');
    if (link) { event.preventDefault(); history.pushState(null,'',link.hash); revealDistribution(link.hash.slice(1)); }
    const tableLink = event.target.closest('a[href="#records-table"]');
    if (tableLink) el('records-table').closest('details').open = true;
  });
  addEventListener('hashchange', () => revealDistribution(location.hash.slice(1)));
  renderDiagnostics();
  renderGalleries();
  if (location.hash.startsWith('#distribution-')) revealDistribution(location.hash.slice(1));
  let lastWidth = 0, resizePending = false;
  new ResizeObserver(entries => {
    const width = Math.round(entries[0].contentRect.width);
    if (width === lastWidth || resizePending) return;
    lastWidth = width; resizePending = true;
    requestAnimationFrame(()=>{
      resizePending=false; tooltip.hidden=true;
      for (const type of Object.keys(categories)) { matchingGroups(type).forEach(draw); sizeWindow(type); }
    });
  }).observe(el('weight-gallery'));
  // Capture nested window scrolling too, so a tooltip cannot float over another card.
  addEventListener('scroll',()=>{tooltip.hidden=true;},{passive:true,capture:true});
})();
