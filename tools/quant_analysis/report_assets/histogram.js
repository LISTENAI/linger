/* Common-bin reconstruction of already sampled histograms, not raw tensors. */
(function (root) {
  'use strict';
  function commonHistogram(group) {
    const sources = group.series.map(r => {
      const h = r.statistics.histogram || {}, counts = h.counts || [], edges = h.edges || [];
      const total = counts.reduce((sum, n) => sum + n, 0);
      return {r, counts, edges, total};
    }).filter(s => s.total > 0 && s.edges.length === s.counts.length + 1 && s.edges.every(Number.isFinite));
    if (!sources.length) return null;
    let low = Math.min(...sources.map(s => s.edges[0]));
    let high = Math.max(...sources.map(s => s.edges.at(-1)));
    if (low === high) { const pad = Math.max(Math.abs(low) * 0.05, 0.5); low -= pad; high += pad; }
    const n = Math.max(4, Math.min(64, Math.max(...sources.map(s => s.counts.length))));
    const width = (high - low) / n;
    const edges = Array.from({length:n + 1}, (_, i) => low + width * i);
    const series = sources.map(source => {
      const probabilities = new Array(n).fill(0);
      source.counts.forEach((count, i) => {
        if (!count) return;
        const lo = source.edges[i], hi = source.edges[i + 1], mass = count / source.total;
        if (lo === hi) {
          probabilities[Math.max(0, Math.min(n - 1, Math.floor((lo - low) / width)))] += mass;
        } else {
          for (let j = 0; j < n; j++) {
            const overlap = Math.max(0, Math.min(hi, edges[j + 1]) - Math.max(lo, edges[j]));
            probabilities[j] += mass * overlap / (hi - lo);
          }
        }
      });
      return {...source, probabilities};
    });
    return {low, high, edges, series, n};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = commonHistogram;
  else root.quantCommonHistogram = commonHistogram;
})(typeof globalThis !== 'undefined' ? globalThis : this);
