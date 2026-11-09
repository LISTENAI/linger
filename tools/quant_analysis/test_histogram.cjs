// Run: node tools/quant_analysis/test_histogram.cjs
const assert = require('node:assert/strict');
const histogram = require('./report_assets/histogram.js');
const series = (label, edges, counts) => ({label,statistics:{histogram:{edges,counts}}});
const near = (a,b) => assert(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const mass = s => s.probabilities.reduce((a,b)=>a+b,0);
// Different source ranges and bin widths occupy one common numerical domain.
let h = histogram({series:[series('quant',[-2,0,2],[50,50]),series('float',[-1,0,1],[50,50])]});
assert.equal(h.low,-2); assert.equal(h.high,2);
for(const s of h.series) near(mass(s),1);
assert.deepEqual(h.series[0].probabilities,[.25,.25,.25,.25]);
assert.deepEqual(h.series[1].probabilities,[0,.5,.5,0]);
// Identical empirical distributions keep identical silhouettes across runs.
h = histogram({series:[series('q',[-1,0,1],[30,70]),series('f',[-1,0,1],[3,7])]});
assert.deepEqual(h.series[0].probabilities,h.series[1].probabilities);
// Constant tensors remain a unit point mass, including negative and zero values.
for(const value of [-3,0,4]) {
  h=histogram({series:[series('q',[value,value],[100])]});
  assert(h.low<value&&h.high>value); near(mass(h.series[0]),1);
  assert.equal(h.series[0].probabilities.filter(x=>x>0).length,1);
}
// Zero-count and malformed sources have no fabricated distribution.
assert.equal(histogram({series:[series('q',[0,1],[0])]}),null);
assert.equal(histogram({series:[series('q',[],[1])]}),null);
assert.equal(histogram({series:[]}),null);
// A point and a broad distribution can be compared without losing their mass.
h=histogram({series:[series('q',[0,0],[20]),series('f',[-1,0,1],[1,1])]});
for(const s of h.series) near(mass(s),1);
console.log('PASS: common numerical bins, mass conservation, identical distributions, constants, empty sources');
