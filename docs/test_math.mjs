import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {random, normalPoints, transform, sampleData, hungarianMax, solveAt, curve} from './math.mjs';

function bruteForce(scores, n) {
  let best = -Infinity;
  function visit(row, used, sum) {
    if (row === n) { best = Math.max(best, sum); return; }
    for (let j = 0; j < n; j++) if (!(used & (1 << j))) visit(row + 1, used | (1 << j), sum + scores[row * n + j]);
  }
  visit(0, 0, 0); return best;
}
const rng = random(20261016);
let assignmentCases = 0;
for (let n = 1; n <= 7; n++) for (let repeat = 0; repeat < 12; repeat++) {
  const scores = Float64Array.from({length: n * n}, () => (rng() - .5) * 50);
  const result = hungarianMax(scores, n);
  assert.equal(new Set(result.assignment).size, n);
  assert.ok(Math.abs(result.score - bruteForce(scores, n)) < 1e-10);
  assignmentCases++;
}
const baseline = curve(sampleData({n: 50, seed: 4, delta: 0, model: 'cosine', sampling: 'paired'}));
assert.ok(Math.max(...baseline.map(p => p.gap)) < 1e-12, 'Paired identity baseline must vanish throughout the PSD slice');
const meanCheck = solveAt({X:[[1,0],[0,1]], Y:[[0,1],[1,0]], reference:[0,1]}, .5);
assert.equal(meanCheck.gap, .5, 'Score difference must be divided by n');
for (const r of [.001,.1,1,2,4]) {
  const y = transform([r,0], {model:'annulus'});
  const qTarget = (y[0] ** 2 - .8 ** 2) / (.85 ** 2 - .8 ** 2);
  assert.ok(Math.abs(qTarget - (-Math.expm1(-r*r/2))) < 2e-14, 'Annulus radial CDF must match Gaussian radial CDF');
}
let protocolCases = 0;
for (const model of ['cosine', 'annulus']) for (const sampling of ['paired', 'independent']) {
  const data = sampleData({n: 40, seed: 19, delta: .329, model, sampling});
  const values = curve(data);
  assert.ok(solveAt(data, .5).gap < 1e-12, 'Observed reference must be optimal at the true cost');
  for (let j = 1; j < values.length - 1; j++) assert.ok(values[j + 1].gap - 2 * values[j].gap + values[j - 1].gap > -1e-10, 'Gap must be convex on a uniform grid');
  if (model === 'annulus') for (const y of data.Y) assert.ok(Math.hypot(...y) >= .8 - 1e-14 && Math.hypot(...y) <= .85 + 1e-14);
  protocolCases++;
}
const moments = normalPoints(100000, 42).reduce((acc, [x,y]) => [acc[0] + x, acc[1] + y, acc[2] + x*x, acc[3] + y*y], [0,0,0,0]).map(v => v / 100000);
assert.ok(Math.abs(moments[0]) < .02 && Math.abs(moments[1]) < .02 && Math.abs(moments[2] - 1) < .03 && Math.abs(moments[3] - 1) < .03);
const origin = transform([0,0], {model: 'annulus'}); assert.equal(Math.hypot(...origin), .8);
assert.throws(() => sampleData({delta:1/3}));
assert.throws(() => sampleData({delta:-.1}));
assert.throws(() => curve(sampleData({n:2}),1));
const start = performance.now(); curve(sampleData({n:160, seed:2, delta:.3, sampling:'independent', model:'cosine'}));
console.log(JSON.stringify({passed:true, brute_force_assignment_cases:assignmentCases, protocol_cases:protocolCases,
  mean_normalization_check:true, annulus_CDF_checks:5,
  gaussian_moments:moments, independent_n160_curve_milliseconds:Math.round(performance.now()-start)}, null, 2));
