const $ = id => document.getElementById(id);
const namespace = 'http://www.w3.org/2000/svg';
const color = {paired: '#087f80', independent: '#8570ad', source: '#547eb8', target: '#de8b4e'};
const state = {data: null, current: null, values: [], otherValues: [], id: 0, worker: null, busy: true};
const defaults = {model: 'cosine', sampling: 'paired', n: 60, delta: 0.3, seed: 1, u: 0.5};
let timer;

function config() {
  return {model: $('model').value, sampling: $('sampling').value, n: Number($('n').value),
    delta: Number($('delta').value), seed: Number($('seed').value) >>> 0};
}
function u() { return Number($('u').value); }
function normalized() { return $('normalize').checked && !$('normalize').disabled; }
function divisor() { return normalized() ? Number($('delta').value) ** 2 : 1; }
function format(value, precise = false) {
  if (value === 0) return '0';
  return Math.abs(value) < 0.0001 || Math.abs(value) >= 1000
    ? value.toExponential(precise ? 4 : 1).replace('e-', 'e−')
    : value.toFixed(precise ? 6 : 4).replace(/0+$/, '').replace(/\.$/, '');
}
function node(tag, attributes = {}, text = '') {
  const element = document.createElementNS(namespace, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text) element.textContent = text;
  return element;
}
function append(parent, tag, attributes, text) { const element = node(tag, attributes, text); parent.append(element); return element; }
function label(svg, x, y, text, attributes = {}) { return append(svg, 'text', {x, y, ...attributes}, text); }

function updateLabels() {
  const c = config(), isCosine = c.model === 'cosine';
  $('n-value').textContent = c.n;
  $('delta-value').textContent = c.delta.toFixed(3);
  $('u-value').textContent = `u = ${u().toFixed(3).replace(/0$/, '')}`;
  $('delta').disabled = !isCosine;
  $('delta-control').classList.toggle('disabled', !isCosine);
  for (const b of document.querySelectorAll('[data-delta]')) b.disabled = !isCosine;
  $('normalize').disabled = !isCosine || c.delta === 0;
  $('normalize-label').classList.toggle('disabled', $('normalize').disabled);
  $('scale-caption').textContent = normalized() ? 'Mean gap per pair, divided by δ²' : 'Mean gap per pair';
  $('sample-badge').textContent = `${c.n} pairs · ${c.sampling === 'paired' ? 'paired' : 'independent'}`;
  $('reference-legend').hidden = !$('reference').checked;
  $('sample-note').textContent = c.sampling === 'paired'
    ? `Observation: identity pairing of (Xᵢ, T(Xᵢ)). ${isCosine && c.delta === 0 ? 'Here T = Id, so the paired loss is identically zero on the entire PSD slice. ' : ''}The target points are dependent on the source points; the n pairs are i.i.d. from the true OT plan. Seed ${c.seed}.`
    : `Observation: optimal assignment at u = ½ between independently sampled marginals, then held fixed. These empirical OT pairs are not i.i.d. samples from the population plan. Seed ${c.seed}.`;
}

function cloud() {
  const svg = $('cloud'); svg.replaceChildren();
  if (!state.data) { label(svg, 340, 185, 'Generating points…', {'text-anchor': 'middle'}); return; }
  const {X, Y, reference} = state.data;
  let radius = 1;
  for (const p of [...X, ...Y]) radius = Math.max(radius, Math.abs(p[0]), Math.abs(p[1]));
  radius *= 1.13;
  const scale = 119 / radius, sy = y => 198 - y * scale;
  const position = (point, right = false) => [(right ? 494 : 186) + point[0] * scale, sy(point[1])];
  for (const [center, title] of [[186, 'SOURCE  αₙ'], [494, 'TARGET  βₙ']]) {
    append(svg, 'rect', {x: center - 137, y: 52, width: 274, height: 286, rx: 9, fill: '#fafcfb', stroke: '#edf1ee'});
    append(svg, 'line', {x1: center - 120, y1: 198, x2: center + 120, y2: 198, stroke: '#e6ece8', 'stroke-width': 1});
    append(svg, 'line', {x1: center, y1: 70, x2: center, y2: 326, stroke: '#e6ece8', 'stroke-width': 1});
    label(svg, center, 32, title, {'text-anchor': 'middle', 'font-size': 10, 'letter-spacing': 1.4});
    label(svg, center + 116, 213, 'x₁', {'font-size': 9});
    label(svg, center + 7, 82, 'x₂', {'font-size': 9});
    label(svg, center, 358, `same scale · ±${radius.toFixed(1)}`, {'text-anchor': 'middle', 'font-size': 9, fill: '#9aa8a1'});
  }
  const connections = append(svg, 'g', {'aria-hidden': 'true'});
  if ($('reference').checked) for (let i = 0; i < X.length; i++) {
    const [x1, y1] = position(X[i]), [x2, y2] = position(Y[reference[i]], true);
    append(connections, 'line', {x1, y1, x2, y2, stroke: '#889994', 'stroke-width': 1,
      opacity: 0.34, 'stroke-dasharray': '4 3'});
  }
  if (state.current) for (let i = 0; i < X.length; i++) {
    const [x1, y1] = position(X[i]), [x2, y2] = position(Y[state.current.assignment[i]], true);
    append(connections, 'line', {x1, y1, x2, y2, stroke: '#13888a', 'stroke-width': 0.9,
      opacity: Math.min(0.35, 12 / X.length)});
  }
  for (const point of X) { const [cx, cy] = position(point); append(svg, 'circle', {cx, cy, r: 3.1, fill: color.source, opacity: .9, stroke: 'white', 'stroke-width': .5}); }
  for (const point of Y) { const [x, y] = position(point, true); append(svg, 'rect', {x: x - 2.8, y: y - 2.8, width: 5.6, height: 5.6, rx: .6, fill: color.target, opacity: .93, stroke: 'white', 'stroke-width': .5}); }
}

function loss() {
  const svg = $('loss'); svg.replaceChildren();
  const c = config(), norm = divisor(), left = 75, right = 650, top = 48, bottom = 318;
  const all = [...state.values, ...(state.otherValues || [])];
  const maximum = Math.max(0, ...all.map(p => p.gap / norm), state.current ? state.current.gap / norm : 0);
  const ymax = maximum > 1e-13 ? maximum * 1.13 : 0.001;
  const px = value => left + (right - left) * value, py = value => bottom - (bottom - top) * value / ymax;
  for (let i = 0; i <= 4; i++) {
    const value = ymax * i / 4, y = py(value);
    append(svg, 'line', {x1: left, x2: right, y1: y, y2: y, stroke: i ? '#edf1ee' : '#aebeb6', 'stroke-width': 1});
    label(svg, left - 10, y + 4, format(value), {'text-anchor': 'end', 'font-size': 10});
  }
  for (const value of [0, .25, .5, .75, 1]) {
    const x = px(value);
    append(svg, 'line', {x1: x, x2: x, y1: bottom, y2: bottom + 5, stroke: '#acbeb4'});
    label(svg, x, bottom + 24, value === .5 ? '½' : String(value), {'text-anchor': 'middle', 'font-size': 11, fill: value === .5 ? color.paired : '#788981'});
  }
  append(svg, 'line', {x1: px(.5), x2: px(.5), y1: top, y2: bottom, stroke: '#c7d9d0', 'stroke-width': 1, 'stroke-dasharray': '3 4'});
  label(svg, left, 24, normalized() ? 'MEAN GAP / δ²' : 'MEAN GAP', {'font-size': 9, 'letter-spacing': 1.3});
  label(svg, right, bottom + 45, 'cost parameter u', {'text-anchor': 'end', 'font-size': 10});
  label(svg, px(.5), top - 8, 'true cost', {'text-anchor': 'middle', 'font-size': 9, fill: '#99ada2'});
  // The δ=0 paired baseline is analytic and need not run a second assignment.
  if (c.model === 'cosine') append(svg, 'line', {x1: left, x2: right, y1: bottom - .5, y2: bottom - .5,
    stroke: '#879b90', 'stroke-width': 1.2, 'stroke-dasharray': '2 4'});
  const draw = (values, sampling, secondary = false) => {
    if (!values?.length) return;
    const points = values.map(p => `${px(p.u)},${py(p.gap / norm)}`).join(' ');
    append(svg, 'polyline', {points, fill: 'none', stroke: color[sampling],
      'stroke-width': secondary ? 1.8 : 2.7, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
      'stroke-dasharray': secondary ? '5 4' : 'none', opacity: secondary ? .83 : 1});
    for (const p of values) append(svg, 'circle', {cx: px(p.u), cy: py(p.gap / norm), r: secondary ? 1.4 : 1.8,
      fill: color[sampling], opacity: .8});
  };
  draw(state.otherValues, state.otherSampling, true); draw(state.values, c.sampling);
  const currentX = px(u());
  append(svg, 'line', {x1: currentX, x2: currentX, y1: top, y2: bottom, stroke: color[c.sampling], opacity: .35, 'stroke-width': 1});
  if (state.current && Math.abs(state.current.u - u()) < 1e-10) {
    const cy = py(state.current.gap / norm);
    append(svg, 'circle', {cx: currentX, cy, r: 5.7, fill: 'white', stroke: color[c.sampling], 'stroke-width': 2.2});
    append(svg, 'circle', {cx: currentX, cy, r: 2.2, fill: color[c.sampling]});
  }
  if (!all.length) label(svg, (left + right) / 2, 165, 'Computing exact assignments…', {'text-anchor': 'middle', fill: '#8da298', 'font-size': 12});
  const legend = $('curve-legend'); legend.replaceChildren();
  const item = (scheme, secondary = false) => {
    const span = document.createElement('span'), mark = document.createElement('i');
    mark.className = `line ${scheme}-line`; if (secondary) mark.style.borderTopStyle = 'dashed';
    span.append(mark, document.createTextNode(scheme === 'paired' ? 'True paired sample' : 'Independent marginals + OT'));
    legend.append(span);
  };
  item(c.sampling); if (state.otherValues?.length) item(state.otherSampling, true);
  if (c.model === 'cosine') {
    const span = document.createElement('span'); span.innerHTML = '<i class="line baseline-line"></i> δ = 0 paired baseline'; legend.append(span);
  }
}

function readout() {
  if (!state.current || Math.abs(state.current.u - u()) > 1e-10) {
    $('gap-value').textContent = '…'; $('changed-value').textContent = '…'; return;
  }
  $('gap-value').textContent = format(state.current.gap / divisor(), true);
  const changed = state.current.assignment.reduce((count, j, i) => count + (j !== state.data.reference[i]), 0);
  $('changed-value').textContent = `${changed} / ${state.data.X.length}`;
}

function sample() {
  clearTimeout(timer); updateLabels();
  state.id += 1; state.busy = true; state.values = []; state.otherValues = []; state.current = null; state.data = null;
  state.worker?.terminate();
  state.worker = new Worker(new URL('./worker.mjs', import.meta.url), {type: 'module'});
  state.worker.onmessage = ({data: result}) => {
    if (result.id !== state.id) return;
    if (result.type === 'points') { state.data = result.data; state.current = result.current; cloud(); readout(); }
    if (result.type === 'progress') { $('progress').value = result.value; $('status').textContent = `Evaluating the curves · ${Math.round(result.value * 100)}%`; }
    if (result.type === 'curves') {
      state.values = result.values; state.otherValues = result.otherValues; state.otherSampling = result.otherSampling;
      state.current = result.current; state.busy = false; $('progress').value = 1;
      $('status').textContent = `Ready · 41 evaluations per curve · n = ${config().n}`;
      $('export').disabled = false; cloud(); loss(); readout();
      if (Math.abs(state.current.u - u()) > 1e-10) current();
    }
    if (result.type === 'current' && Math.abs(result.current.u - u()) < 1e-10) { state.current = result.current; cloud(); loss(); readout(); }
    if (result.type === 'error') { $('status').textContent = `Computation failed: ${result.error}`; $('progress').value = 0; }
  };
  state.worker.onerror = () => { $('status').textContent = 'The worker could not load. Serve this folder over HTTP; see the numerical guide.'; };
  $('status').textContent = 'Computing exact, unregularized assignments…'; $('progress').value = 0; $('export').disabled = true;
  cloud(); loss(); readout();
  state.worker.postMessage({type: 'sample', id: state.id, config: config(), u: u(), compare: $('compare').checked});
}

function current() {
  updateLabels(); loss(); readout();
  clearTimeout(timer);
  timer = setTimeout(() => state.worker?.postMessage({type: 'current', id: state.id, u: u()}), 35);
}

function exportCSV() {
  const c = config(), rows = [['model','n','delta','seed','sampling','u','mean_gap','displayed_loss','divided_by_delta_squared']];
  for (const [values, scheme] of [[state.values, c.sampling], [state.otherValues, state.otherSampling]]) {
    for (const p of values || []) rows.push([c.model, c.n, c.model === 'cosine' ? c.delta : '', c.seed, scheme,
      p.u, p.gap, p.gap / divisor(), normalized()]);
  }
  const url = URL.createObjectURL(new Blob([rows.map(row => row.join(',')).join('\n') + '\n'], {type: 'text/csv'}));
  const a = document.createElement('a'); a.href = url; a.download = `iot-${c.model}-n${c.n}-seed${c.seed}.csv`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

for (const id of ['model', 'sampling', 'n', 'delta', 'compare']) $(id).addEventListener('change', sample);
for (const id of ['n', 'delta']) $(id).addEventListener('input', updateLabels);
$('seed').addEventListener('change', () => { if (!$('seed').value || !Number.isFinite(Number($('seed').value))) $('seed').value = 1; sample(); });
$('u').addEventListener('input', current);
$('reference').addEventListener('change', () => { updateLabels(); cloud(); });
$('normalize').addEventListener('change', () => { updateLabels(); loss(); readout(); });
$('new-seed').addEventListener('click', () => { $('seed').value = (Number($('seed').value) + 1) >>> 0; sample(); });
$('reset').addEventListener('click', () => { for (const [key, value] of Object.entries(defaults)) $(key).value = value; $('compare').checked = true; $('reference').checked = false; $('normalize').checked = false; sample(); });
$('export').addEventListener('click', exportCSV);
for (const button of document.querySelectorAll('[data-delta]')) button.addEventListener('click', () => { $('delta').value = button.dataset.delta; sample(); });
sample();
