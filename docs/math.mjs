/** Dependency-free numerical kernel. All displayed gaps use unregularized assignments. */
export function random(seed = 1) {
  let state = Number(seed) >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function normalPoints(n, seed) {
  const rng = random(seed), points = [];
  for (let i = 0; i < n; i++) {
    const radius = Math.sqrt(-2 * Math.log(Math.max(rng(), Number.MIN_VALUE)));
    const angle = 2 * Math.PI * rng();
    points.push([radius * Math.cos(angle), radius * Math.sin(angle)]);
  }
  return points;
}

export function transform(point, {model = 'cosine', delta = 0.3, inner = 0.8, outer = 0.85} = {}) {
  const [x, y] = point;
  if (model === 'cosine') {
    const s = Math.sin(x + y);
    return [x - delta * (Math.sin(x) + s), y - delta * (Math.sin(y) + s)];
  }
  const r = Math.hypot(x, y);
  // The value at the Gaussian-null singleton {0} does not affect the pushforward.
  if (r === 0) return [inner, 0];
  const q = -Math.expm1(-r * r / 2);
  const rho = Math.sqrt(inner * inner + q * (outer * outer - inner * inner));
  return [rho * x / r, rho * y / r];
}

/** Deterministic samples; common X across models, deltas and sampling schemes. */
export function sampleData(config = {}) {
  const {n = 60, seed = 1, sampling = 'paired', model = 'cosine', delta = .3} = config;
  if (!Number.isInteger(n) || n < 1) throw new Error('n must be a positive integer');
  if (!['paired', 'independent'].includes(sampling)) throw new Error('Unknown sampling scheme');
  if (!['cosine', 'annulus'].includes(model)) throw new Error('Unknown transport model');
  if (model === 'cosine' && !(delta >= 0 && delta < 1/3)) throw new Error('Cosine delta must satisfy 0 ≤ delta < 1/3');
  const X = normalPoints(n, seed);
  const Z = sampling === 'paired' ? X : normalPoints(n, (Number(seed) ^ 0x9E3779B9) >>> 0);
  const Y = Z.map(point => transform(point, config));
  const reference = sampling === 'paired'
    ? Int32Array.from({length: n}, (_, i) => i)
    : hungarianMax(scoreMatrix(X, Y, 0.5), n).assignment;
  return {X, Y, reference, config: {...config, n, seed, sampling, model, delta}};
}

export function scoreMatrix(X, Y, u) {
  const n = X.length, matrix = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    const a = u * X[i][0], b = (1 - u) * X[i][1];
    for (let j = 0; j < n; j++) matrix[i * n + j] = a * Y[j][0] + b * Y[j][1];
  }
  return matrix;
}

/** Exact combinatorial maximizer up to IEEE-754 arithmetic; square uniform OT. */
export function hungarianMax(scores, n) {
  if (!Number.isInteger(n) || n < 1 || scores.length !== n * n) throw new Error('Assignment scores must be a nonempty square matrix');
  const rowPotential = new Float64Array(n + 1), colPotential = new Float64Array(n + 1);
  const matching = new Int32Array(n + 1), way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) {
    matching[0] = i;
    let j0 = 0;
    const min = new Float64Array(n + 1).fill(Infinity), used = new Uint8Array(n + 1);
    do {
      used[j0] = 1;
      const i0 = matching[j0];
      let d = Infinity, j1 = 0;
      for (let j = 1; j <= n; j++) if (!used[j]) {
        const cost = -scores[(i0 - 1) * n + j - 1] - rowPotential[i0] - colPotential[j];
        if (cost < min[j]) { min[j] = cost; way[j] = j0; }
        if (min[j] < d) { d = min[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { rowPotential[matching[j]] += d; colPotential[j] -= d; }
        else min[j] -= d;
      }
      j0 = j1;
    } while (matching[j0] !== 0);
    do { const j1 = way[j0]; matching[j0] = matching[j1]; j0 = j1; } while (j0);
  }
  const assignment = new Int32Array(n);
  for (let j = 1; j <= n; j++) assignment[matching[j] - 1] = j - 1;
  let score = 0;
  for (let i = 0; i < n; i++) score += scores[i * n + assignment[i]];
  return {assignment, score};
}

export function solveAt(data, u) {
  const n = data.X.length, scores = scoreMatrix(data.X, data.Y, u);
  const {assignment, score} = hungarianMax(scores, n);
  let referenceScore = 0;
  for (let i = 0; i < n; i++) referenceScore += scores[i * n + data.reference[i]];
  const rawGap = (score - referenceScore) / n;
  const tolerance = 5e-13 * (1 + Math.abs(score / n) + Math.abs(referenceScore / n));
  if (rawGap < -tolerance) throw new Error('Negative optimality gap exceeds roundoff tolerance');
  return {u, gap: Math.max(rawGap, 0), rawGap, assignment: Array.from(assignment),
    optimalScore: score / n, referenceScore: referenceScore / n};
}

export function curve(data, count = 41, progress = () => {}) {
  if (!Number.isInteger(count) || count < 2) throw new Error('A curve needs at least two evaluation nodes');
  return Array.from({length: count}, (_, i) => {
    const result = solveAt(data, i / (count - 1));
    progress(i + 1, count);
    return {u: result.u, gap: result.gap};
  });
}
