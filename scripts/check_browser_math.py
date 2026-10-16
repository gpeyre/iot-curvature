"""Compare the browser's numerical kernel against independent SciPy objectives."""
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
node = shutil.which('node')
if node is None: raise SystemExit('Node.js is needed for this browser/Python cross-check')
fixtures = []
for model in ('cosine','annulus'):
    for n in (7,31):
        rng = np.random.default_rng(n)
        X = rng.standard_normal((n,2))
        for paired in (True,False):
            Z = X if paired else rng.standard_normal((n,2))
            if model == 'cosine':
                Y = Z - .32*(np.sin(Z)+np.sin(Z.sum(axis=1))[:,None])
            else:
                r = np.linalg.norm(Z,axis=1)
                rho = np.sqrt(.8**2+(.85**2-.8**2)*(-np.expm1(-r*r/2)))
                Y = Z*(rho/r)[:,None]
            K = X @ Y.T
            ref = np.arange(n) if paired else linear_sum_assignment(K,maximize=True)[1]
            for u in (0,.23,.5,.9,1):
                K = u*np.outer(X[:,0],Y[:,0])+(1-u)*np.outer(X[:,1],Y[:,1])
                rows,cols = linear_sum_assignment(K,maximize=True)
                expected = (K[rows,cols].sum()-K[np.arange(n),ref].sum())/n
                fixtures.append({'X':X.tolist(),'Y':Y.tolist(),'reference':ref.tolist(),'u':u,'expected':max(0.,expected)})
program = f'''
import {{ solveAt }} from {json.dumps((ROOT/'docs/math.mjs').as_uri())};
let input='';
for await (const part of process.stdin) input+=part;
const fixtures=JSON.parse(input);
process.stdout.write(JSON.stringify(fixtures.map(f=>solveAt(f,f.u).gap)));
'''
run = subprocess.run([node,'--input-type=module','-e',program],input=json.dumps(fixtures),text=True,capture_output=True,check=True)
actual = json.loads(run.stdout)
error = max(abs(a-f['expected']) for a,f in zip(actual,fixtures))
assert len(actual) == len(fixtures) == 40 and error < 1e-11
result = {'passed':True,'paired_and_independent_cases':len(fixtures),'max_scipy_javascript_gap_difference':error}
destination = ROOT / 'validation'
destination.mkdir(exist_ok=True)
(destination / 'browser_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

