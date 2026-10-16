"""Copy only the four verified publication PDFs into a manuscript checkout."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--paper', type=Path, required=True, help='Manuscript directory containing main.tex')
args = parser.parse_args()
paper = args.paper.resolve()
if not (paper / 'main.tex').is_file(): parser.error('--paper must contain main.tex')
manifest = json.loads((ROOT / 'figures/data/manifest.json').read_text())
if manifest['preview']: raise ValueError('Cannot install reduced preview figures')
names = ['figure_1_cycles','figure_2_cosine','figure_3_annulus','figure_4_annulus_maps']
destination = paper / 'Pictures/generated'
destination.mkdir(parents=True,exist_ok=True)
record = {'description':'Generated publication figures from the self-contained numerical companion', 'figures':[]}
for name in names:
    source = ROOT / 'figures/generated' / (name+'.pdf')
    relative = str(source.relative_to(ROOT))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if manifest['checksums'][relative] != digest: raise ValueError('Figure differs from manifest: '+relative)
    target = destination / source.name
    shutil.copy2(source,target)
    record['figures'].append({'file':str(target.relative_to(paper)),'sha256':digest,'repository_file':relative})
(destination / 'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print('Installed four verified PDFs into', destination)
