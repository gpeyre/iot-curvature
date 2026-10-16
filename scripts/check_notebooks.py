"""Execute the notebooks with the current Python interpreter, without a server."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT / 'notebooks' / 'executed'
destination.mkdir(exist_ok=True)
for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv[0] = sys.executable
    try:
        NotebookClient(notebook, km=manager, timeout=180, resources={'metadata': {'path': str(ROOT)}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.write(notebook, destination / path.name)
    print('PASS', path.name, flush=True)
