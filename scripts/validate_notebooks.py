"""Execute the debugging notebooks in fresh kernels and retain outputs under results."""
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'results/notebook-validation'
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    for name in ('pandapickcube', 'pandapushcube'):
        notebook = nbformat.read(root / 'debugging' / f'{name}.ipynb', as_version=4)
        nbformat.validate(notebook)
        print(f'Executing {name}', flush=True)
        client = NotebookClient(
            notebook, timeout=1200, kernel_name='python3',
            resources={'metadata': {'path': str(root / 'debugging')}},
        )
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output / f'{name}.ipynb')
        print(f'Passed {name}: {output / (name + ".ipynb")}', flush=True)


if __name__ == '__main__':
    main()
