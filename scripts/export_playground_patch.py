"""Refresh the distributable patch after editing the pinned Playground checkout."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
repo = root / 'dependencies/mujoco_playground'
patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=repo)
for name in subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=repo, text=True).splitlines():
    if name.startswith('mujoco_playground/') and name.endswith('.py'):
        result = subprocess.run(['git', 'diff', '--no-index', '--', '/dev/null', name],
                                cwd=repo, stdout=subprocess.PIPE, check=False)
        if result.returncode != 1:
            raise RuntimeError(f'Cannot export {name}')
        patch += result.stdout
(root / 'patches/playground-modern.patch').write_bytes(patch)
