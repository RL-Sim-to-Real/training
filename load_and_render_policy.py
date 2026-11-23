"""Restore a modern run with its saved configuration and record a policy rollout."""
import argparse
import json
from pathlib import Path
from training_cli import main


def evaluate():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device_id', type=int, default=0)
    p.add_argument('--rollout_steps', type=int, default=150)
    opts = p.parse_args()
    manifest = json.loads((opts.run_dir / 'config.json').read_text())
    args = manifest['arguments']
    checkpoint_root = opts.run_dir / 'checkpoints'
    checkpoints = [d for d in checkpoint_root.iterdir()
                   if d.is_dir() and d.name.isdigit()] if checkpoint_root.exists() else []
    checkpoint = opts.checkpoint or (max(checkpoints, key=lambda d: int(d.name)) if checkpoints else None)
    if checkpoint is None:
        p.error('No numeric checkpoint found in this run; supply --checkpoint.')
    args.update(load_checkpoint_path=str(checkpoint.resolve()), play_only=True,
                record_video=True, device_id=opts.device_id, rollout_steps=opts.rollout_steps,
                use_tb=False, use_wandb=False, suffix='-evaluation')
    # Smoke network widths must match, but rollout length may be overridden.
    args['smoke'] = bool(args.get('smoke', False))
    argv = []
    for key, value in args.items():
        if value is None:
            continue
        if key == 'smoke':
            if value:
                argv.append('--smoke')
        elif isinstance(value, bool):
            argv.append('--' + ('' if value else 'no-') + key)
        else:
            argv.append(f'--{key}={value}')
    main(args['env_name'], argv)


if __name__ == '__main__':
    evaluate()
