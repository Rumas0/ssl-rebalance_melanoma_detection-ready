"""
Checkpoint/resume utilities for Kaggle sessions.

Kaggle caps GPU sessions at 12 hours and does not preserve /kaggle/working
across sessions. Long runs must therefore save partial state and skip
already-completed work on restart.

Usage pattern:
    from shared.resume import run_and_save, already_done, save_checkpoint

    for seed in [42, 123, 2024]:
        tag = f'adaptive_seed{seed}'
        run_and_save(tag, lambda: train_one_seed(seed), results_dir=RESULT_DIR)

On session restart, completed tags are detected and skipped.
"""
import json
import os
import subprocess
import hashlib
from pathlib import Path
import torch


def _git_commit():
    """Best-effort: return the current commit hash, or 'unknown'."""
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)
        )
        return out.decode().strip()
    except Exception:
        return 'unknown'


def already_done(tag, results_dir):
    return (Path(results_dir) / f'{tag}.json').exists()


def run_and_save(tag, fn, results_dir, extra_meta=None):
    """
    If tag's result JSON exists, load and return it. Otherwise run fn(),
    which must return a JSON-serializable dict, and save it.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f'{tag}.json'

    if out_path.exists():
        print(f'[skip] {tag} already complete')
        return json.loads(out_path.read_text())

    print(f'[run]  {tag}')
    result = fn()

    payload = {
        'tag': tag,
        'git_commit': _git_commit(),
        'result': result,
    }
    if extra_meta:
        payload['meta'] = extra_meta
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f'[done] {tag} saved to {out_path}')
    return result


def save_checkpoint(path, model, optimizer=None, epoch=None, batch_idx=None, extra=None):
    """Save a resumable checkpoint, including optimizer and progress markers."""
    payload = {'model_state': model.state_dict()}
    if optimizer is not None:
        payload['opt_state'] = optimizer.state_dict()
    if epoch is not None:
        payload['epoch'] = epoch
    if batch_idx is not None:
        payload['batch_idx'] = batch_idx
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, map_location='cpu'):
    """Load a checkpoint saved by save_checkpoint. Returns (epoch, batch_idx)."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'opt_state' in ckpt:
        optimizer.load_state_dict(ckpt['opt_state'])
    return ckpt.get('epoch'), ckpt.get('batch_idx')