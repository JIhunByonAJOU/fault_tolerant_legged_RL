"""Read-only checkpoint evaluation sweep; never trains or changes checkpoints."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'logs/jt_wim243_failure_fullrange_onset/Aug14_00-37-37_'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    p.add_argument('--checkpoints', type=int, nargs='+', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[101])
    args = p.parse_args()
    folder = Path(args.output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed in args.seeds:
        for iteration in args.checkpoints:
            checkpoint = RUN / ('model_{}.pt'.format(iteration))
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            target = folder / ('student_{}_seed{}.jsonl'.format(iteration, seed))
            if target.exists():
                raise FileExistsError(target)
            cmd = [sys.executable, '-m', 'legged_gym.scripts.evaluate_onset_v2',
                   '--checkpoint-path', str(checkpoint), '--policy-mode', 'student',
                   '--output', str(target), '--replicates', '48', '--headless',
                   '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--seed', str(seed)]
            manifest.append(cmd)
            (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            print('START', iteration, seed, flush=True)
            with target.with_suffix('.log').open('w') as log:
                subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)
            # Common configuration must yield exactly matched physical initialization.
            old = ROOT / ('logs/evaluations/onset-v2-diagnostic-20260928/student_seed{}.jsonl'.format(seed))
            a = json.loads(target.open().readline())
            b = json.loads(old.open().readline())
            assert a['initial_state_hashes'] == b['initial_state_hashes']
            assert a['specs_sha256'] == b['specs_sha256']
            print('DONE (initial states matched)', iteration, seed, flush=True)

if __name__ == '__main__':
    main()
