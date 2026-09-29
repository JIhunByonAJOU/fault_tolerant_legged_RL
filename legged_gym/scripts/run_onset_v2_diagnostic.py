"""Run sequential GPU diagnostics, reject unmatched initial states, aggregate results."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = {
    'teacher': 'logs/official_wim_teacher243_failure_fullrange_fromscratch/Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange/model_43000.pt',
    'student': 'logs/jt_wim243_failure_fullrange_onset/Aug14_00-37-37_/model_71500.pt',
}
METRICS = ['survived_post_failure_horizon', 'recovered_stable_window', 'tracking_time_fraction',
           'tracking_time_fraction_strict', 'tracking_time_fraction_loose', 'final_5s_tracking_fraction',
           'post_command_vx_rmse', 'post_command_vy_rmse', 'post_yaw_rate_rmse', 'post_mean_vx',
           'failed_before_onset']


def aggregate(rows):
    result = {'n': len(rows)}
    for key in METRICS:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = statistics.mean(values) if values else None
    result['survived_without_legacy_recovery'] = statistics.mean(
        r['survived_post_failure_horizon'] and not r['recovered_stable_window'] for r in rows)
    return result


def summarize(folder, seeds):
    data = {}
    for mode in CHECKPOINTS:
        data[mode] = []
        for seed in seeds:
            records = [json.loads(x) for x in (folder / '{}_seed{}.jsonl'.format(mode, seed)).read_text().splitlines()]
            data[mode].append((records[0], records[1:]))
    paired = []
    for (tm, _), (sm, _) in zip(data['teacher'], data['student']):
        if tm['initial_state_hashes'] != sm['initial_state_hashes'] or tm['specs_sha256'] != sm['specs_sha256']:
            raise RuntimeError('Pairing mismatch seed {}'.format(tm['seed']))
        paired.append(tm['seed'])
    result = {'paired_initial_states_verified_seeds': paired, 'purpose': 'diagnostic, not final independent test'}
    for mode, runs in data.items():
        rows = [r for _, rr in runs for r in rr]
        result[mode] = {
            'damaged': aggregate([r for r in rows if r['degradation_rate'] > 0]),
            'intact': aggregate([r for r in rows if r['degradation_rate'] == 0]),
            'by_severity': {str(d): aggregate([r for r in rows if r['degradation_rate'] == d]) for d in sorted({r['degradation_rate'] for r in rows})},
            'damaged_by_seed': {str(m['seed']): aggregate([r for r in rr if r['degradation_rate'] > 0]) for m, rr in runs},
            'by_joint_severity': [dict(joint_name=j, degradation_rate=d, **aggregate([r for r in rows if r['joint_name'] == j and r['degradation_rate'] == d])) for j in sorted({r['joint_name'] for r in rows}) for d in sorted({r['degradation_rate'] for r in rows})],
        }
    (folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[101, 102, 103])
    parser.add_argument('--replicates', type=int, default=48)
    args = parser.parse_args()
    folder = Path(args.output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {'seeds': args.seeds, 'replicates': args.replicates, 'commands': []}
    for seed in args.seeds:
        for mode, checkpoint in CHECKPOINTS.items():
            output = folder / '{}_seed{}.jsonl'.format(mode, seed)
            if output.exists():
                raise FileExistsError(output)
            command = [sys.executable, '-m', 'legged_gym.scripts.evaluate_onset_v2',
                       '--checkpoint-path', str(ROOT / checkpoint), '--policy-mode', mode,
                       '--output', str(output), '--replicates', str(args.replicates),
                       '--headless', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--seed', str(seed)]
            manifest['commands'].append(command)
            (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print('START {} seed {}'.format(mode, seed), flush=True)
            with output.with_suffix('.log').open('w') as log:
                subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)
            print('DONE {} seed {}'.format(mode, seed), flush=True)
    result = summarize(folder, args.seeds)
    print(json.dumps({mode: result[mode]['damaged'] for mode in CHECKPOINTS}, indent=2), flush=True)

if __name__ == '__main__':
    main()
