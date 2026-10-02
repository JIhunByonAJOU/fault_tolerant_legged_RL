"""Run a fixed, audited inference protocol beside authorized MORAI workloads.

Existing outputs are reused only when their source, weights and complete protocol
match. This runner never stops another process or starts training.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/submission/ieie2026/results/independent_small_protocol.json'
PYTHON = '/home/jihun/Capstone2/miniconda3/envs/WIM/bin/python'
BASE = ROOT / 'logs/manual/teacher243-seed1-500iter/model_22500.pt'
FINAL = ROOT / 'logs/jt_wim243_width64_lr1e5_fresh_20260929/run_seed1_35000/model_77500.pt'
MODES = {'tb22500': 'teacher', 'jt77500': 'student',
         'oracle77500': 'oracle', 'history_repeat77500': 'history_repeat'}
FIXED = {'training': False, 'command': [.5, 0, 0], 'dt_s': .02,
         'post_fault_s': 20, 'onset_range_s': [2, 10], 'terrain': 'mixed',
         'noise': False, 'pushes': False, 'joint_indices': list(range(12)),
         'rates': [0, .2, .4, .6, .8, 1],
         'shared_gpu_limit': {'sustained_util_pct': 90, 'consecutive_checks': 6,
                              'check_every_steps': 50, 'min_free_mib': 2048}}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol):
    """CPU-only validation of all fixed evaluator/resource assumptions."""
    for key, expected in FIXED.items():
        if key not in protocol or protocol[key] != expected:
            raise ValueError(f'Unsupported protocol {key}: expected {expected!r}')
    if any(protocol[key] is not False for key in ('training','noise','pushes')):
        raise ValueError('Training, noise and pushes must be explicit false booleans')
    if protocol.get('pre_seconds', 2) != 2:
        raise ValueError('Evaluator pre-window is fixed to 2 seconds')
    if protocol.get('start_max_gpu_util_pct', 80) != 80 or protocol.get('start_min_free_mib', 4096) != 4096:
        raise ValueError('Start guard is fixed to <=80% utilization and >=4096 MiB free')
    seeds = protocol.get('seeds')
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    repeats = protocol.get('replicates_per_seed')
    if type(repeats) is not int or repeats < 1:
        raise ValueError('Replicates must be a positive integer')
    if protocol.get('environments_per_run') != 72 * repeats:
        raise ValueError('Environment count must be 12 joints * 6 rates * repeats')
    if protocol.get('episodes_per_cell') != repeats * len(seeds):
        raise ValueError('Cell episode count must be repeats * seed count')
    models = protocol.get('models')
    if not isinstance(models, list) or not models or len(set(models)) != len(models) or any(m not in MODES for m in models):
        raise ValueError('Unsupported or duplicate model identifiers')
    pacing = protocol.get('step_sleep_ms')
    if type(pacing) not in (int, float) or not 30 <= pacing <= 1000:
        raise ValueError('Shared GPU pacing must be between 30 and 1000 milliseconds')
    hashes = protocol.get('checkpoint_sha256', {})
    for key in ('normal_base', 'final_student'):
        value = hashes.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Missing/invalid frozen checkpoint hash: {key}')


def expected_specs(seed, repeats, rates):
    # Load this simulator-independent module without importing the environment.
    path = ROOT / 'legged_gym/evaluation/onset_protocol_v2.py'
    spec = importlib.util.spec_from_file_location('paper_onset_specs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_specs(seed, repeats, [float(rate) for rate in rates], .02)


def validate_existing(records, protocol, seed, model, evaluator_sha, checkpoint_sha):
    """Reject stale, incomplete or mismatched output before any GPU query."""
    if not records or len(records) != 1 + protocol['environments_per_run']:
        raise ValueError('Incomplete existing evaluation file')
    metadata = records[0]
    expected = {'record_type': 'metadata', 'schema_version': 5,
                'evaluation': 'paper_paired_random_onset_mobility_v2',
                'evaluator_sha256': evaluator_sha,
                'checkpoint_sha256': checkpoint_sha,
                'protocol_sha256': sha256_file(ROOT / 'legged_gym/evaluation/onset_protocol_v2.py'),
                'policy_mode': MODES[model], 'seed': seed,
                'num_envs': protocol['environments_per_run'], 'dt': .02,
                'command': [.5, 0, 0], 'post_seconds': 20,
                'pre_seconds': 2, 'onset_range_seconds': [2, 10],
                'terrain': 'mixed', 'rates': protocol['rates'],
                'replicates': protocol['replicates_per_seed'],
                'noise': False, 'pushes': False, 'initialization_neutral_steps': 1,
                'step_sleep_ms': protocol['step_sleep_ms'],
                'scientific_result': True, 'runtime_gpu_pipeline': True,
                'runtime_physx_gpu': True,
                'strict_thresholds': [.05, .05, .1]}
    for key, value in expected.items():
        if key not in metadata or metadata[key] != value:
            raise ValueError(f'Existing result does not match {key}: expected {value!r}')
    # Require booleans rather than accidentally accepting integer 1.
    if any(metadata[key] is not True for key in ('scientific_result', 'runtime_gpu_pipeline', 'runtime_physx_gpu')):
        raise ValueError('Existing result lacks actual scientific GPU flags')
    specs = expected_specs(seed, protocol['replicates_per_seed'], protocol['rates'])
    if metadata.get('specs_sha256') != hashlib.sha256(json.dumps(specs).encode()).hexdigest():
        raise ValueError('Existing result does not match generated onset specifications')
    robot_rows = records[1:]
    by_id = {r.get('env_id'): r for r in robot_rows}
    if len(by_id) != len(specs) or set(by_id) != set(range(len(specs))):
        raise ValueError('Duplicate or missing environment rows')
    for env_id, (joint, rate, onset, replicate) in enumerate(specs):
        row = by_id[env_id]
        fields = {'record_type': 'robot', 'seed': seed, 'joint_index': joint,
                  'degradation_rate': rate, 'replicate': replicate,
                  'post_horizon_seconds': 20, 'onset_seconds': onset * .02}
        if any(row.get(key) != value for key, value in fields.items()):
            raise ValueError(f'Existing result episode {env_id} does not match its specification')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--python', default=PYTHON if Path(PYTHON).exists() else sys.executable)
    parser.add_argument('--base-checkpoint', type=Path, default=BASE)
    parser.add_argument('--final-checkpoint', type=Path, default=FINAL)
    parser.add_argument('--isaacgym-python', type=Path,
                        default=Path(os.environ.get('ISAAC_GYM_PYTHON', '/home/jihun/Capstone2/isaacgym/python')))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--summary-output-dir', type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    evaluator = ROOT / 'legged_gym/scripts/evaluate_paper_mobility.py'
    evaluator_sha = sha256_file(evaluator)
    checkpoints = {'normal_base': args.base_checkpoint, 'final_student': args.final_checkpoint}
    actual_hashes = {name: sha256_file(path) for name, path in checkpoints.items()}
    if actual_hashes != protocol['checkpoint_sha256']:
        raise ValueError('Checkpoint files do not match frozen protocol hashes')
    out = args.output_dir or ROOT / protocol.get('output_dir', 'logs/evaluations/ieie2026_independent_small_v1')
    summary_out = args.summary_output_dir or ROOT / protocol.get('summary_output_dir', 'docs/submission/ieie2026/results/independent_small')
    env = dict(os.environ, PATH=str(Path(args.python).parent) + ':' + os.environ['PATH'],
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONNOUSERSITE='1', PYTHONPATH=str(ROOT) + ':' + str(args.isaacgym_python))
    out.mkdir(parents=True, exist_ok=True)
    for seed in protocol['seeds']:
        for model in protocol['models']:
            output = out / f'{model}_seed{seed}.jsonl'
            checkpoint_key = 'normal_base' if model == 'tb22500' else 'final_student'
            checkpoint = checkpoints[checkpoint_key]
            if output.exists():
                records = [json.loads(line) for line in output.read_text().splitlines()]
                validate_existing(records, protocol, seed, model, evaluator_sha, actual_hashes[checkpoint_key])
                print(f'Validated complete: {output}', flush=True)
                continue
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=process_name', '--format=csv,noheader'], text=True).splitlines()
            unknown = [p for p in processes if p and p != '/usr/bin/sunshine' and not p.endswith('/Simulator.x86_64')]
            if unknown:
                raise RuntimeError(f'Other compute workload appeared; no evaluation started: {unknown}')
            baseline = []
            for _ in range(3):
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.free', '--format=csv,noheader,nounits'], text=True)
                util, free = [int(v.strip()) for v in raw.strip().split(',')]
                baseline.append(util)
                if util > 80 or free < 4096:
                    raise RuntimeError('Insufficient GPU margin before run')
                time.sleep(1)
            command = [args.python, str(evaluator), '--checkpoint-path', str(checkpoint),
                       '--policy-mode', MODES[model], '--output', str(output),
                       '--replicates', str(protocol['replicates_per_seed']),
                       '--rates', *[str(r) for r in protocol['rates']], '--pre-seconds', '2',
                       '--post-seconds', '20', '--command-x', '.5', '--terrain', 'mixed',
                       '--headless', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0',
                       '--seed', str(seed), '--step-sleep-ms', str(protocol['step_sleep_ms']),
                       '--shared-gpu-monitor', '--num_threads', '1']
            print(f'Running {model} seed{seed}, baseline={baseline}, {protocol["environments_per_run"]} envs', flush=True)
            with output.with_suffix('.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, preexec_fn=lambda: os.nice(10))
            if result.returncode:
                raise RuntimeError(f'Evaluation stopped; see {output.with_suffix(".log")}')
            # Check newly generated output too, before reporting success.
            validate_existing([json.loads(line) for line in output.read_text().splitlines()],
                              protocol, seed, model, evaluator_sha, actual_hashes[checkpoint_key])
    subprocess.run([sys.executable, str(ROOT / 'legged_gym/scripts/summarize_paper_evaluation.py'),
                    '--raw-dir', str(out), '--output-dir', str(summary_out),
                    '--seeds', *[str(s) for s in protocol['seeds']], '--models', *protocol['models']], check=True)


if __name__ == '__main__':
    main()
