"""Plot held-out fixed-TF student imitation over supervised updates."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path('logs/evaluations')
OUT = Path('docs/results/jt_student_encoder_experiments_20260928.png')


def rows(folder):
    data = json.loads((ROOT / folder / 'results.json').read_text())
    run = data['results']['latent_tf']
    return [(int(step), entry['evaluation']) for step, entry in run['checkpoints'].items()]


def main():
    quick = rows('student-distillation-20260928')
    long = rows('student-distillation-long10k-20260928')
    severe = rows('student-distillation-long10k-severe50-20260928')
    points = sorted({step: result for step, result in quick + long}.items())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=170)
    axes[0].plot([s for s, _ in points],
                 [e['validation_post_fault']['latent_relative_rms'] for _, e in points],
                 marker='o', label='all damaged')
    axes[0].plot([s for s, _ in points],
                 [e['validation_d1_post_fault']['latent_relative_rms'] for _, e in points],
                 marker='s', label='d=1.0')
    axes[0].set(xlabel='Supervised encoder updates', ylabel='Held-out relative latent RMS error',
                title='Frozen TF43000 target')
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.2)
    for label, source, marker in [('natural sampling', long, 'o'), ('d=1 sampled 50%', severe, 's')]:
        axes[1].plot([s for s, _ in source],
                     [e['validation_post_fault']['action_mae'] for _, e in source],
                     marker=marker, label=label)
    axes[1].set(xlabel='Supervised encoder updates', ylabel='Held-out action MAE (normalized action)',
                title='Frozen TF43000 actor')
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


if __name__ == '__main__':
    main()
