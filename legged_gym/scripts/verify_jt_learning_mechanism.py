"""Independent numerical checks and a figure for the read-only JT audit."""
import json
from pathlib import Path

import torch
from torch.distributions import Normal, kl_divergence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_jt_learning_mechanism import kl


def main():
    torch.set_num_threads(1)
    torch.manual_seed(29)
    root = Path(__file__).resolve().parents[2]
    folder = root / 'logs/evaluations/jt_learning_mechanism_20260929'
    data = json.loads((folder / 'results.json').read_text())
    a, b = torch.randn(100, 12, dtype=torch.float64), torch.randn(100, 12, dtype=torch.float64)
    s, t = torch.rand(12, dtype=torch.float64) + .1, torch.rand(12, dtype=torch.float64) + .1
    reference = kl_divergence(Normal(a, s), Normal(b, t)).sum(-1)
    error = (reference - kl(a, s, b, t)).abs().max().item()
    assert error < 1e-10
    checks = {'gaussian_kl_max_difference_vs_torch_distribution': error,
              'cuda_initialized': torch.cuda.is_initialized(), 'corpora': {}}
    group = 'post_fault_1s_plus'
    for name, corpus in data['corpora'].items():
        checks['corpora'][name] = {}
        for label, startup in corpus['startup'].items():
            zero = startup['stored_update_interpolation']['0.0'][group]
            full = startup['stored_update_interpolation']['1.0'][group]
            first = startup['first_saved_update'][group]
            assert zero['action_mae'] < 1e-6 and zero['kl_mean'] < 1e-6
            assert abs(full['action_mae'] - first['action_mae']) < 1e-5
            assert abs(full['kl_mean'] - first['kl_mean']) < 1e-3
            assert abs(first['joint_target_mae_rad'] - .25 * first['action_mae']) < 1e-10
            assert corpus['checkpoints'][label+'/43000']['alpha'] == 0
            checks['corpora'][name][label] = 'zero/full interpolation, units, initial alpha: pass'
    assert not torch.cuda.is_initialized()
    (folder / 'verification.json').write_text(json.dumps(checks, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label, color in [('original32', '#374151'), ('fresh64', '#137c8b')]:
        rows = data['training'][label]['first_20']
        axes[0].plot([r['iteration'] - 43000 for r in rows], [r['PPO/mean_kl'] for r in rows], label=label, color=color)
        corpus = data['corpora']['common_teacher_inputs.pt']
        interp = corpus['startup'][label]['stored_update_interpolation']
        scales = [float(x) for x in interp if float(x) > 0]
        axes[1].plot(scales, [interp[str(x)][group]['kl_mean'] for x in scales], marker='.', color=color)
        rows = sorted((int(k.split('/')[1]), v) for k,v in corpus['checkpoints'].items() if k.startswith(label+'/') and 49000 <= int(k.split('/')[1]) <= 52500)
        axes[2].plot([i for i,v in rows], [v['student_vs_fused'][group]['kl_mean'] for i,v in rows], marker='.', color=color)
    axes[0].set(title='Recorded startup policy KL', xlabel='JT iterations after TF43000', ylabel='Mean KL (log scale)', yscale='log')
    axes[0].legend()
    axes[1].set(title='Stored update interpolation (not retraining)', xlabel='Fraction of first stored parameter change', ylabel='KL vs TF43000', xscale='log', yscale='log')
    axes[2].set(title='Pure student vs fused policy', xlabel='Checkpoint iteration', ylabel='Same-input mean KL', yscale='log')
    for ax in axes:
        ax.axhline(.02, color='#888888', linestyle=':', linewidth=1)
        ax.grid(alpha=.15)
    fig.suptitle('Logs and CPU inference only; no new closed-loop performance measurement', fontsize=11)
    fig.tight_layout()
    output = root / 'docs/results/jt_learning_mechanism_20260929'
    output.mkdir(exist_ok=True)
    fig.savefig(output / 'mechanism_audit.png', dpi=180)
    fig.savefig(output / 'mechanism_audit.pdf')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
