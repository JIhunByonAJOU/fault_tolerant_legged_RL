#!/usr/bin/env python3
"""Draw compact publication panels from a frozen evaluation summary.

No checkpoint selection or raw-data inference is performed here. All ratios
are converted to percentages only for display; source CSV units are retained.
"""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    with args.summary.open() as f:
        rows = list(csv.DictReader(f))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8,
                         'legend.fontsize': 6.5, 'savefig.dpi': 240})
    series = [('tb22500', 'Normal Base (privileged)', '#4d4d4d', 's'),
              ('jt77500', 'Final Student (history)', '#007fa3', 'o')]
    panels = [('survival', 'S20', 100, '20-s survival (%)'),
              ('progress', 'world_progress_ratio', 100, 'World-forward progress / 10 m (%)'),
              ('body_progress', 'body_P_cmd', 100, 'Body-forward progress / 10 m (%)'),
              ('rmse', 'vx_alive_pooled_rmse', 1, 'Longitudinal RMSE (m/s)'),
              ('strict', 'Q_strict', 100, 'Strict tracking occupancy (%)'),
              ('stall', 'stall_time_s', 1, 'Alive stationary time (s)'),
              ('backward', 'backward_time_s', 1, 'Alive backward time (s)')]
    for name, key, scale, label in panels:
        fig, ax = plt.subplots(figsize=(3.45, 2.30), constrained_layout=True)
        for model, legend, color, marker in series:
            data = sorted((r for r in rows if r['model'] == model),
                          key=lambda r: float(r['degradation_rate']))
            if not data or any(not r.get(key) for r in data):
                raise ValueError(f'Missing {key} for {model}')
            ax.plot([float(r['degradation_rate']) for r in data],
                    [float(r[key]) * scale for r in data],
                    label=legend, color=color, marker=marker, ms=3, lw=1.4)
        ax.set(xlabel='Torque-multiplier loss d', ylabel=label,
               xticks=[0, .2, .4, .6, .8, 1])
        ax.grid(alpha=.25)
        ax.legend(frameon=False, loc='best')
        for ext in ['png', 'pdf']:
            fig.savefig(args.output / f'paper_{name}.{ext}')
        plt.close(fig)


if __name__ == '__main__':
    main()
