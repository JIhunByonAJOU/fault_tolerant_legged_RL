#!/usr/bin/env python3
"""Generate paper-sized figures with one explicit, embedded text font."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.text import Text

BASE = Path(__file__).resolve().parent
JOINTS = [f'{leg} {part}' for leg in ['FL', 'FR', 'RL', 'RR'] for part in ['hip', 'thigh', 'calf']]
RATES = [.2, .4, .6, .8, 1.]
METRICS = [('survived', 'Survival (%)'), ('command_vx_rmse', 'vx RMSE (m/s)'),
           ('command_vy_rmse', 'vy RMSE (m/s)'), ('command_wz_rmse', 'Yaw-rate RMSE (rad/s)'),
           ('cross_track_rmse', 'Cross-track RMSE (m)')]
POLICY_STYLE = {'Normal': dict(color='#b35a00', linestyle='--', marker='o'),
                'Student': dict(color='#1466a2', linestyle='-', marker='s')}


def text_color(rgb):
    rgb = np.asarray(rgb[:3])
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = float(linear @ np.array([.2126, .7152, .0722]))
    black, white = (luminance + .05) / .05, 1.05 / (luminance + .05)
    return ('black', black) if black >= white else ('white', white)


def save(fig, name, output, audit):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    sizes = set()
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():
            continue
        sizes.add(float(t.get_fontsize()))
        b = t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width + 1 or b.y1 > fig.bbox.height + 1:
            outside.append(t.get_text())
    assert not outside, (name, 'Text outside figure', outside)
    for ext in ['pdf', 'png']:
        fig.savefig(output / f'{name}.{ext}', dpi=300)
    audit['figures'][name] = {'width_mm': float(fig.get_figwidth() * 25.4),
                             'height_mm': float(fig.get_figheight() * 25.4),
                             'text_sizes_pt': sorted(sizes), 'outside_text': outside}
    plt.close(fig)


def heatmap(rows, output, audit):
    groups = defaultdict(list)
    for r in rows:
        if r['condition']['loss']:
            groups[r['condition']['joint'], r['condition']['loss'], r['policy']].append(r)
    values = np.zeros((12, 5))
    for j in range(12):
        for k, d in enumerate(RATES):
            means = {}
            for policy in POLICY_STYLE:
                rs = groups[j, d, policy]
                assert len(rs) == 216
                assert all(r['valid_post_samples'] > 0 for r in rs)
                means[policy] = np.mean([np.sqrt(sum(r['error_sum_sq'][:2]) / r['valid_post_samples']) for r in rs])
            values[j, k] = means['Normal'] - means['Student']
    bound = max(float(np.abs(values).max()), 1e-9)
    fig, ax = plt.subplots(figsize=(170 / 25.4, 95 / 25.4), constrained_layout=True)
    im = ax.imshow(values, cmap='RdBu', norm=Normalize(-bound, bound), aspect='auto')
    ax.set(xticks=range(5), xticklabels=[20, 40, 60, 80, 100], yticks=range(12), yticklabels=JOINTS,
           xlabel='Torque degradation (%)', title='Planar velocity-command RMSE: Normal - Student')
    annotations, contrasts = [], []
    for j in range(12):
        for k in range(5):
            color, contrast = text_color(im.cmap(im.norm(values[j, k])))
            annotations.append((j, k, ax.text(k, j, f'{values[j, k]:+.3f}',
                                ha='center', va='center', color=color, fontsize=9)))
            contrasts.append(contrast)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.35)
    ax.tick_params(which='minor', bottom=False, left=False)
    fig.colorbar(im, ax=ax, label='RMSE reduction (m/s)', fraction=.07, pad=.025,
                 ticks=np.linspace(-bound, bound, 5), format='%.2f')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for j, k, t in annotations:
        b = t.get_window_extent(renderer)
        p = ax.transData.transform([[k - .5, j - .5], [k + .5, j + .5]])
        lo, hi = p.min(axis=0), p.max(axis=0)
        if b.x0 < lo[0] + 1 or b.x1 > hi[0] - 1 or b.y0 < lo[1] + 1 or b.y1 > hi[1] - 1:
            clipped.append([j, k])
    assert not clipped, clipped
    audit['heatmap'] = {'cells': len(annotations), 'trials_per_policy_cell': 216,
                        'decimals': 3, 'minimum_text_background_contrast': min(contrasts),
                        'annotations_outside_cells': clipped, 'symmetric_color_limit_mps': bound,
                        'metric': 'mean_i sqrt((SSE_vx_i+SSE_vy_i)/N_i), Normal minus Student'}
    save(fig, 'paper_joint_vxy_gain', output, audit)


def curves(rows, output, audit):
    groups = defaultdict(list)
    for r in rows:
        groups[r['policy'], r['condition']['loss']].append(r)
    fig, axes = plt.subplots(2, 3, figsize=(170 / 25.4, 112 / 25.4), constrained_layout=True)
    all_rates = [0.] + RATES
    for i, (key, title) in enumerate(METRICS):
        ax = axes.flat[i]
        for policy, style in POLICY_STYLE.items():
            middle, low, high = [], [], []
            factor = 100 if key == 'survived' else 1
            for d in all_rates:
                rs = groups[policy, d]
                middle.append(factor * np.mean([r[key] for r in rs]))
                seeds = [factor * np.mean([r[key] for r in rs if r['seed'] == s]) for s in [1101, 1102, 1103]]
                low.append(min(seeds)); high.append(max(seeds))
            ax.plot(np.array(all_rates) * 100, middle, label=policy, markersize=3, linewidth=1.1, **style)
            ax.fill_between(np.array(all_rates) * 100, low, high, color=style['color'], alpha=.13, linewidth=0)
        ax.set(title=f'({chr(97 + i)}) {title}', xlabel='Torque degradation (%)',
               xticks=[0, 20, 40, 60, 80, 100], xlim=(-3, 103))
        ax.set_ylim(bottom=0)
        if key == 'survived': ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=.2, linewidth=.4)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].axis('off')
    axes.flat[5].legend(handles, labels, loc='center', frameon=False)
    audit['curve_bands'] = 'Minimum to maximum of the three evaluation-seed means; not a confidence interval.'
    save(fig, 'paper_metrics_by_degradation', output, audit)


def terrain_overview(output, audit):
    p = BASE / 'terrain/heightfield_s1101.npz'
    if not p.exists(): return
    with np.load(p, allow_pickle=False) as z:
        height = z['height_field_raw'][250:650, 250:1210] * float(z['vertical_scale'])
    fig, ax = plt.subplots(figsize=(170 / 25.4, 125 / 25.4), constrained_layout=True)
    im = ax.imshow(height.T, origin='lower', extent=[0, 40, 0, 96], aspect='auto', cmap='terrain')
    names = ['01  -Slope', '02  +Slope', '03  Rough'] + [f'{c:02d}  -Stairs' for c in range(4, 8)]
    names += [f'{c:02d}  +Stairs' for c in range(8, 11)] + ['11  Obstacles', '12  Obstacles']
    ax.set(yticks=np.arange(4, 96, 8), yticklabels=names, xticks=[0, 8, 16, 24, 32, 40],
           xlabel='World x (m)', ylabel='Terrain column (world y increases upward)',
           title='Verified height map, seed 1101 (axis scales differ)')
    for y in np.arange(0, 97, 8): ax.axhline(y, color='black', linewidth=.35, alpha=.4)
    for x in [8, 16, 24, 32]: ax.axvline(x, color='black', linewidth=.35, alpha=.4)
    fig.colorbar(im, ax=ax, label='Height (m)', fraction=.06,
                 ticks=np.linspace(float(height.min()), float(height.max()), 5), format='%.2f')
    save(fig, 'terrain_heightmap_s1101', output, audit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, help='Local font file; not copied into the dataset')
    parser.add_argument('--font-family', default='HYSinMyeongJo-Medium')
    parser.add_argument('--output', type=Path, default=BASE / 'generated/paper_figures')
    args = parser.parse_args(argv)
    if args.font:
        font_manager.fontManager.addfont(str(args.font))
        font_path = str(args.font.resolve())
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        try:
            font_path = font_manager.findfont(font_manager.FontProperties(family=[args.font_family]), fallback_to_default=False)
        except ValueError:
            # Fonts can be installed after Matplotlib built its cache. Register
            # an exact family match; never substitute an unrelated fallback.
            candidates = font_manager.findSystemFonts()
            user_fonts = Path.home() / '.local/share/fonts'
            if user_fonts.exists():
                candidates += font_manager.findSystemFonts(fontpaths=[str(user_fonts)])
            for candidate in candidates:
                try:
                    if font_manager.FontProperties(fname=candidate).get_name() == args.font_family:
                        font_manager.fontManager.addfont(candidate)
                        break
                except (RuntimeError, ValueError):
                    continue
            try:
                font_path = font_manager.findfont(font_manager.FontProperties(family=[args.font_family]), fallback_to_default=False)
            except ValueError as e:
                raise SystemExit('Required font unavailable. Install HY Shin Myeongjo or pass --font PATH / --font-family NAME.') from e
        family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family': [family], 'font.size': 9, 'axes.titlesize': 9,
                         'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'legend.fontsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.unicode_minus': False, 'savefig.dpi': 300})
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    audit = {'ta_guidance_source': 'TA_GUIDANCE.md', 'ta_guidance_reviewed': True,
             'font_and_9pt_are_project_choices': True, 'font_family': family,
             'font_file': font_path, 'font_sha256': hashlib.sha256(Path(font_path).read_bytes()).hexdigest(),
             'font_size_pt': 9, 'figures': {}}
    rows = json.loads((BASE / 'records.json').read_text(encoding='utf-8'))
    heatmap(rows, output, audit); curves(rows, output, audit); terrain_overview(output, audit)
    (output / 'style_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
