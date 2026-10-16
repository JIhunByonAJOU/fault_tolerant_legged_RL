#!/usr/bin/env python3
"""Generate paper-sized figures with one explicit, embedded text font."""
import argparse
import csv
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
from matplotlib.ticker import MaxNLocator

BASE = Path(__file__).resolve().parent
JOINTS = [f'{leg} {part}' for leg in ['FL', 'FR', 'RL', 'RR'] for part in ['hip', 'thigh', 'calf']]
RATES = [.2, .4, .6, .8, 1.]
METRICS = [('survived', 'Survival (%)'), ('command_vx_rmse', r'$v_x$ RMSE (m/s)'),
           ('command_vy_rmse', r'$v_y$ RMSE (m/s)'), ('command_wz_rmse', r'$\omega_z$ RMSE (rad/s)'),
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
    assert not any(ax.get_title(loc=loc) for ax in fig.axes for loc in ['left', 'center', 'right']), name
    assert fig._suptitle is None or not fig._suptitle.get_text(), name
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
                             'text_sizes_pt': sorted(sizes), 'outside_text': outside, 'internal_title': False}
    plt.close(fig)


def heatmap(rows, output, audit):
    groups = defaultdict(list)
    for r in rows:
        if r['policy'] == 'Student' and r['condition']['loss']:
            groups[r['condition']['joint'], r['condition']['loss']].append(r)
    values = np.zeros((12, 5))
    table = []
    for j in range(12):
        for k, d in enumerate(RATES):
            rs = groups[j, d]
            assert len(rs) == 216
            survived = sum(r['survived'] for r in rs)
            values[j, k] = 100 * survived / len(rs)
            table.append(dict(joint_index=j, joint=JOINTS[j], degradation=d, policy='Student',
                              episodes=len(rs), survived=survived, survival_percent=values[j, k]))
    with (output / 'paper_student_survival_heatmap.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    fig, ax = plt.subplots(figsize=(80 / 25.4, 100 / 25.4), constrained_layout=True)
    im = ax.imshow(values, cmap='Blues', norm=Normalize(0, 100), aspect='auto', interpolation='nearest')
    ax.set(xticks=range(5), xticklabels=[20, 40, 60, 80, 100], yticks=range(12), yticklabels=JOINTS,
           xlabel='Torque degradation (%)')
    annotations, contrasts = [], []
    for j in range(12):
        for k in range(5):
            color, contrast = text_color(im.cmap(im.norm(values[j, k])))
            annotations.append((j, k, ax.text(k, j, f'{values[j, k]:.1f}',
                                ha='center', va='center', color=color, fontsize=9)))
            contrasts.append(contrast)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
    ax.grid(which='minor', color='#eeeeee', linewidth=.25)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(axis='both', length=2, width=.5, pad=2)
    for spine in ax.spines.values(): spine.set_linewidth(.5)
    bar = fig.colorbar(im, ax=ax, label='Survival (%)', fraction=.045, pad=.035, aspect=40,
                 ticks=[0, 20, 40, 60, 80, 100], format='%.0f')
    bar.outline.set_linewidth(.5)
    bar.ax.tick_params(length=2, width=.5, pad=2)
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
                        'policy': 'Student', 'decimals': 1,
                        'minimum_text_background_contrast': min(contrasts),
                        'annotations_outside_cells': clipped, 'color_limits_percent': [0, 100],
                        'metric': '100 * nonfall trials / all trials; goal and nonfall administrative cap count as survival'}
    save(fig, 'paper_student_survival_heatmap', output, audit)


def curves(rows, output, audit):
    groups = defaultdict(list)
    for r in rows:
        groups[r['policy'], r['condition']['loss']].append(r)
    names = ['survival', 'vx_rmse', 'vy_rmse', 'yaw_rate_rmse', 'cross_track_rmse']
    all_rates = [0.] + RATES
    for i, (key, title) in enumerate(METRICS):
        fig, ax = plt.subplots(figsize=(80 / 25.4, 65 / 25.4), constrained_layout=True)
        for policy, style in POLICY_STYLE.items():
            middle, low, high = [], [], []
            factor = 100 if key == 'survived' else 1
            for d in all_rates:
                rs = groups[policy, d]
                middle.append(factor * np.mean([r[key] for r in rs]))
                seeds = [factor * np.mean([r[key] for r in rs if r['seed'] == s]) for s in [1101, 1102, 1103]]
                low.append(min(seeds)); high.append(max(seeds))
            audit.setdefault('curve_values', {}).setdefault(key, {})[policy] = {
                'degradation': all_rates, 'mean': middle, 'seed_min': low, 'seed_max': high}
            ax.plot(np.array(all_rates) * 100, middle, label=policy, markersize=3, linewidth=1.1, **style)
            ax.fill_between(np.array(all_rates) * 100, low, high, color=style['color'], alpha=.13, linewidth=0)
        ax.set(ylabel=title, xlabel='Torque degradation (%)',
               xticks=[0, 20, 40, 60, 80, 100], xlim=(-3, 103))
        ax.set_ylim(bottom=0)
        if key == 'survived':
            ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
        else:
            ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, ax.get_ylim()[1])
            ax.set_yticks(ticks); ax.set_ylim(0, ticks[-1])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=.2, linewidth=.4)
        ax.legend(loc='best', frameon=False, handlelength=1.7, borderaxespad=.25)
        save(fig, f'paper_{names[i]}_by_degradation', output, audit)
    audit['curve_bands'] = 'Minimum to maximum of the three evaluation-seed means; not a confidence interval.'


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
           xlabel='World x (m)', ylabel='Terrain column (world y increases upward)')
    for y in np.arange(0, 97, 8): ax.axhline(y, color='black', linewidth=.35, alpha=.4)
    for x in [8, 16, 24, 32]: ax.axvline(x, color='black', linewidth=.35, alpha=.4)
    fig.colorbar(im, ax=ax, label='Height (m)', fraction=.06,
                 ticks=np.linspace(float(height.min()), float(height.max()), 5), format='%.2f')
    save(fig, 'terrain_heightmap_s1101', output, audit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, help='Local font file; not copied into the dataset')
    parser.add_argument('--font-family', default='Times New Roman')
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
                except (RuntimeError, ValueError):
                    continue
            try:
                font_path = font_manager.findfont(font_manager.FontProperties(family=[args.font_family]), fallback_to_default=False)
            except ValueError as e:
                raise SystemExit('Required font unavailable. Install Times New Roman or pass --font PATH / --font-family NAME.') from e
        family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family': [family], 'font.size': 9, 'axes.titlesize': 9,
                         'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'legend.fontsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'mathtext.fontset': 'custom', 'mathtext.rm': family,
                         'mathtext.it': family + ':italic', 'mathtext.bf': family + ':bold',
                         'axes.unicode_minus': False, 'savefig.dpi': 300})
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    audit = {'ta_guidance_source': 'TA_GUIDANCE.md', 'ta_guidance_reviewed': True,
             'font_and_9pt_are_project_choices': True, 'font_family': family,
             'font_file': font_path, 'font_sha256': hashlib.sha256(Path(font_path).read_bytes()).hexdigest(),
             'font_size_pt': 9, 'figure_titles': 'External manuscript captions only',
             'graphics_reference': 'https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/improve-your-graphics/',
             'heatmap_layout': 'Single-column 80 x 100 mm; single-hue Blues; thin rules; fixed 0-100 percent',
             'figures': {}}
    rows = json.loads((BASE / 'records.json').read_text(encoding='utf-8'))
    heatmap(rows, output, audit); curves(rows, output, audit); terrain_overview(output, audit)
    (output / 'style_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
