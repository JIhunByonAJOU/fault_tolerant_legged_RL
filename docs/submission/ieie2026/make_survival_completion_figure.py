"""Render the paired survival/completion panel from the frozen evaluation records."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text

BASE = Path(__file__).resolve().parent
DATA = BASE.parents[2] / 'datasets/ieie2026_fixed2112'
OUT = BASE / 'paper_assets'
RATES = [0., .2, .4, .6, .8, 1.]
STYLES = {'Normal': dict(color='#b35a00', linestyle='--', marker='o'),
          'Student': dict(color='#1466a2', linestyle='-', marker='s')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path)
    args = parser.parse_args()
    if args.font:
        font_manager.fontManager.addfont(str(args.font))
        font = str(args.font.resolve())
    else:
        for candidate in (Path.home() / '.local/share/fonts').rglob('*.ttf'):
            try:
                if font_manager.FontProperties(fname=candidate).get_name() == 'Times New Roman':
                    font_manager.fontManager.addfont(str(candidate))
            except (RuntimeError, ValueError):
                continue
        font = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),
                                     fallback_to_default=False)
    assert font_manager.FontProperties(fname=font).get_name() == 'Times New Roman'
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9,
                         'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'legend.fontsize': 9, 'pdf.fonttype': 42, 'axes.unicode_minus': False})
    records_path = DATA / 'records.json'
    records = json.loads(records_path.read_text(encoding='utf-8'))
    assert len(records) == 26352
    groups = defaultdict(list)
    for record in records:
        groups[record['policy'], record['condition']['loss']].append(record)
    table = []
    for policy in STYLES:
        for loss in RATES:
            rows = groups[policy, loss]
            assert len(rows) == (216 if loss == 0 else 2592)
            surviving = sum(r['survived'] for r in rows)
            completed = sum(r['termination'] == 'goal' for r in rows)
            assert all(r['survived'] for r in rows if r['termination'] == 'goal')
            seed_groups = [[r for r in rows if r['seed'] == seed] for seed in [1101, 1102, 1103]]
            survival_seeds = [100 * sum(r['survived'] for r in group) / len(group) for group in seed_groups]
            completion_seeds = [100 * sum(r['termination'] == 'goal' for r in group) / len(group)
                                for group in seed_groups]
            table.append(dict(policy=policy, loss=loss, episodes=len(rows),
                              survived=surviving, completed=completed,
                              survival_percent=100 * surviving / len(rows),
                              completion_percent=100 * completed / len(rows),
                              survival_seed_min=min(survival_seeds), survival_seed_max=max(survival_seeds),
                              completion_seed_min=min(completion_seeds), completion_seed_max=max(completion_seeds)))
    # Use common scales and one legend; no panel headings or new uncertainty text.
    fig, axes = plt.subplots(1, 2, figsize=(80 / 25.4, 44 / 25.4))
    fig.subplots_adjust(left=.16, right=.95, top=.79, bottom=.29, wspace=.95)
    for ax, key, label in zip(axes, ['survival_percent', 'completion_percent'],
                              ['Survival (%)', 'Completion (%)']):
        for policy, style in STYLES.items():
            rows = [r for r in table if r['policy'] == policy]
            ax.plot([100 * r['loss'] for r in rows], [r[key] for r in rows],
                    label=policy, linewidth=1, markersize=2.6, **style)
            prefix = key.replace('_percent', '')
            ax.fill_between([100 * r['loss'] for r in rows], [r[prefix + '_seed_min'] for r in rows],
                            [r[prefix + '_seed_max'] for r in rows], color=style['color'],
                            alpha=.13, linewidth=0)
        ax.set(xlim=(-3, 103), ylim=(0, 105), xticks=[0, 50, 100], yticks=[0, 50, 100],
               ylabel=label)
        ax.yaxis.labelpad = 1
        ax.tick_params(length=2, width=.6, pad=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=.2, linewidth=.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55, 1.01), ncol=2,
               frameon=False, handlelength=1.4, columnspacing=.8, handletextpad=.35)
    fig.supxlabel('Torque degradation (%)', y=.025, fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside, sizes = [], set()
    text_boxes = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        sizes.add(float(text.get_fontsize()))
        text_boxes.append((text.get_text(), box))
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
            outside.append(text.get_text())
    assert sizes == {9.} and not outside, (sizes, outside)
    overlaps = [(a, b) for i, (a, box_a) in enumerate(text_boxes)
                for b, box_b in text_boxes[i + 1:] if box_a.overlaps(box_b)]
    assert not overlaps, overlaps
    assert not any(ax.get_title() for ax in axes) and fig._suptitle is None
    OUT.mkdir(exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(OUT / f'paper_survival_completion.{extension}', dpi=300)
    plt.close(fig)
    with (OUT / 'survival_completion.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    audit = {'records_sha256': hashlib.sha256(records_path.read_bytes()).hexdigest(),
             'font_family': 'Times New Roman', 'font_size_pt': 9,
             'font_sha256': hashlib.sha256(Path(font).read_bytes()).hexdigest(),
             'width_mm': 80, 'height_mm': 44, 'internal_title': False,
             'outside_text': outside, 'text_overlaps': overlaps,
             'shared_legend': True, 'common_axes': True,
             'panels': ['survival_percent', 'completion_percent'],
             'denominator': 'All trials at each policy and degradation, including falls',
             'completion_definition': 'termination == goal', 'values': table}
    (OUT / 'survival_completion_audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Generated {OUT / "paper_survival_completion.pdf"}: 12 conditions, 24 rates')


if __name__ == '__main__':
    main()
