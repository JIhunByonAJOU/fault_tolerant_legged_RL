"""Offline test of whether history can identify the failed joint and severity.

This is a diagnostic classifier on recorded trajectories, not a locomotion
policy or a deployed fault detector.
"""

import argparse
import json
from pathlib import Path

import isaacgym  # noqa: F401
import torch
import torch.nn as nn

from legged_gym.learning.joint_teacher_student_actor_critic import StudentHistoryEncoder


ROOT = Path('logs/evaluations/jt-transition-audit-20260928')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--updates', type=int, default=5000)
    parser.add_argument('--student-width', type=int, choices=(32, 64), default=32)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.manual_seed(11)
    tensors = {name: [] for name in ('history', 'joint', 'severity', 'replicate', 'elapsed')}
    for filename in ('common_teacher_inputs.pt', 'common_student_inputs.pt'):
        data = torch.load(ROOT / filename, map_location='cpu')
        mask = (data['rate'] > 0) & (data['elapsed_since_onset'] >= 1)
        ids = data['env_id'][mask].long()
        tensors['history'].append(data['obs'][mask, 235:].reshape(-1, 50, 48))
        tensors['joint'].append((ids // 6) % 12)
        tensors['severity'].append((data['rate'][mask] * 5).round().long() - 1)
        tensors['replicate'].append(ids // 72)
        tensors['elapsed'].append(data['elapsed_since_onset'][mask])
    data = {name: torch.cat(parts) for name, parts in tensors.items()}
    train = (data['replicate'] // 12) % 4 != 3
    validation = ~train
    encoder = StudentHistoryEncoder(args.student_width)
    head_joint = nn.Linear(8, 12)
    head_severity = nn.Linear(8, 5)
    model = nn.ModuleDict(dict(encoder=encoder, joint=head_joint, severity=head_severity))
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    train_ids = train.nonzero().flatten()
    rng = torch.Generator().manual_seed(112)
    for step in range(args.updates):
        picks = train_ids[torch.randint(len(train_ids), (512,), generator=rng)]
        z = encoder(data['history'][picks])
        loss = nn.functional.cross_entropy(head_joint(z), data['joint'][picks])
        loss += nn.functional.cross_entropy(head_severity(z), data['severity'][picks])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.inference_mode():
        z = encoder(data['history'])
        predicted_joint = head_joint(z).argmax(dim=1)
        predicted_severity = head_severity(z).argmax(dim=1)
        groups = {
            'train': train,
            'validation': validation,
            'validation_d1': validation & (data['severity'] == 4),
            'validation_late5s': validation & (data['elapsed'] >= 5),
        }
        results = {}
        for name, mask in groups.items():
            results[name] = {
                'n': int(mask.sum()),
                'joint_accuracy': float((predicted_joint[mask] == data['joint'][mask]).float().mean()),
                'severity_accuracy': float((predicted_severity[mask] == data['severity'][mask]).float().mean()),
                'joint_and_severity_accuracy': float(((predicted_joint[mask] == data['joint'][mask]) & (predicted_severity[mask] == data['severity'][mask])).float().mean()),
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'updates': args.updates, 'student_width': args.student_width,
                                  'holdout': 'replicate block of 12 modulo 4 equals 3; all joints present',
                                  'results': results}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
