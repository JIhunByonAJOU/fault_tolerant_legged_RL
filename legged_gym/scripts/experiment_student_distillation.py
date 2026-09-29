"""Controlled offline student-encoder distillation on recorded TF/JT states.

This diagnoses imitation objectives and a moving teacher target. It is not a
replacement for closed-loop PPO training or independent trajectory testing.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import isaacgym  # noqa: F401 -- import before torch in the Isaac Gym environment
import torch

from legged_gym.learning.joint_teacher_student_actor_critic import (
    JointTeacherStudentActorCritic,
    StudentHistoryEncoder,
)
from legged_gym.learning.official_wim_teacher_actor_critic import (
    OfficialWimTeacherActorCritic,
)


ROOT = Path('logs/evaluations/jt-transition-audit-20260928')
TF = Path('logs/official_wim_teacher243_failure_fullrange_fromscratch/'
          'Aug13_03-44-07_seed1-50000iter-fromscratch-fullrange/model_43000.pt')
JT = Path('logs/jt_wim243_failure_fullrange_onset/Aug14_00-37-37_')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_data(device, extra_corpora=(), extra_cap=None, seed=0):
    parts = []
    for source in ('common_teacher_inputs.pt', 'common_student_inputs.pt'):
        item = torch.load(ROOT / source, map_location='cpu')
        item['source'] = torch.full((len(item['obs']),), len(parts), dtype=torch.long)
        parts.append(item)
    for path in extra_corpora:
        item = torch.load(path, map_location='cpu')
        if extra_cap is not None and len(item['obs']) > extra_cap:
            indices = torch.randperm(len(item['obs']), generator=torch.Generator().manual_seed(seed))[:extra_cap]
            item = {key: value[indices] for key, value in item.items()}
        item['source'] = torch.full((len(item['obs']),), len(parts), dtype=torch.long)
        parts.append(item)
    data = {key: torch.cat([p[key] for p in parts]).to(device)
            for key in ('obs', 'privileged', 'env_id', 'rate',
                        'elapsed_since_onset', 'source')}
    # The collector samples joint=(replicate % 12), so modulo-4 would withhold
    # only three joints. Hold out an entire 12-replicate block instead.
    replicate = data['env_id'] // 72
    data['validation'] = ((replicate // 12) % 4 == 3) & (data['source'] < 2)
    data['train'] = ~data['validation']
    data['history'] = data['obs'][:, 235:].reshape(-1, 50, 48)
    data['current'] = data['obs'][:, :235]
    data['post_fault'] = (data['rate'] > 0) & (data['elapsed_since_onset'] >= 1)
    return data


@torch.no_grad()
def outputs_from_encoder(encoder, history, batch_size=2048):
    return torch.cat([encoder(history[i:i + batch_size])
                      for i in range(0, len(history), batch_size)])


def evaluate(encoder, data, targets, actor, target_key):
    encoder.eval()
    target = targets[target_key]
    outputs = outputs_from_encoder(encoder, data['history'])
    with torch.no_grad():
        actions = actor(torch.cat([data['current'], outputs], dim=1))
        reference_actions = actor(torch.cat([data['current'], target], dim=1))
    result = {}
    for split in ('train', 'validation'):
        for group in ('all', 'post_fault', 'd1_post_fault'):
            mask = data[split]
            if group != 'all':
                mask = mask & data['post_fault']
            if group == 'd1_post_fault':
                mask = mask & (data['rate'] == 1)
            delta = outputs[mask] - target[mask]
            result[split + '_' + group] = {
                'n': int(mask.sum().item()),
                'latent_relative_rms': float(delta.square().mean().sqrt()
                    / target[mask].square().mean().sqrt().clamp(min=1e-8)),
                'latent_l2': float(delta.norm(dim=1).mean()),
                'action_mae': float((actions[mask] - reference_actions[mask]).abs().mean()),
                'joint_target_mae_rad': float((actions[mask] - reference_actions[mask]).abs().mean() * .25),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--conditions', nargs='+', choices=(
        'latent_tf', 'action_tf', 'latent_jt53000_fixed',
        'latent_jt53000_moving', 'normalized_latent_tf',
        'latent_action_tf', 'squared_latent_tf', 'latent_fault_tf'), default=None)
    parser.add_argument('--action-weight', type=float, default=10.,
                        help='Weight of action MSE in the combined TF objective')
    parser.add_argument('--fault-weight', type=float, default=0.25,
                        help='Weight of joint and severity classification on post-fault states')
    parser.add_argument('--d1-probability', type=float, default=None,
                        help='If set, sample this fraction from d=1 post-fault states')
    parser.add_argument('--history-alignment', choices=('original', 'recent48'),
                        default='original')
    parser.add_argument('--extra-corpus', action='append', default=[])
    parser.add_argument('--extra-cap', type=int)
    parser.add_argument('--init-student')
    parser.add_argument('--student-width', type=int, choices=(32, 64), default=32)
    args = parser.parse_args()
    if args.updates < 4 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('Invalid optimization settings')
    if args.d1_probability is not None and not 0 < args.d1_probability < 1:
        raise ValueError('d1 probability must be strictly between 0 and 1')
    if args.action_weight <= 0:
        raise ValueError('action weight must be positive')
    if args.fault_weight <= 0:
        raise ValueError('fault weight must be positive')
    if args.extra_cap is not None and args.extra_cap < 1:
        raise ValueError('extra cap must be positive')
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data = load_data(device, args.extra_corpus, args.extra_cap, args.seed)
    if args.history_alignment == 'recent48':
        # The first convolution consumes positions 0..47. Put the most recent
        # 48 frames there; the duplicated final two positions remain unused.
        data['history'] = torch.cat((data['history'][:, 2:],
                                    data['history'][:, -2:]), dim=1)
    tf = OfficialWimTeacherActorCritic(235, 45, 12).to(device).eval()
    tf.load_state_dict(torch.load(TF, map_location=device)['model_state_dict'])
    for parameter in tf.parameters():
        parameter.requires_grad_(False)
    targets = {}
    with torch.no_grad():
        targets['tf43000'] = tf.teacher_encoder(data['privileged']).detach()
    stage_iterations = (43000, 45000, 49000, 53000)
    for iteration in stage_iterations:
        model = JointTeacherStudentActorCritic(2635, 45, 12).to(device).eval()
        model.load_state_dict(torch.load(JT / ('model_%d.pt' % iteration), map_location=device)['model_state_dict'])
        with torch.no_grad():
            targets['jt%d' % iteration] = model.teacher_encoder(data['privileged']).detach()
        del model
    base = StudentHistoryEncoder(args.student_width).to(device)
    if args.init_student:
        source_state = torch.load(args.init_student, map_location=device)
        base.load_state_dict(source_state['student_encoder_state_dict'], strict=True)
    initial = copy.deepcopy(base.state_dict())
    actor = tf.actor
    train_indices = torch.where(data['train'])[0]
    teacher_dim_std = targets['tf43000'][train_indices].std(dim=0, unbiased=False).clamp(min=.1)
    severe_indices = torch.where(data['train'] & data['post_fault'] & (data['rate'] == 1))[0]
    other_indices = torch.where(data['train'] & ~(data['post_fault'] & (data['rate'] == 1)))[0]
    specs = {
        'latent_tf': ('latent', ['tf43000'] * args.updates, 'tf43000'),
        'action_tf': ('action', ['tf43000'] * args.updates, 'tf43000'),
        'latent_jt53000_fixed': ('latent', ['jt53000'] * args.updates, 'jt53000'),
        'latent_jt53000_moving': ('latent',
            ['jt%d' % stage_iterations[min(3, 4 * step // args.updates)]
             for step in range(args.updates)], 'jt53000'),
        'normalized_latent_tf': ('normalized_latent', ['tf43000'] * args.updates, 'tf43000'),
        'latent_action_tf': ('latent_action', ['tf43000'] * args.updates, 'tf43000'),
        'latent_fault_tf': ('latent_fault', ['tf43000'] * args.updates, 'tf43000'),
        'squared_latent_tf': ('squared_latent', ['tf43000'] * args.updates, 'tf43000'),
    }
    results = {}
    for name in (args.conditions or specs.keys()):
        objective, schedule, final_target = specs[name]
        encoder = StudentHistoryEncoder(args.student_width).to(device)
        encoder.load_state_dict(initial)
        joint_head = torch.nn.Linear(8, 12).to(device) if objective == 'latent_fault' else None
        severity_head = torch.nn.Linear(8, 5).to(device) if objective == 'latent_fault' else None
        parameters = list(encoder.parameters())
        if joint_head is not None:
            parameters += list(joint_head.parameters()) + list(severity_head.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
        generator = torch.Generator(device=device).manual_seed(args.seed + 101)
        checkpoints = {}
        initial_evaluation = evaluate(encoder, data, targets, actor, final_target)
        for step, target_name in enumerate(schedule, start=1):
            encoder.train()
            if args.d1_probability is None:
                picks = train_indices[torch.randint(
                    len(train_indices), (args.batch_size,), device=device, generator=generator)]
            else:
                n_severe = round(args.batch_size * args.d1_probability)
                picks = torch.cat((
                    severe_indices[torch.randint(len(severe_indices), (n_severe,),
                        device=device, generator=generator)],
                    other_indices[torch.randint(len(other_indices),
                        (args.batch_size - n_severe,), device=device, generator=generator)]))
            target = targets[target_name][picks]
            predicted = encoder(data['history'][picks])
            if objective == 'latent':
                # Exactly the norm used by the JT adaptation term.
                loss = torch.linalg.vector_norm(predicted - target, dim=1).mean()
            elif objective == 'latent_fault':
                loss = torch.linalg.vector_norm(predicted - target, dim=1).mean()
                post = data['post_fault'][picks]
                if post.any():
                    ids = data['env_id'][picks][post].long()
                    joint_label = (ids // 6) % 12
                    severity_label = (data['rate'][picks][post] * 5).round().long() - 1
                    fault_loss = torch.nn.functional.cross_entropy(joint_head(predicted[post]), joint_label)
                    fault_loss += torch.nn.functional.cross_entropy(severity_head(predicted[post]), severity_label)
                    loss = loss + args.fault_weight * fault_loss
            elif objective == 'normalized_latent':
                loss = (((predicted - target) / teacher_dim_std).square().mean())
            elif objective == 'squared_latent':
                loss = (predicted - target).square().sum(dim=1).mean()
            else:
                predicted_action = actor(torch.cat((data['current'][picks], predicted), dim=1))
                with torch.no_grad():
                    teacher_action = actor(torch.cat((data['current'][picks], target), dim=1))
                action_loss = (predicted_action - teacher_action).square().mean()
                if objective == 'latent_action':
                    loss = torch.linalg.vector_norm(predicted - target, dim=1).mean() + args.action_weight * action_loss
                else:
                    loss = action_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
            optimizer.step()
            if step in (args.updates // 4, args.updates // 2,
                        (3 * args.updates) // 4, args.updates):
                checkpoints[str(step)] = {
                    'target_during_update': target_name,
                    'last_training_loss': float(loss.item()),
                    'evaluation': evaluate(encoder, data, targets, actor, final_target),
                }
                print(name, step, checkpoints[str(step)]['evaluation']['validation_post_fault'], flush=True)
        torch.save({'student_encoder_state_dict': encoder.cpu().state_dict(),
                    'objective': objective, 'final_target': final_target,
                    'source_tf_sha256': digest(TF)}, output_dir / (name + '.pt'))
        results[name] = {'initial': initial_evaluation, 'checkpoints': checkpoints}
        (output_dir / 'results.json').write_text(json.dumps({
            'protocol': {'updates': args.updates, 'batch_size': args.batch_size,
                'learning_rate': args.lr, 'seed': args.seed,
                'd1_probability': args.d1_probability,
                'history_alignment': args.history_alignment,
                'action_weight': args.action_weight,
                'fault_weight': args.fault_weight,
                'student_width': args.student_width,
                'extra_corpora': args.extra_corpus,
                'extra_cap': args.extra_cap,
                'initial_student_checkpoint': args.init_student,
                'training_rows': int(data['train'].sum()),
                'validation_rows': int(data['validation'].sum()),
                'source_tf_sha256': digest(TF),
                'dataset': [str(ROOT / x) for x in
                    ('common_teacher_inputs.pt', 'common_student_inputs.pt')],
                'holdout': 'replicate_block_of_12 modulo 4 equals 3; all 12 joints and both trajectories held out',
                'actor': 'frozen TF43000; no PPO or new environment interaction'},
            'results': results}, indent=2))


if __name__ == '__main__':
    main()
