"""Frozen eight-motion campaign specifications, independent of the simulator."""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ['smooth_low', 'smooth_high', 'rough_slope', 'stairs_low', 'stairs_high', 'obstacles']
BASE_COUNTS = [1, 1, 1, 4, 3, 2]
SOURCE_COLUMNS = [[1], [2], [3], [4, 5, 6, 7], [8, 9, 10], [11, 12]]
FAMILIES = ['forward', 'diagonal', 'lateral', 'sine']
PROFILES = [
    dict(id='F01', family='forward', yaw=0., amplitude=0.),
    dict(id='F02', family='forward', yaw=0., amplitude=0.),
    dict(id='DL', family='diagonal', yaw=-math.pi/4, amplitude=0.),
    dict(id='DR', family='diagonal', yaw=math.pi/4, amplitude=0.),
    dict(id='L', family='lateral', yaw=-math.pi/2, amplitude=0.),
    dict(id='R', family='lateral', yaw=math.pi/2, amplitude=0.),
    dict(id='S20', family='sine', yaw=None, amplitude=2.),
    dict(id='S15', family='sine', yaw=None, amplitude=1.5),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def stable_seed(*args):
    return int(hashlib.sha256(json.dumps(args).encode()).hexdigest()[:8], 16) % (2**31-1)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def conditions():
    return [dict(id='healthy', joint=None, loss=0.)] + [
        dict(id='j%02d_d%03d' % (j, round(100*d)), joint=j, loss=d)
        for j in range(12) for d in [.2, .4, .6, .8, 1.]]


def binary_survival(termination):
    if termination == 'fall': return 0
    if termination in ('goal', 'administrative_cap'): return 1
    raise ValueError('Incomplete/technical-error runs are not survival observations')


def make_protocol():
    old = json.loads((ROOT/'legged_gym/evaluation/ieie_mixed_collection_v1.json').read_text())
    return dict(protocol_id='ieie_eight_course_v1', policies=old['policies'],
                profiles=PROFILES, classes=CLASSES, baseline_column_counts=BASE_COUNTS,
                exploration_seeds=[1101,1102,1103], confirmation_seeds=[2101,2102,2103],
                start_x=-1., end_x=40.5, wavelength=16., speed=.5, dt=.02,
                onset_s=2., survival_definition='binary: fall=0; goal or administrative_cap without fall=1',
                stall_window_s=30., stall_progress_m=.5,
                budget_speed=.1, budget_margin_s=30., controller=dict(lateral_gain=.7, heading_gain=1.5,
                lookahead_m=.4, max_yaw_rate=.8), noise=False, pushes=False,
                selection=dict(task_units=8, task_family_bounds=[1,3],
                terrain_counts_change_max=1, terrain_count_min=1,
                objective='maximize survival gain subject to every tracking gain >= uniform baseline',
                confirmation_task_units=8, confirmation_family_count_bounds=[1,3],
                confirmation_within_family='counterbalanced subprofile allocation across conditions and seeds; same total run budget'),
                trajectory_metrics='post-onset pre-action valid samples; terminal states stored separately',
                early_goal_survival='survived=1 at successful goal; no fixed 60-second window',
                fault_model='training-compatible single-joint motor-strength multiplier before torque clipping')


def route_length(p, amplitude):
    x=np.linspace(p['start_x'],p['end_x'],8193)
    slope=amplitude*2*np.pi/p['wavelength']*np.cos(2*np.pi*(x-p['start_x'])/p['wavelength'])
    return float(np.trapz(np.sqrt(1+slope*slope),x))


def specs(p, stage, seed, conds, family_repeats=None, column_counts=None):
    repeats=family_repeats or [1]*4
    counts=column_counts or BASE_COUNTS
    cols=[]
    for k,count in enumerate(counts):
        for i in range(count):
            cols.append((CLASSES[k], SOURCE_COLUMNS[k][i % len(SOURCE_COLUMNS[k])]))
    rows=[]
    for cond in conds:
        allocations={profile['id']:repeats[FAMILIES.index(profile['family'])] for profile in p['profiles']}
        if stage=='confirmation' and family_repeats is not None:
            allocations={profile['id']:0 for profile in p['profiles']}
            cond_index=next(i for i,c in enumerate(conditions()) if c['id']==cond['id'])
            seed_index=p['confirmation_seeds'].index(seed)
            for fi,family in enumerate(FAMILIES):
                members=[profile for profile in p['profiles'] if profile['family']==family]
                offset=(cond_index+seed_index+fi)%2
                for count in range(repeats[fi]):
                    allocations[members[(count+offset)%2]['id']]+=1
        for profile in p['profiles']:
            for rep in range(allocations[profile['id']]):
                length=route_length(p,profile['amplitude'])
                cap=math.ceil((length/p['budget_speed']+p['budget_margin_s'])/p['dt'])*p['dt']
                for slot,(cls,col) in enumerate(cols):
                    group='%s_s%d_%s_r%d_slot%02d' % (stage,seed,profile['id'],rep,slot)
                    pair_id=group+'_'+cond['id']
                    rows.append(dict(profile=profile, repeat=rep, terrain_class=cls, terrain_column_id=col,
                        allocation_slot=slot, condition=cond, paired_trial_id=pair_id,
                        physical_pair_group_id=group, initial_seed=stable_seed('init',group),
                        physics_seed=stable_seed('physics',group), seed=seed, stage=stage,
                        arc_length_m=length, post_cap_s=cap))
    return rows


def make_jobs(p, stage, family_repeats=None, column_counts=None, max_envs=3264):
    seeds=p[stage+'_seeds']
    n_per_cond=len(specs(p,stage,seeds[0],conditions()[:1],family_repeats,column_counts))
    capacity=max(1,max_envs//n_per_cond)
    groups=[conditions()[:1]]+[conditions()[i:i+capacity] for i in range(1,61,capacity)]
    jobs=[]
    for group_index, group in enumerate(groups):
        for seed in seeds:
            for policy in p['policies']:
                jobs.append(dict(id='%s_g%02d_s%d_%s'%(stage,group_index,seed,policy),
                    pair_job='%s_g%02d_s%d'%(stage,group_index,seed), stage=stage,
                    seed=seed, policy=policy, conditions=group, family_repeats=family_repeats or [1]*4,
                    column_counts=column_counts or BASE_COUNTS, num_envs=n_per_cond*len(group)))
    return jobs
