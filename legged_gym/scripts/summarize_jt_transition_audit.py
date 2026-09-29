"""Aggregate staged checkpoint diagnostics without the legacy recovery metric."""
import json
import statistics
from pathlib import Path

KEYS = ['survived_post_failure_horizon','tracking_time_fraction','final_5s_tracking_fraction',
        'post_command_vx_rmse','post_command_vy_rmse','post_yaw_rate_rmse','post_mean_vx']


def aggregate(rows):
    result = {'n':len(rows)}
    for key in KEYS:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = statistics.mean(values) if values else None
    return result


def main():
    root = Path('logs/evaluations/jt-transition-audit-20260928')
    files = list((root/'screen').glob('student_*_seed*.jsonl')) + list((root/'confirm').glob('student_*_seed*.jsonl'))
    sources = [(int(f.stem.split('_')[1]),f) for f in files if '.trace.' not in f.name]
    base = Path('logs/evaluations/onset-v2-diagnostic-20260928')
    sources += [(71500,base/'student_seed{}.jsonl'.format(s)) for s in [101,102,103]]
    results = {}; raw = {}
    for it,f in sources:
        with f.open() as stream:
            meta=json.loads(next(stream));rows=[json.loads(l) for l in stream]
        seed=meta['seed']; assert (it,seed) not in raw
        assert len(rows)==3456
        reference=json.loads((base/'student_seed{}.jsonl'.format(seed)).open().readline())
        assert meta['initial_state_hashes']==reference['initial_state_hashes']
        assert meta['specs_sha256']==reference['specs_sha256']
        raw[it,seed]=rows
        results.setdefault(str(it),{'seeds':{},'sources':[]})
        results[str(it)]['sources'].append(str(f))
        results[str(it)]['seeds'][str(seed)]={
            'damaged':aggregate([r for r in rows if r['degradation_rate']>0]),
            'by_severity':{str(d):aggregate([r for r in rows if r['degradation_rate']==d]) for d in [0.,.2,.4,.6,.8,1.]}}
    for it,entry in results.items():
        rows=[r for (i,s),rr in raw.items() if i==int(it) for r in rr]
        entry['aggregate']={'damaged':aggregate([r for r in rows if r['degradation_rate']>0]),
            'by_severity':{str(d):aggregate([r for r in rows if r['degradation_rate']==d]) for d in [0.,.2,.4,.6,.8,1.]}}
    output={'checkpoints':results,'note':'Screen seed101; selected stages confirmed with seeds102/103. Recovery excluded. One training seed.'}
    (root/'summary.json').write_text(json.dumps(output,indent=2))
    for it,e in sorted(results.items(),key=lambda kv:int(kv[0])):
        r=e['seeds']['101']['damaged'];d=e['seeds']['101']['by_severity']['1.0']
        print(it,'seeds',list(e['seeds']),'S {:.2f} Q {:.2f} d1Q {:.2f} yaw {:.3f}'.format(100*r['survived_post_failure_horizon'],100*r['tracking_time_fraction'],100*d['tracking_time_fraction'],r['post_yaw_rate_rmse']))

if __name__ == '__main__':
    main()
