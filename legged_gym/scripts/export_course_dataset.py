"""Lossless, portable export of every selected 2:1:1:2 robot and 50 Hz row."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'logs/evaluation/ieie2026_reused2112_20261016'
OUT=ROOT/'datasets/ieie2026_fixed2112'
FIELDS=['values','flags','env_id','step']
lock=threading.Lock();progress={'started_at':time.time(),'completed_jobs':0,'total_jobs':18,'bytes':0,'rows':0}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def payload(arrays):
 h=hashlib.sha256()
 for k in FIELDS:
  a=arrays[k];h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()


def export_job(item):
 job,rows=item;src=Path(rows[0]['source_folder']);dst=OUT/'raw'/job;dst.mkdir(parents=True,exist_ok=True)
 source_manifest=json.loads((src/'trace_manifest.json').read_text());meta=json.loads((src/'metadata.json').read_text())
 n=int(json.loads((src/'COMPLETE.json').read_text())['episodes']);ids=np.array([r['env_id'] for r in rows]);lookup=np.zeros(n,dtype=bool);lookup[ids]=True
 count=np.zeros(n,dtype=np.int64);post=np.zeros(n,dtype=np.int64);chunks=[]
 for info in source_manifest['chunks']:
  f=src/info['path'];assert sha(f)==info['sha256']
  with np.load(f,allow_pickle=False) as z:
   keep=lookup[z['env_id']]
   if not keep.any():continue
   arrays={k:z[k][keep] for k in FIELDS}
  target=dst/info['path'];np.savez_compressed(target,**arrays)
  count+=np.bincount(arrays['env_id'],minlength=n)
  mask=(arrays['flags']&2)>0;post+=np.bincount(arrays['env_id'][mask],minlength=n)
  chunks.append(dict(path=target.name,rows=len(arrays['env_id']),sha256=sha(target),payload_sha256=payload(arrays),
                     first_step=int(arrays['step'].min()),last_step=int(arrays['step'].max()),source_sha256=info['sha256']))
 for r in rows:
  assert count[r['env_id']]==round(r['end_time_s']/.02),(job,r['env_id'],'rows')
  assert post[r['env_id']]==r['valid_post_samples'],(job,r['env_id'],'post')
 for name in ['metadata.json','initial_applied_state.npz','fault_applied.npz']:
  if (src/name).exists():shutil.copy2(src/name,dst/name)
 shutil.copy2(src/'results.json',dst/'source_results.json')
 shutil.copy2(src/'COMPLETE.json',dst/'source_complete.json')
 shutil.copy2(src/'trace_manifest.json',dst/'source_trace_manifest.json')
 dump(dst/'selected_env_ids.json',ids.tolist())
 dump(dst/'selected_results.json',rows)
 dump(dst/'trace_manifest.json',dict(fields=source_manifest['fields'],valid_rows=int(count.sum()),selected_episodes=len(rows),
        source_episodes=n,chunks=chunks,selection='All valid 50 Hz rows of every selected robot; no temporal subsampling',
        extra_initial_states='Initial-state arrays retain original env indexing; select by selected_env_ids.json'))
 size=sum(p.stat().st_size for p in dst.iterdir() if p.is_file())
 with lock:
  progress['completed_jobs']+=1;progress['bytes']+=size;progress['rows']+=int(count.sum());progress['updated_at']=time.time()
  dump(SOURCE/'export_progress.json',progress);print(json.dumps(dict(job=job,**progress)),flush=True)
 return dict(job=job,episodes=len(rows),rows=int(count.sum()),bytes=size,source_folder=str(src))


def main():
 OUT.mkdir(exist_ok=True)
 rows=json.loads((SOURCE/'derived_records.json').read_text());assert len(rows)==26352
 groups={}
 for r in rows:groups.setdefault(r['source_job'],[]).append(r)
 portable=[]
 for r in rows:
  a=dict(r);a['origin_source_folder']=a['source_folder'];a['source_folder']='raw/'+a['source_job'];portable.append(a)
 (OUT/'records.json').write_text(json.dumps(portable,ensure_ascii=False,separators=(',',':'))+'\n')
 for name in ['protocol.json','fixed_design.json','derivation_manifest.json']:
  shutil.copy2(SOURCE/name,OUT/name)
 for name in ['analysis/fixed2112','delivery','explanatory']:
  target=OUT/({'analysis/fixed2112':'analysis','delivery':'reports','explanatory':'explanatory'}[name]);target.mkdir(exist_ok=True)
  for f in (SOURCE/name).iterdir():
   if f.is_file() and f.suffix in ['.csv','.md','.json','.png','.pdf']:shutil.copy2(f,target/f.name)
 parent=ROOT/'logs/evaluation/ieie2026_eight_20261016'
 shutil.copytree(parent/'source_snapshot',OUT/'source_snapshot',dirs_exist_ok=True)
 shutil.copy2(parent/'protocol.json',OUT/'source_snapshot/source_protocol.json')
 shutil.copy2(parent/'source_manifest.json',OUT/'source_snapshot/source_manifest.json')
 dump(SOURCE/'export_progress.json',progress)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:jobs=list(pool.map(export_job,sorted(groups.items())))
 files={str(f.relative_to(OUT)).replace('\\','/'):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in OUT.rglob('*') if f.is_file() and not any(part in ['generated','__pycache__'] for part in f.relative_to(OUT).parts) and f.name not in ['manifest.json','EXPORT_COMPLETE.json']}
 manifest=dict(dataset_id='ieie2026_fixed2112_complete_v1',policy_seed_runs=2196,robot_episodes=26352,paired_trials=13176,
   task_family_ratio=[2,1,1,2],evaluation_seeds=[1101,1102,1103],dt=.02,raw_rows=sum(j['rows'] for j in jobs),
   temporal_subsampling=False,numeric_quantization=False,all_selected_episodes_included=True,jobs=jobs,files=files,
   total_bytes=sum(v['bytes'] for v in files.values()),created_at=time.time())
 dump(OUT/'manifest.json',manifest)
 dump(OUT/'EXPORT_COMPLETE.json',dict(status='COMPLETE',robot_episodes=26352,raw_rows=manifest['raw_rows'],manifest_sha256=sha(OUT/'manifest.json'),completed_at=time.time()))
 print('COMPLETE',json.dumps(dict(episodes=26352,rows=manifest['raw_rows'],bytes=manifest['total_bytes'])),flush=True)


if __name__=='__main__':main()
