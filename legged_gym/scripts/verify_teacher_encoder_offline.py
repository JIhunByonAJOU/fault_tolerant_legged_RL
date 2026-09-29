"""Validate diagnostic inference against actual source class definitions on CPU.

Extract definitions with AST to avoid importing Isaac Gym/environment packages.
No simulator, optimizer, network training, checkpoint saving, or GPU context.
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path

from diagnose_teacher_encoder_offline import torch,nn,ROOT,TF,JT,CORPUS,mlp,Student,call,sha


def definitions(path,names,scope):
    tree=ast.parse(path.read_text())
    nodes=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    assert {node.name for node in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=ROOT/'logs/evaluations/teacher_encoder_offline_20260929/conformance.json'
    if out.exists():raise FileExistsError(out)
    scope={'torch':torch,'nn':nn,'Normal':torch.distributions.Normal,
           'WIM_OBSERVATION_DIM':235,'PRIVILEGED_OBSERVATION_DIM':45,'TEACHER_LATENT_DIM':8,
           'ACTOR_INPUT_DIM':243,'CURRENT_OBSERVATION_DIM':235,'HISTORY_FRAME_DIM':48,
           'HISTORY_LENGTH':50,'JOINT_OBSERVATION_DIM':2635,'LATENT_DIM':8,'POLICY_INPUT_DIM':243}
    source1=ROOT/'legged_gym/learning/official_wim_teacher_actor_critic.py'
    source2=ROOT/'legged_gym/learning/joint_teacher_student_actor_critic.py'
    definitions(source1,['_activation','_mlp','OfficialWimTeacherActorCritic'],scope)
    definitions(source2,['StudentHistoryEncoder','JointTeacherStudentActorCritic'],scope)
    report={'source_sha256':{str(p):sha(p) for p in [source1,source2]},'comparisons':[]}
    with torch.inference_mode():
        for filename in ['common_teacher_inputs.pt','common_student_inputs.pt']:
            c=torch.load(CORPUS/filename,map_location='cpu')
            indices=torch.linspace(0,len(c['obs'])-1,257).round().long()
            obs,p=c['obs'][indices],c['privileged'][indices]
            for name,path in [('TF43000',TF),('JT45000',JT/'model_45000.pt'),('JT53000',JT/'model_53000.pt')]:
                state=torch.load(path,map_location='cpu')['model_state_dict']
                with contextlib.redirect_stdout(io.StringIO()):
                    model=(scope['OfficialWimTeacherActorCritic'](235,45,12) if name.startswith('TF')
                           else scope['JointTeacherStudentActorCritic'](2635,45,12))
                model.load_state_dict(state,strict=True);model.eval()
                z=model.encode_privileged(p)
                observed=obs[:,:235] if name.startswith('TF') else obs
                a=model.act_inference_with_latent(observed,z)
                zz=call(mlp(state,'teacher_encoder'),p)
                aa=call(mlp(state,'actor'),torch.cat((obs[:,:235],zz),1))
                row={'corpus':filename,'checkpoint':name,'teacher_max_abs':float((z-zz).abs().max()),
                     'actor_max_abs':float((a-aa).abs().max())}
                if name.startswith('JT'):
                    zs=model.encode_history(obs);zss=call(Student(state),obs)
                    row['student_max_abs']=float((zs-zss).abs().max())
                assert all(v<1e-6 for k,v in row.items() if k.endswith('max_abs'))
                report['comparisons'].append(row)
    report['cuda_initialized']=torch.cuda.is_initialized()
    assert not report['cuda_initialized']
    original=json.loads((out.parent/'results.json').read_text())['protocol']
    report['checkpoint_hashes_still_match']=all(sha(original['checkpoints'][k])==v for k,v in original['checkpoint_sha256'].items())
    assert report['checkpoint_hashes_still_match']
    # This is a read-only live-process identity check, not a process control operation.
    pid=602002
    cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    report['existing_training']={'pid':pid,'cmdline':cmdline,'still_same_width64_training':
        'train_jt_student_width_fresh.py' in cmdline and '--student-width 64' in cmdline}
    assert report['existing_training']['still_same_width64_training']
    report['result']='PASS'
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
