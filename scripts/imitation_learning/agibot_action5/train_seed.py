"""Train a genuine five-output BC-RNN with unchanged production hyperparameters."""
import argparse,fcntl,json,runpy,sys,time
from pathlib import Path
from types import SimpleNamespace
import h5py,torch

parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(101,111),required=True)
parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--preflight',action='store_true')
options=parser.parse_args();ROOT=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent;RUN=ROOT/'artifacts/action5'
RUN.mkdir(parents=True,exist_ok=True)
with h5py.File(options.dataset,'r') as f:
    if f.attrs.get('policy_action_dimension')!=5:raise ValueError('Training dataset must declare five action outputs')
    if not options.preflight and len(f['data'])!=1000:raise ValueError('Production training requires exactly1000 successes')
    if any(ep['actions'].shape[1]!=5 for ep in f['data'].values()):raise ValueError('Every training action must be5D')
lock=(RUN/'gpu_stage.lock').open('a')
print('Waiting for pipeline GPU stage lock',flush=True);fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
print('Acquired pipeline GPU stage lock',flush=True)
config=CODE/f'configs/bc_seed{options.seed}.json';epochs=1 if options.preflight else 2000
suffix=f'preflight_seed{options.seed}' if options.preflight else f'seed{options.seed}'
name=f'bc_rnn_toy2box_source100_v2_action5_mimic1000_{suffix}'
result_path=RUN/f'training_{suffix}_result.json'
if result_path.exists():raise FileExistsError(result_path)
task='Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0'
experiment_path=ROOT/'logs/agibot_submission_training'/task/name
if experiment_path.exists():raise FileExistsError(f'Preserve existing training experiment: {experiment_path}')
if options.preflight:
    cfg=json.loads(config.read_text());cfg['experiment']['save']['every_n_epochs']=1
    cfg['experiment']['epoch_every_n_steps']=2;cfg['train']['num_data_workers']=0;cfg['train']['num_epochs']=1
    config=RUN/f'preflight_seed{options.seed}.json'
    with config.open('x') as stream:json.dump(cfg,stream,indent=2)
script=ROOT/'scripts/imitation_learning/robomimic/train.py';sys.argv=[str(script)];started=time.time()
namespace=runpy.run_path(str(script),run_name='action5_training')
try:
    namespace['gym'].spec(task).kwargs['robomimic_bc_cfg_entry_point']=str(config)
    args=SimpleNamespace(task=task,algo='bc',dataset=str(options.dataset.resolve()),name=name,epochs=epochs,
                         log_dir='agibot_submission_training',normalize_training_actions=False)
    namespace['main'](args)
    checkpoints=sorted((ROOT/'logs/agibot_submission_training'/task/name).glob(f'**/model_epoch_{epochs}.pth'))
    if len(checkpoints)!=1:raise RuntimeError(f'Expected one final checkpoint, found {checkpoints}')
    c=torch.load(checkpoints[0],map_location='cpu',weights_only=False);shape=c['shape_metadata']
    baseline=json.loads((CODE/'baseline_shape_metadata.json').read_text())
    if shape['ac_dim']!=5:raise RuntimeError('Model checkpoint is not genuinely five-output')
    if shape['all_shapes']!=baseline['all_shapes']:raise RuntimeError('Policy observation dimensions changed')
    result={'seed':options.seed,'epochs':epochs,'dataset':str(options.dataset.resolve()),'checkpoint':str(checkpoints[0]),
            'wall_time_seconds':time.time()-started,'policy_action_dimension':5,'environment_action_dimension':7,
            'observation_dimension':sum(v[0] for v in shape['all_shapes'].values()),'preflight_only':options.preflight}
    result_path.write_text(json.dumps(result,indent=2)+'\n');print('ACTION5_TRAINING_COMPLETE',json.dumps(result),flush=True)
finally:namespace['simulation_app'].close()
