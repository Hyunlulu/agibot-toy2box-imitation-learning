"""Generate actual constrained trajectories; every executed/recorded rx/rz is zero."""
from pathlib import Path
import argparse,fcntl,json,runpy,sys,time,traceback
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robomimic'))
from action_adapter import generation_complete,install_generation_constraint
from progress_tracker import GenerationMonitor
from generation_cli import generation_arguments

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--pilot_attempts',type=int,default=None)
parser.add_argument('--tag',default='generation')
if '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
    parser.print_help()
    print('IsaacLab options: --task TASK --input_file FILE --output_file FILE --num_envs 16 --generation_num_trials 1000 --headless')
    raise SystemExit(0)
options,remaining=parser.parse_known_args()
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'artifacts/action5'
RUN.mkdir(parents=True,exist_ok=True)
script=ROOT/'scripts/imitation_learning/isaaclab_mimic/generate_dataset.py'
if options.pilot_attempts is not None and options.pilot_attempts<=0:raise ValueError('Pilot attempt bound must be positive')
# Parse effective duplicate options and the real .hdf5 destinations, then recheck after locking below.
remaining,output=generation_arguments(remaining,options.pilot_attempts)
lock=(RUN/'gpu_stage.lock').open('a')
print('Waiting for pipeline GPU stage lock',flush=True);fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
print('Acquired pipeline GPU stage lock',flush=True)
# A queued invocation must see files written by the previous lock holder.
for path in [output,output.with_name(output.stem+'_failed.hdf5'),RUN/f'{options.tag}_result.json',RUN/f'{options.tag}_progress.json']:
    if path.exists():raise FileExistsError(f'Preserve existing generation output: {path}')
sys.argv=[str(script),*remaining]
monitor=GenerationMonitor(RUN,options.tag,pilot_attempts=options.pilot_attempts)
namespace=runpy.run_path(str(script),run_name='constrained_generation')
main_globals=namespace['main'].__globals__
original_config=main_globals['setup_env_config'];original_setup=main_globals['setup_async_generation']

def constrained_config(*args,**kwargs):
    cfg,success_term=original_config(*args,**kwargs)
    cfg.datagen_config.generation_guarantee=options.pilot_attempts is None
    cfg.datagen_config.generation_num_trials=1000 if options.pilot_attempts is None else options.pilot_attempts
    print('ACTION5_GENERATION_PROTOCOL',json.dumps({'guarantee_successes':cfg.datagen_config.generation_guarantee,
        'num_trials':cfg.datagen_config.generation_num_trials,'mask_indices':[3,5],'success_criteria_changed':False}),flush=True)
    return cfg,success_term

def checked_setup(*args,**kwargs):
    env=kwargs.get('env',args[0] if args else None)
    monitor.constraint_stats=install_generation_constraint(env)
    print('ACTION5_CONSTRAINT_INSTALLED: after target conversion/noise and before env.step/recording',flush=True)
    parts=original_setup(*args,**kwargs);loop=parts['event_loop'];original_run=loop.run_until_complete
    last_print=time.monotonic()
    def checked_run(awaitable):
        nonlocal last_print
        result=original_run(awaitable)
        for task in parts['tasks']:
            if task.done() and not task.cancelled():task.result()
        import isaaclab_mimic.datagen.generation as generation
        complete=generation_complete(generation.num_success,generation.num_attempts,pilot_attempts=options.pilot_attempts)
        if complete or time.monotonic()-last_print>20:
            state=monitor.update(generation.num_success,generation.num_attempts,generation.num_failures)
            print('CONSTRAINED_GENERATION_PROGRESS',json.dumps(state),flush=True)
            last_print=time.monotonic()
        return result
    loop.run_until_complete=checked_run
    return parts

main_globals['setup_env_config']=constrained_config
main_globals['setup_async_generation']=checked_setup
try:
    namespace['main']()
    import isaaclab_mimic.datagen.generation as generation
    result=monitor.update(generation.num_success,generation.num_attempts,generation.num_failures)
    if not result['completed']:raise RuntimeError(f'Constrained generation stopped before target: {result}')
except BaseException:
    (RUN/f'{options.tag}_error.txt').write_text(traceback.format_exc());raise
finally:
    namespace['simulation_app'].close()
