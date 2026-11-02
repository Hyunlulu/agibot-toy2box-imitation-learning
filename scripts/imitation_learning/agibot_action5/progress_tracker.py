"""Persist completion during the event loop, before IsaacSim may fast-exit."""
from pathlib import Path
import json,time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robomimic'))
from action_adapter import generation_complete

class GenerationMonitor:
    def __init__(self,directory,tag,target_successes=1000,pilot_attempts=None):
        self.directory=Path(directory);self.tag=tag;self.target_successes=target_successes
        self.pilot_attempts=pilot_attempts;self.started=time.time();self.constraint_stats={}
    def update(self,successes,attempts,failures):
        state={'successes':int(successes),'attempts':int(attempts),'failures':int(failures),
               'wall_time_seconds':time.time()-self.started,'parallel_envs':16,
               'target_successes':self.target_successes if self.pilot_attempts is None else None,
               'target_attempts':self.pilot_attempts,
               'completed':generation_complete(successes,attempts,self.target_successes,self.pilot_attempts),
               'policy_action_dimension':5,'environment_action_dimension':7,
               'constraint_stats':dict(self.constraint_stats)}
        self._write(f'{self.tag}_progress.json',state)
        if state['completed']:self._write(f'{self.tag}_result.json',state)
        return state
    def _write(self,name,state):
        path=self.directory/name;temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)
