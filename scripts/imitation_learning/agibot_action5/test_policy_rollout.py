"""Exercise the real repository rollout function against the 5D policy boundary."""
import ast,copy,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'robomimic'))
import action_adapter as A
class PolicyRolloutTest(unittest.TestCase):
    def test_real_rollout_consumes_five_outputs_without_losing_orientation_observations(self):
        self.assertTrue(hasattr(A,'rollout_action5'),'Evaluation bridge is not implemented')
        tree=ast.parse((Path(__file__).resolve().parents[3]/'scripts/imitation_learning/robomimic/play.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rollout')
        scope={'copy':copy,'torch':torch,'args_cli':SimpleNamespace(norm_factor_min=None,norm_factor_max=None)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'real_rollout','exec'),scope)
        class Policy:
            def start_episode(self):pass
            def __call__(self,obs):
                np.testing.assert_array_equal(obs['eef_quat'].numpy(),[1,0,0,0])
                return np.array([1,2,3,4,-1],dtype=np.float32)
        class Env:
            def __init__(self):
                self.unwrapped=self;self.cfg=SimpleNamespace();self.action_space=SimpleNamespace(shape=(1,7));self.n=0;self.executed=[]
            def obs(self):return {'policy':{'eef_quat':torch.tensor([[1.,0,0,0]])}}
            def reset(self):self.n=0;return self.obs(),{}
            def step(self,actions):
                self.executed.append(actions.numpy().copy());self.n+=1
                return self.obs(),0,False,False,{}
        env=Env();term=SimpleNamespace(func=lambda env:torch.tensor([env.n>=2]),params={})
        success,traj=A.rollout_action5(scope['rollout'],Policy(),env,term,4,'cpu')
        self.assertTrue(success);self.assertEqual(len(traj['actions']),2)
        for actions in env.executed:np.testing.assert_array_equal(actions,[[1,2,3,0,4,0,-1]])
if __name__=='__main__':unittest.main()
