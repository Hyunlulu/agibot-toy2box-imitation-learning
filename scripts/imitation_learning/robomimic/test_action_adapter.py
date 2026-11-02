import unittest
import numpy as np
import torch
try:
    import action_adapter as A
except ModuleNotFoundError:
    A=None

class ActionAdapterTest(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(A,'5D action adapter has not been implemented')
    def test_numpy_mapping_keeps_y_rotation_and_gripper_in_correct_slots(self):
        x=np.array([1,2,3,4,5],dtype=np.float32)
        np.testing.assert_array_equal(A.expand_action5(x),[1,2,3,0,4,0,5])
        self.assertEqual(A.expand_action5(x).dtype,np.float32)
    def test_mask_preserves_allowed_components_and_does_not_mutate_input(self):
        x=np.array([[1,2,3,44,4,66,5]],dtype=np.float64)
        result=A.mask_rotation7(x)
        np.testing.assert_array_equal(result,[[1,2,3,0,4,0,5]])
        np.testing.assert_array_equal(x,[[1,2,3,44,4,66,5]])
    def test_torch_sequence_mapping_preserves_dtype_device_and_batch_axes(self):
        x=torch.tensor([[[1,2,3,0,4,0,5],[6,7,8,0,9,0,10]]],dtype=torch.float64)
        short=A.reduce_action7(x)
        torch.testing.assert_close(short,torch.tensor([[[1,2,3,4,5],[6,7,8,9,10]]],dtype=torch.float64))
        torch.testing.assert_close(A.expand_action5(short),x)
        self.assertEqual(short.device,x.device)
    def test_wrong_shape_and_nonfinite_policy_actions_are_rejected(self):
        for x in [np.zeros(7),np.zeros((2,6)),np.array([1,2,np.nan,4,5])]:
            with self.assertRaises(ValueError):A.expand_action5(x)
    def test_guard_rejects_any_nonzero_removed_rotation(self):
        with self.assertRaises(ValueError):A.assert_constrained7(np.array([1,2,3,1e-9,4,0,5]))
    def test_bound_pilot_stops_on_attempts_even_with_no_success(self):
        self.assertFalse(A.generation_complete(0,127,pilot_attempts=128))
        self.assertTrue(A.generation_complete(0,128,pilot_attempts=128))
        self.assertFalse(A.generation_complete(999,6000))
        self.assertTrue(A.generation_complete(1000,6001))
    def test_policy_expansion_preserves_episode_reset_and_observations(self):
        class Policy:
            def start_episode(self):self.n=0
            def __call__(self,obs):self.n+=1;return np.array([obs['x'],2,3,4,self.n],dtype=np.float32)
        policy=A.ExpandedPolicy(Policy());policy.start_episode();obs={'x':8}
        np.testing.assert_array_equal(policy(obs),[8,2,3,0,4,0,1]);self.assertEqual(obs,{'x':8})
        policy.start_episode();np.testing.assert_array_equal(policy(obs),[8,2,3,0,4,0,1])
    def test_generation_constraint_changes_executed_action_before_underlying_step(self):
        class Env:
            def __init__(self):self.executed=[]
            def target_eef_pose_to_action(self,*a,**kw):return np.array([1,2,3,44,4,66,5],dtype=np.float32)
            def step(self,a):self.executed.append(a.copy());return 'stepped'
        env=Env();stats=A.install_generation_constraint(env)
        self.assertEqual(env.step(env.target_eef_pose_to_action({},{},action_noise_dict={'eef':.01})),'stepped')
        np.testing.assert_array_equal(env.executed[0],[1,2,3,0,4,0,5])
        self.assertEqual(stats['guarded_step_calls'],1)
        with self.assertRaises(ValueError):env.step(np.ones(7))
        self.assertEqual(len(env.executed),1)
if __name__=='__main__':unittest.main()
