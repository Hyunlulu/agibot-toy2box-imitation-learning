"""Behavioral checks for the portable course evaluation entry point."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import action_adapter as adapter
try:
    from evaluation_results import write_evaluation
except ImportError:
    write_evaluation = None

TASK = 'Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0'


class ConstantPolicy:
    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.float32)
    def start_episode(self):
        self.steps = 0
    def __call__(self, obs):
        self.steps += 1
        return self.actions.copy()


class SubmissionCompatibilityTest(unittest.TestCase):
    def prepare(self, policy, policy_dim, env_dim, task=TASK):
        helper = getattr(adapter, 'prepare_policy', None)
        self.assertTrue(callable(helper), 'portable checkpoint compatibility has not been implemented')
        return helper(policy, policy_dim, env_dim, task)

    def test_five_output_policy_reaches_correct_seven_action_slots(self):
        policy = self.prepare(ConstantPolicy([1,2,3,4,5]), 5, 7)
        policy.start_episode()
        obs = {'eef_pos': np.array([.7,-.2,.4])}
        np.testing.assert_array_equal(policy(obs), [1,2,3,0,4,0,5])
        np.testing.assert_array_equal(obs['eef_pos'], [.7,-.2,.4])

    def test_matching_seven_output_policy_is_not_wrapped(self):
        original = ConstantPolicy([1,2,3,4,5,6,7])
        self.assertIs(self.prepare(original,7,7), original)

    def test_five_to_seven_expansion_is_restricted_to_the_agibot_task(self):
        with self.assertRaises(ValueError):
            self.prepare(ConstantPolicy([1,2,3,4,5]),5,7,'Unrelated-Task-v0')

    def test_incompatible_checkpoint_dimension_is_rejected(self):
        for policy_dim,env_dim in [(6,7),(5,6),(0,7)]:
            with self.subTest(policy_dim=policy_dim,env_dim=env_dim), self.assertRaises(ValueError):
                self.prepare(ConstantPolicy(np.zeros(max(policy_dim,1))),policy_dim,env_dim)

    def test_five_output_policy_rejects_unnormalization_that_changes_zero_rotations(self):
        helper = getattr(adapter,'prepare_policy',None)
        self.assertTrue(callable(helper))
        with self.assertRaises(ValueError):
            helper(ConstantPolicy([1,2,3,4,5]),5,7,TASK,norm_factor_min=-.2,norm_factor_max=.8)


class EvaluationEvidenceTest(unittest.TestCase):
    def write(self, target, rows):
        self.assertTrue(callable(write_evaluation), 'portable result writer has not been implemented')
        return write_evaluation(target, task=TASK, checkpoint='checkpoints/final.pth', seed=731,
            horizon=800, policy_action_dimension=5, environment_action_dimension=7, trials=rows)

    def test_json_keeps_arbitrary_evaluation_seed_and_actual_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'nested/results.json'
            self.write(target,[{'trial':0,'success':True,'steps':200},{'trial':1,'success':False,'steps':800}])
            value = json.loads(target.read_text())
            self.assertEqual(value['evaluation_seed'],731)
            self.assertEqual(value['successes'],1)
            self.assertEqual(value['num_rollouts'],2)
            self.assertEqual(value['success_rate'],.5)
            self.assertEqual(value['policy_action_dimension'],5)
            self.assertEqual(value['environment_action_dimension'],7)

    def test_existing_evaluation_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'results.json'
            target.write_text('preserve prior evidence')
            with self.assertRaises(FileExistsError):
                self.write(target,[{'trial':0,'success':True,'steps':200}])
            self.assertEqual(target.read_text(),'preserve prior evidence')


if __name__=='__main__':unittest.main()
