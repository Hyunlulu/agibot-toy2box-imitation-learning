"""Five commanded actions, with exact zero rx/rz at the seven-action environment boundary."""
import functools
import numpy as np
import torch

KEEP=[0,1,2,4,6]
REMOVED=[3,5]


def _checked(actions,dimension):
    value=actions if isinstance(actions,torch.Tensor) else np.asarray(actions)
    if value.ndim<1 or value.shape[-1]!=dimension:
        raise ValueError(f'Expected last action dimension {dimension}, got {value.shape}')
    if isinstance(value,torch.Tensor):finite=bool(torch.isfinite(value).all())
    else:finite=np.issubdtype(value.dtype,np.number) and bool(np.isfinite(value).all())
    if not finite:raise ValueError('Actions must be finite numeric values')
    return value


def mask_rotation7(actions):
    value=_checked(actions,7)
    result=value.clone() if isinstance(value,torch.Tensor) else value.copy()
    result[...,REMOVED]=0
    return result


def assert_constrained7(actions):
    value=_checked(actions,7)
    nonzero=bool(torch.count_nonzero(value[...,REMOVED])) if isinstance(value,torch.Tensor) else bool(np.count_nonzero(value[...,REMOVED]))
    if nonzero:raise ValueError('Executed/recorded rx and rz must be exactly zero')
    return value


def reduce_action7(actions):
    return assert_constrained7(actions)[...,KEEP]


def expand_action5(actions):
    value=_checked(actions,5)
    shape=(*value.shape[:-1],7)
    result=value.new_zeros(shape) if isinstance(value,torch.Tensor) else np.zeros(shape,dtype=value.dtype)
    result[...,KEEP]=value
    return result


def generation_complete(successes,attempts,target_successes=1000,pilot_attempts=None):
    target=target_successes if pilot_attempts is None else pilot_attempts
    if target<=0:raise ValueError('Generation target must be positive')
    return (successes if pilot_attempts is None else attempts)>=target


class ExpandedPolicy:
    def __init__(self,policy):
        self.policy=policy
        self.calls=0
    def start_episode(self):
        self.calls=0
        return self.policy.start_episode()
    def __call__(self,observations):
        actions=expand_action5(self.policy(observations))
        self.calls+=1
        return actions
    def __getattr__(self,name):
        return getattr(self.policy,name)


def prepare_policy(policy, policy_dimension, environment_dimension, task, *, norm_factor_min=None, norm_factor_max=None):
    """Keep compatible policies intact; expand only the course Agibot 5D policy."""
    if policy_dimension <= 0 or environment_dimension <= 0:
        raise ValueError('Action dimensions must be positive')
    if policy_dimension == environment_dimension:
        return policy
    if (policy_dimension, environment_dimension) == (5, 7) and task == 'Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0':
        if norm_factor_min is not None or norm_factor_max is not None:
            raise ValueError('The submitted 5D policy uses raw actions; normalization bounds would alter zero rotation constraints')
        return ExpandedPolicy(policy)
    raise ValueError(f'Unsupported policy/environment action dimensions: {policy_dimension}/{environment_dimension} for {task}')


def install_generation_constraint(env):
    """Project commands after the original conversion/noise; reject any unmasked step."""
    original_target=env.target_eef_pose_to_action
    original_step=env.step
    stats={'projected_action_calls':0,'guarded_step_calls':0,'max_executed_removed_abs':0.0}
    @functools.wraps(original_target)
    def projected(*args,**kwargs):
        result=mask_rotation7(original_target(*args,**kwargs))
        stats['projected_action_calls']+=1
        return result
    @functools.wraps(original_step)
    def checked_step(actions,*args,**kwargs):
        assert_constrained7(actions)
        stats['guarded_step_calls']+=1
        return original_step(actions,*args,**kwargs)
    env.target_eef_pose_to_action=projected
    env.step=checked_step
    return stats


def rollout_action5(original_rollout,policy,env,success_term,horizon,device):
    result=original_rollout(ExpandedPolicy(policy),env,success_term,horizon,device)
    success,trajectory=result
    if trajectory['actions']:assert_constrained7(np.asarray(trajectory['actions']))
    return success,trajectory
