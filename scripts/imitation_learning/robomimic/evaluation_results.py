"""Write measured evaluation outcomes once, preserving earlier evidence."""
from pathlib import Path
import json


def write_evaluation(path, *, task, checkpoint, seed, horizon, policy_action_dimension,
                     environment_action_dimension, trials):
    if not trials:
        raise ValueError('At least one evaluated rollout is required')
    successes = sum(bool(row['success']) for row in trials)
    result = dict(task=task, checkpoint=str(checkpoint), evaluation_seed=seed, horizon=horizon,
                  policy_action_dimension=policy_action_dimension,
                  environment_action_dimension=environment_action_dimension,
                  num_rollouts=len(trials), successes=successes,
                  success_rate=successes/len(trials), results=trials)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result
