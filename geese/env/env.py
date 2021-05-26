from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from geese.env.dena_env import Environment as DenaEnv
from geese.structure import Observation, Reward
from typing import List, Tuple
from geese.constants import NUM_GEESE


def action2int(action: Action) -> int:
    if action == Action.NORTH:
        return 0
    elif action == Action.SOUTH:
        return 1
    elif action == Action.WEST:
        return 2
    elif action == Action.EAST:
        return 3
    else:
        raise ValueError("Unexpected Action Input")


class Env:

    def __init__(self):
        self._env = DenaEnv()

    def reset(self) -> List[Observation]:
        self._env.reset()
        return [self._env.observation(p) for p in range(NUM_GEESE)]

    def step(self, actions: List[Action]) -> Tuple[List[Observation], List[Reward], List[bool]]:
        actions = {p: action2int(actions[p]) for p in range(NUM_GEESE)}
        self._env.step(actions)
        reward = [self._env.env.state[p]["reward"]
                  for p in range(len(actions))]
        done = [True if self._env.env.state[p]["status"] != "ACTIVE"
                else False for p in range(NUM_GEESE)]
        return [self._env.observation(p) for p in range(NUM_GEESE)], reward, done

    def __str__(self) -> str:
        return str(self._env)
