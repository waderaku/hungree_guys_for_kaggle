from geese.util.converter import action2int
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from geese.env.dena_env import Environment as DenaEnv
from geese.structure import Observation, Reward
from typing import List, Tuple
from geese.constants import NUM_GEESE
from scipy.stats import rankdata
from geese.structure.parameter import EnvParameter


class Env:

    def __init__(self, parameter: EnvParameter):
        self._env = DenaEnv()
        self._reward_list = parameter.reward_list

    def reset(self) -> List[Observation]:
        self._env.reset()
        return [self._env.observation(p) for p in range(NUM_GEESE)]

    def step(self, actions: List[Action]) -> Tuple[List[Observation], List[Reward], List[bool]]:
        actions = {p: action2int(actions[p]) for p in range(NUM_GEESE)}
        # Envを次の状態へ遷移させる
        self._env.step(actions)

        # 順位に基づくRewardの計算
        rank = rankdata([self._env.env.state[p]["reward"]
                         for p in range(len(actions))])
        reward = list(map(self._rank2reward, rank))

        # Gooseごとの終了判定
        done = [self._env.env.state[p]["status"]
                != "ACTIVE" for p in range(NUM_GEESE)]

        # 全Geeseが終了したらリセット
        if sum(map(int, done)) == NUM_GEESE:
            self._env.reset()

        # Gooseごとの観測
        observation = [self._env.observation(p) for p in range(NUM_GEESE)]
        return observation, reward, done

    def __str__(self) -> str:
        return str(self._env)

    def _rank2reward(self, rank: int) -> float:
        return self._reward_list[rank - 1]
