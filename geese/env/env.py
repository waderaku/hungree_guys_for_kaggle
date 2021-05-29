from typing import List, Tuple

import numpy as np
from geese.constants import NUM_GEESE
from geese.env.dena_env import Environment as DenaEnv
from geese.structure import Observation, Reward
from geese.structure.parameter import EnvParameter
from geese.util.converter import action2int
from kaggle_environments.envs.hungry_geese.hungry_geese import Action


class Env:

    def __init__(self, parameter: EnvParameter):
        self._env = DenaEnv()
        self._reward_list = parameter.reward_list

    def reset(self) -> List[Observation]:
        self._env.reset()
        return [self._env.observation(p) for p in range(NUM_GEESE)]

    def step(self, actions: List[Action]) -> Tuple[List[Observation], List[Reward], List[bool]]:
        # 今回死ぬGooseを判定するために、１個前のStateですでに死んでいるかどうかを保持
        pre_done = np.array([self._env.env.state[p]["status"]
                             != "ACTIVE" for p in range(NUM_GEESE)])
        actions = {p: action2int(actions[p]) for p in range(NUM_GEESE)}
        # Envを次の状態へ遷移させる
        self._env.step(actions)

        # Gooseごとの終了判定
        done = np.array([self._env.env.state[p]["status"]
                         != "ACTIVE" for p in range(NUM_GEESE)])

        # Envの報酬
        env_reward = [self._env.env.state[p]["reward"]
                      for p in range(len(actions))]

        # 順位に基づくRawRewardの計算
        raw_reward = np.array(self._compute_reward(env_reward))

        # 前回生きていて(1 - pre_done)今回死んだ(done)GooseにのみRewardをリターン
        reward: list = ((1 - pre_done) * done * raw_reward).tolist()

        # 全Geeseが終了したらリセット
        if sum(map(int, done)) == NUM_GEESE:
            self._env.reset()

        # Gooseごとの観測
        observation = [self._env.observation(p) for p in range(NUM_GEESE)]
        return observation, reward, done

    def __str__(self) -> str:
        return str(self._env)

    def _compute_reward(self, env_rewards: List[float]) -> List[Reward]:
        # env_rewardが小さい順に(index, env_reward)を格納したリスト
        target = [(i, v) for i, v in zip(range(len(env_rewards)), env_rewards)]
        target.sort(key=lambda x: x[1])

        # 順位配列（スコアが等しいときは繰り下げ順位を適用）
        charge = 1
        rank = NUM_GEESE
        state = -1
        rank_array = [None for _ in range(len(target))]
        for i, v in target:
            if state != v:
                rank -= charge
                charge = 1
                state = v
            else:
                charge += 1
            assert 0 <= rank < NUM_GEESE
            rank_array[i] = rank

        return [self._reward_list[rank] for rank in rank_array]
