
from geese.env.env import Env
from geese.constants import NUM_GEESE, TIME_LIMIT

import tensorflow as tf
import numpy as np
import time
import math
from typing import List

from geese.env import Env
from geese.util.converter import to_tf_tensor
from geese.structure import Observation


class MCTS:
    def __init__(self,
                 model: tf.keras.models.Model,
                 env: Env,
                 cpuct: float = 1.0,
                 eps: float = 1e-8):
        self.model = model
        self._env = env
        self._cpuct = cpuct
        self._eps = eps
        self._last_obs = None

        self._num_state_action = {}
        self._num_state = {}
        self._policy_state = {}
        self._valid_state = {}
        self._Q_state_action = {}

    def get_prob(self, obs: Observation) -> np.ndarray:
        start_time = time.time()
        obs.last_obs = self._last_obs

        # 一定時間treeを展開する
        while time.time() - start_time < TIME_LIMIT:
            self.search(obs)

        s = self._env.get_representation(obs)
        i = obs.index
        counts = [
            self._num_state_action[(s, i, a)] if (
                s, i, a) in self._num_state_action else 0
            for a in range(self._env.get_action_size())
        ]
        prob = counts / np.sum(counts)
        self._last_obs = obs
        return np.array(prob)

    def search(self, obs: Observation) -> List[float]:
        state = self._env.get_representation(obs)

        # 初めて来た場所なら、そこで展開
        if state not in self._num_state:

            # 価値の初期化
            value_list = [-10] * NUM_GEESE
            for i in range(NUM_GEESE):
                if len(obs.now_obs.geese[i]) == 0:
                    continue

                # 葉ノード
                # model.predictこれで問題ないのか？
                self._policy_state[(state, i)], value_list[i] = self.model.predict(to_tf_tensor(
                    obs, i))

                self._policy_state[(state, i)] = self._policy_state[(
                    state, i)].numpy()
                value_list[i] = value_list[i].numpy()

                # 行動可能エリアでフィルターをかける
                valid_list = self._env.get_valid_moves(obs,  i)
                self._policy_state[(state, i)] = self._policy_state[(state, i)] * \
                    valid_list

                # フィルターの結果から、policyの確率変換
                sum_policy_state = np.sum(self._policy_state[(state, i)])
                if sum_policy_state > 0:
                    self._policy_state[(state, i)] /= sum_policy_state

                # stateに対する初期化
                self._valid_state[(state, i)] = valid_list
                self._num_state[state] = 0
            return value_list

        best_acts = [None] * 4
        for i in range(4):
            if len(obs.now_obs.geese[i]) == 0:
                continue

            valid_list = self._valid_state[(state, i)]
            cur_best = -float('inf')
            best_act = self._env.action_list[-1]

            # best_actionの調査
            for a in range(self._env.get_action_size()):
                if valid_list[a]:
                    if (state, i, a) in self._Q_state_action:
                        u = self._Q_state_action[(state, i, a)] + self._cpuct *\
                            self._policy_state[(state, i)][a] * math.sqrt(
                            self._num_state[state]) / (1 + self._num_state_action[(state, i, a)])
                    else:
                        u = self._cpuct * self._policy_state[(state, i)][a] * math.sqrt(
                            self._num_state[state] + self._eps)

                    if u > cur_best:
                        cur_best = u
                        best_act = self._env.action_list[a]

            best_acts[i] = best_act

        next_obs = self._env.step(obs, best_acts)
        value_list = self.search(next_obs)

        for i in range(4):
            if len(obs.now_obs.geese[i]) == 0:
                continue

            a = self._env.action_list.index(best_acts[i])
            v = value_list[i]

            # Qの更新
            if (state, i, a) in self._Q_state_action:
                self._Q_state_action[(state, i, a)] = \
                    (self._num_state_action[(state, i, a)] *
                     self._Q_state_action[(state, i, a)] + v) / \
                    (self._num_state_action[(state, i, a)] + 1)

                self._num_state_action[(state, i, a)] += 1

            else:
                self._Q_state_action[(state, i, a)] = v
                self._num_state_action[(state, i, a)] = 1

        self._num_state[state] += 1
        return value_list
