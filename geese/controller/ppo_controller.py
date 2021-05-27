from typing import List
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
import numpy as np
from collections import deque

from geese.controller.ppo_helper import calc_reward, calc_n_step_return
from geese.structure.observation import Observation
from geese.structure.ppo_parameter import PPOParameter as ppp
from geese.structure.sample import Sample
from geese.env.vecenv.vecenv import VecEnv
from geese.agent.model.model import BaseModel
from geese.agent.ppo_agent import PPOAgent
from geese.constants import NUM_GEESE


class PPOController():
    def __init__(self, PPOParameter: ppp):
        self._PPOParameter = PPOParameter

        self.obs_list = []
        self.action_list = []
        self.v_list = []
        self.pi_list = []

    def train(self):
        agent = PPOAgent(BaseModel())
        vec_env = VecEnv()
        obs = vec_env.reset()
        done = True
        reward_q = deque()
        index = 0
        while done:
            action, value, prob = agent.step(
                np.array(obs))
            next_obs, reward, done = vec_env.step(action)

            done = sum(done) == NUM_GEESE

            self._update_PPO_list(obs, action, value, prob)

            obs = next_obs

            reward_q.append(calc_reward(
                done, reward, self._PPOParameter.reward))

            if len(reward_q) < self._PPOParameter.n:
                continue
            n_step_return = calc_n_step_return(
                reward_q, self._PPOParameter.gunnma)
            reward_q.popleft()
        Sample(obs, action,)

    def _update_PPO_list(
        self,
        obs: List[Observation],
        action: List[Action],
        value: np.ndarray,
        prob: np.ndarray
    ) -> None:
        for i in range()
        self.obs_list.append(obs)
        self.action_list.append(action)
        self.v_list.append(value)
        self.pi_list.append(prob)
