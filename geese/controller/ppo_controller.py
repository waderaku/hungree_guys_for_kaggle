import numpy as np

from geese.structure.observation import Observation
from geese.env import Env
from geese.agent.model.model import BaseModel
from geese.agent.ppo_agent import PPOAgent


class PPOController():

    def train(self):
        agent = PPOAgent(BaseModel())
        env = Env()
        obs_list = env.reset()
        done = True

        while done:
            action_list, value_list, prob_list = agent.step(np.array(obs_list))
            obs_list, reward_list, done_list = env.step(action_list)
