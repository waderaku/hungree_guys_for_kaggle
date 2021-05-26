from geese.structure.observation import Observation
from geese.env import Env
from geese.agent.model.model import BaseModel
from geese.agent.ppo_agent import PPOAgent


class PPOController():
    def __init__(self):
        pass

    def run(self):
        agent = PPOAgent(BaseModel())
        env = Env()
        obs_list = env.reset()
        done = True

        while done:
            action_list = [agent.step(obs) for obs in obs_list]
            env.step(action_list)
