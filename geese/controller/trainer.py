from geese.env import Env
from geese.agent import Agent


class Trainer:
    def __init__(self, agent: Agent):
        self._agent = agent

    def train(self, num_epochs: int = 100) -> None:
        raise NotImplementedError
