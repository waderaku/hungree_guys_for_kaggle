from kaggle_environments import make, evaluate
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, translate
from typing import List


class Env():
    def __init__(
        num_rows: int = 7,
        num_colums: int = 11,
        action_list: List[Action] = [Action.NORTH,
                                     Action.SOUTH, Action.WEST, Action.EAST],
        hunger_rate: int = 40
    ) -> None:
        raise NotImplementedError()

    def getActionSize(self):
        return len(self.actions)

    def step(self,
             obs: Observation,
             last_obs: Observation,
             direction_list: List[int]) -> None:
        return


env = make("hungry_geese", debug=False)
env.reset()
trainer = env.train([None, "greedy", "greedy", "greedy"])

test = trainer.reset()

trainer.step(1)

pass
