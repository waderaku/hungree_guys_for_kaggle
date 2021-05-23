from geese.structure import Observation
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, translate
from kaggle_environments.helpers import histogram

from typing import List
from copy import deepcopy


class Env():
    def __init__(
        self,
        num_rows: int = 7,
        num_colums: int = 11,
        action_list: List[Action] = [Action.NORTH,
                                     Action.SOUTH, Action.WEST, Action.EAST],
        hunger_rate: int = 40
    ):
        self.num_rows = num_rows
        self.num_columns = num_colums
        self.action_list = action_list
        self.hunger_rate = hunger_rate

    def get_action_size(self) -> int:
        return len(self.action_list)

    def step(self,
             obs: Observation,
             last_obs: Observation,
             direction_list: List[int]
             ) -> Observation:
        next_obs = deepcopy(obs)
        next_obs.step += 1
        geese = next_obs.geese
        food = next_obs.food

        # 各gooseを行動させる
        for i, goose in enumerate(geese):

            if len(goose) == 0:
                continue

            # 今のpositionから、今回のActionをした時に移動するheadのpositionを計算
            head = translate(goose[0], direction_list[i],
                             self.columns, self.rows)

            # 前回の行動の逆の行動をした場合、失格
            if last_obs is not None and head == last_obs.geese[i][0]:
                geese[i] = []
                continue

            # headがfoodの位置に来た場合、そのfoodを食べられるよ！
            if head in food:
                food.remove(head)
            else:
                goose.pop()

            # 頭の移動
            goose.insert(0, head)

            # hunger_rateに一回、小さくなる
            if next_obs.step % self.hunger_rate == 0:
                if len(goose) > 0:
                    goose.pop()

        # 今どこになにがあるかリスト(dict？)
        goose_positions = histogram(
            position
            for goose in geese
            for position in goose
        )

        # Check for collisions.
        # 各ガチョウの頭が誰かの体、頭にぶつかっていた場合、そのガチョウはゲームオーバー
        for i in range(4):
            if len(geese[i]) > 0:
                head = geese[i][0]
                if goose_positions[head] > 1:
                    geese[i] = []

        return next_obs

    def get_representation(self, obs: Observation) -> str:
        return str(obs.geese + obs.food)
