from copy import deepcopy
from typing import List

from geese.structure import Observation
from kaggle_environments.envs.hungry_geese.hungry_geese import (Action,
                                                                translate)
from kaggle_environments.helpers import histogram


class Env():
    def __init__(
        self,
        num_rows: int = 7,
        num_colums: int = 11,
        action_list: List[Action] = [Action.NORTH,
                                     Action.SOUTH, Action.WEST, Action.EAST],
        hunger_rate: int = 40
    ):
        self._num_rows = num_rows
        self._num_columns = num_colums
        self.action_list = action_list
        self._hunger_rate = hunger_rate

    def get_action_size(self) -> int:
        return len(self.action_list)

    def step(self,
             obs: Observation,
             direction_list: List[int]
             ) -> Observation:
        last_obs = obs.last_obs
        obs = obs.now_obs
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
                             self._num_columns, self._num_rows)

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
            if next_obs.step % self._hunger_rate == 0:
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

    def get_valid_moves(
            self,
            obs: Observation,
            index: int) -> List[bool]:
        last_obs = obs.last_obs
        obs = obs.now_obs
        geese = obs.geese
        pos = geese[index][0]
        obstacles = {position for goose in geese for position in goose[:-1]}
        if last_obs is not None:
            obstacles.add(last_obs.geese[index][0])

        valid_moves = [
            translate(pos, action, self._num_columns,
                      self._num_rows) not in obstacles
            for action in self.action_list
        ]

        return valid_moves

    def get_representation(self, obs: Observation) -> str:
        obs = obs.now_obs
        return str(obs.geese + obs.food)
