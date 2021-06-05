from collections import deque
import datetime
from geese.util.tensor_boad_logger import TensorBoardLogger
from geese.constants import LOG_BASE_DIR
from geese.structure.sample.ppo_sample import PPOSample
from pathlib import Path

import numpy as np
from geese.env.vecenv.solo_vecenv import VecSoloEnv
from geese.controller.ppo_helper import (
    add_delta,
    add_to_que,
    create_padding_data,
    reset_que,
    reset_train_data,
    update_self_PPO_list,
)
from geese.trainer.ppo_trainer import PPOTrainer
from geese.structure.train_data import TrainData
from geese.controller.controller import Controller
from geese.structure.parameter import PPOParameter
from geese.agent import PPOAgent


class PPOSoloController(Controller):
    def __init__(self, ppo_parameter: PPOParameter):
        self._ppo_parameter = ppo_parameter
        self._agent = PPOAgent(ppo_parameter.agent_parameter)

    def train(self) -> None:
        today = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        logger = TensorBoardLogger(f"{LOG_BASE_DIR}/{today}")

        num_parallels = self._ppo_parameter.num_parallels
        train_data = TrainData()
        ppo_trainer = PPOTrainer(self._ppo_parameter.ppo_trainer_parameter, logger)
        vec_solo_env = VecSoloEnv(
            self._ppo_parameter.num_parallels, self._ppo_parameter.env_parameter
        )
        agent = self._agent

        obs_list = vec_solo_env.reset()

        obs_q_list = reset_que(self._ppo_parameter.num_parallels)
        reward_q_list = reset_que(self._ppo_parameter.num_parallels)
        action_q_list = reset_que(self._ppo_parameter.num_parallels)
        value_q_list = reset_que(self._ppo_parameter.num_parallels)
        prob_q_list = reset_que(self._ppo_parameter.num_parallels)
        delta_q_list = reset_que(self._ppo_parameter.num_parallels)

        step = 1
        before_done_list = [False] * self._ppo_parameter.num_parallels
        before_game_done_list = [True] * self._ppo_parameter.num_parallels
        value_o_list = []
        reward_o_list = []

        while True:
            action_list, value_n_list, prob_list = agent.step(obs_list)

            next_obs_list, reward_list, done_list = vec_solo_env.step(action_list)

            [
                add_delta(
                    delta_q_list[i],
                    reward_o_list[i],
                    value_o_list[i],
                    value_n_list[i],
                    self._ppo_parameter.gamma,
                )
                for i in range(num_parallels)
                if not before_game_done_list[i]
            ]

            [
                update_self_PPO_list(
                    reward_q_list[i],
                    obs_q_list[i],
                    action_q_list[i],
                    value_q_list[i],
                    prob_q_list[i],
                    delta_q_list[i],
                    self._ppo_parameter.gae_param,
                    train_data,
                    before_done_list[i],
                )
                for i in range(num_parallels)
                if len(reward_q_list[i]) == self._ppo_parameter.num_step
            ]

            # n回分の行動をキューで管理
            add_to_que(obs_q_list, obs_list)
            add_to_que(action_q_list, action_list)
            add_to_que(value_q_list, value_n_list)
            add_to_que(reward_q_list, reward_list)
            add_to_que(prob_q_list, prob_list)

            # 今回ゲームが終了した場合、パディングしてデータを格納
            [
                create_padding_data(
                    self._ppo_parameter,
                    train_data,
                    obs_q_list[i],
                    action_q_list[i],
                    reward_q_list[i],
                    delta_q_list[i],
                    value_q_list[i],
                    prob_q_list[i],
                )
                for i, (done, before_done) in enumerate(
                    zip(done_list, before_done_list)
                )
                if done != before_done
            ]

            for i in range(num_parallels):
                if done_list[i]:
                    # queのリセット
                    obs_q_list[i] = deque()
                    action_q_list[i] = deque()
                    reward_q_list[i] = deque()
                    value_q_list[i] = deque()
                    prob_q_list[i] = deque()
                    delta_q_list[i] = deque()
                    before_done_list[i] = False
                else:
                    before_done_list[i] = done_list[i]

            if len(train_data.obs_list) > self._ppo_parameter.num_sample_size:
                ppo_sample = PPOSample(
                    np.array(train_data.obs_list),
                    np.array(train_data.action_list),
                    np.array(train_data.gae_list),
                    np.array(train_data.v_list),
                    np.array(train_data.pi_list),
                )
                ppo_trainer.train(agent.model, ppo_sample)

                # trainに投げたデータ全削除
                reset_train_data(train_data)
            value_o_list = value_n_list
            reward_o_list = reward_list
            obs_list = next_obs_list
            step += 1
            # Save
            if (
                step % self._ppo_parameter.save_freq == 0
                and self._ppo_parameter.save_dir is not None
            ):
                save_dir = Path(self._ppo_parameter.save_dir).joinpath(str(step))
                self._agent.save(str(save_dir))
