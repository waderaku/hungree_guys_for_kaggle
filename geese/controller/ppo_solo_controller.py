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
        # ロギングファイル生成
        today = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        logger = TensorBoardLogger(f"{LOG_BASE_DIR}/{today}")

        # 並列ユーザー数
        num_parallels = self._ppo_parameter.num_parallels

        # 一連のオブジェクトのインスタンス化
        train_data = TrainData()
        ppo_trainer = PPOTrainer(self._ppo_parameter.ppo_trainer_parameter, logger)
        vec_solo_env = VecSoloEnv(
            self._ppo_parameter.num_parallels, self._ppo_parameter.env_parameter
        )
        agent = self._agent

        # すべてのゲームを初期化する
        obs_list = vec_solo_env.reset()

        # 学習用データの初期化
        obs_q_list = reset_que(self._ppo_parameter.num_parallels)
        reward_q_list = reset_que(self._ppo_parameter.num_parallels)
        action_q_list = reset_que(self._ppo_parameter.num_parallels)
        value_q_list = reset_que(self._ppo_parameter.num_parallels)
        prob_q_list = reset_que(self._ppo_parameter.num_parallels)
        delta_q_list = reset_que(self._ppo_parameter.num_parallels)
        step = 1
        before_game_done_list = [True] * self._ppo_parameter.num_parallels
        value_o_list = list()
        reward_o_list = list()

        # ロギングデータ初期化
        reward_log_list = list()
        episodelen_log_list = list()

        while True:
            # 今回の行動をエージェントが決定する
            action_list, value_n_list, prob_list = agent.step(
                obs_list, True, before_game_done_list
            )

            # 上で決定した行動をとる
            next_obs_list, reward_list, done_list = vec_solo_env.step(action_list)

            # 前回ゲームが終了していない場合（＝まだゲームが続いている場合）、δの計算を行う
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

            # num_step回行動をとった場合、
            # GAEが計算できるため、そこから学習データを追加していく
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
                )
                for i in range(num_parallels)
                if len(reward_q_list[i]) == self._ppo_parameter.num_step
            ]

            # n回分の行動をキューで管理
            # 今回の情報をキューに追加
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
                for i, done in enumerate(done_list)
                if done
            ]

            for i in range(num_parallels):
                # ゲームが終了している場合、リセット処理を行う
                if done_list[i]:
                    # 今回のリワードをlogに追加
                    reward_log_list.append(reward_q_list[i][-1])
                    episodelen_log_list.append(len(reward_q_list[i]))

                    # queのリセット
                    obs_q_list[i] = deque()
                    action_q_list[i] = deque()
                    reward_q_list[i] = deque()
                    value_q_list[i] = deque()
                    prob_q_list[i] = deque()
                    delta_q_list[i] = deque()

            # 一定数以上トレーニングデータがたまったら学習を行う
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

            # 次ステップへ進む
            before_game_done_list = done_list
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

            # reward log
            if len(reward_log_list) >= self._ppo_parameter.reward_log_freq:
                logger.logging_scaler(
                    "reward", sum(reward_log_list) / len(reward_log_list)
                )
                logger.logging_scaler(
                    "episode_length",
                    sum(episodelen_log_list) / len(episodelen_log_list),
                )
                reward_log_list = []
