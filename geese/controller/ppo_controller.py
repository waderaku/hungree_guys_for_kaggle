
from geese.structure.train_data import TrainData
import numpy as np
from geese.util.converter import action2int
from geese.trainer.ppo_trainer import PPOTrainer
from geese.controller.ppo_helper import add_to_que, calc_n_step_return, create_padding_data, \
    create_que_list, reset_que, reset_train_data, update_PPO_list
from geese.structure.parameter.ppo_parameter import PPOParameter
from geese.structure.sample import PPOSample
from geese.env.vecenv.vecenv import VecEnv
from geese.agent.model.model import BaseModel
from geese.agent.ppo_agent import PPOAgent
from geese.constants import NUM_GEESE


class PPOController():
    def __init__(self, ppo_parameter: PPOParameter):
        self._ppo_parameter = ppo_parameter
        self._agent = PPOAgent(ppo_parameter.agent_parameter)

    def train(self) -> None:
        train_data = TrainData([], [], [], [], [], [])
        ppo_trainer = PPOTrainer(self._ppo_parameter.ppo_trainer_parameter)
        vec_env = VecEnv(self._ppo_parameter.num_parallels,
                         self._ppo_parameter.env_parameter)
        agent = self._agent

        obs_list = vec_env.reset()

        obs_q_list = create_que_list(
            self._ppo_parameter.num_parallels, NUM_GEESE)
        reward_q_list = create_que_list(
            self._ppo_parameter.num_parallels, NUM_GEESE)
        action_q_list = create_que_list(
            self._ppo_parameter.num_parallels, NUM_GEESE)
        value_q_list = create_que_list(
            self._ppo_parameter.num_parallels, NUM_GEESE)
        prob_q_list = create_que_list(
            self._ppo_parameter.num_parallels, NUM_GEESE)

        step = 0
        before_done_list = [[False]*NUM_GEESE] * \
            self._ppo_parameter.num_parallels

        while True:
            action_list, value_n_list, prob_list =\
                tuple(zip(*[agent.step(obs) for obs in obs_list]))

            next_obs_list, reward_list, done_list = vec_env.step(action_list)

            game_done_list = [sum(done) == NUM_GEESE for done in done_list]

            for i, (reward_q, value_q) in enumerate(zip(reward_q_list, value_q_list)):
                if len(reward_q[0]) == self._ppo_parameter.num_step:
                    n_step_return_list = calc_n_step_return(
                        reward_q, self._ppo_parameter.gamma)
                    [r_q.popleft() for r_q in reward_q]
                    o = [o_q.popleft() for o_q in obs_q_list[i]]
                    a = [action2int(a_q.popleft()) for a_q in action_q_list[i]]
                    v = [v_q.popleft() for v_q in value_q]
                    p = [p_q.popleft() for p_q in prob_q_list[i]]

                    update_PPO_list(
                        train_data,
                        o,
                        a,
                        n_step_return_list,
                        v,
                        value_n_list[i],
                        p,
                        before_done_list[i]
                    )

            # n回分の行動をキューで管理
            add_to_que(obs_q_list, obs_list)
            add_to_que(action_q_list, action_list)
            add_to_que(value_q_list, value_n_list)
            add_to_que(reward_q_list, reward_list)
            add_to_que(prob_q_list, prob_list)

            for i, (reward_q, value_q) in enumerate(zip(reward_q_list, value_q_list)):
                # 今回終了したアクションに対してパディングを行ってデータを格納する
                [
                    create_padding_data(
                        self._ppo_parameter,
                        train_data,
                        obs_q_list[i][j],
                        action_q_list[i][j],
                        reward_q_list[i][j],
                        value_q_list[i][j],
                        prob_q_list[i][j]
                    )
                    for j, (done, before_done) in enumerate(zip(done_list[i], before_done_list[i]))
                    if done != before_done
                ]

                if game_done_list[i]:
                    # queのリセット
                    obs_q_list[i] = reset_que(NUM_GEESE)
                    action_q_list[i] = reset_que(NUM_GEESE)
                    reward_q_list[i] = reset_que(NUM_GEESE)
                    value_q_list[i] = reset_que(NUM_GEESE)
                    prob_q_list[i] = reset_que(NUM_GEESE)
                    before_done_list[i] = [False]*NUM_GEESE
                else:
                    before_done_list[i] = done_list[i]

            if len(train_data.obs_list) > self._ppo_parameter.num_sample_size:
                ppo_sample = PPOSample(
                    np.array(train_data.obs_list),
                    np.array(train_data.action_list),
                    np.array(train_data.n_step_return_list),
                    np.array(train_data.v_list),
                    np.array(train_data.v_n_list),
                    np.array(train_data.pi_list)
                )
                ppo_trainer.train(agent.model, ppo_sample)

                # trainに投げたデータ全削除
                reset_train_data(self._ppo_parameter)

            obs_list = next_obs_list
            step += 1
