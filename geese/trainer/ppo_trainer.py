from typing import Tuple
import numpy as np
import tensorflow as tf

from geese.trainer.trainer import Trainer
from geese.structure.sample import PPOSample
from geese.structure.parameter import PPOTrainerParameter
from geese.constants import ACTIONLIST
from geese.util.converter import type32


class PPOTrainer(Trainer):
    def __init__(self, parameter: PPOTrainerParameter):
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=parameter.learning_rate)
        self._batch_size = parameter.batch_size
        self._num_epoch = parameter.num_epoch
        self._clip_eps = parameter.clip_eps
        self._entropy_coefficient = parameter.entropy_coefficient
        self._n_action = len(ACTIONLIST)

    def train(self, model: tf.keras.models.Model, sample: PPOSample) -> None:
        sample_size = len(sample)
        assert self._batch_size < sample_size

        for _ in range(self._num_epoch):
            idx = np.random.randint(sample_size, size=self._batch_size)

            tmp_args = [
                sample.observation,
                sample.action,
                sample.n_step_return,
                sample.v,
                sample.v_n,
                sample.pi
            ]

            def indexing(value: np.ndarray) -> np.ndarray:
                return value[idx]

            args = [model] + \
                list(map(type32, map(tf.convert_to_tensor, map(indexing, tmp_args))))
            self._train(*args)

    # @tf.function
    def _train(
        self,
        model: tf.keras.models.Model,
        observation: tf.Tensor,
        action: tf.Tensor,
        n_step_return: tf.Tensor,
        v_old: tf.Tensor,
        v_old_n: tf.Tensor,
        pi_old: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            # B, A and B
            pi_new, v_new = model(observation)
            # Policy Lossの計算
            # B, A
            advantage = n_step_return + v_old_n - v_old

            # B, A
            action = tf.one_hot(action, depth=self._n_action, dtype=tf.float32)

            # B
            policy_rate = tf.stop_gradient(tf.reduce_sum(
                action * pi_new, axis=-1)) / tf.reduce_sum(action * pi_old, axis=-1)
            # B
            clipped_advantage = tf.minimum(
                policy_rate,
                tf.clip_by_value(
                    policy_rate * advantage,
                    (1 - self._clip_eps) * advantage,
                    (1 + self._clip_eps) * advantage
                )
            )
            # B
            logit = tf.reduce_mean(tf.math.log(pi_new * action))

            # TFは勾配降下しかできないので、最大化したい目的関数の逆符号の最小化を行う
            loss_policy = -clipped_advantage * logit

            # Value Lossの計算
            loss_value = tf.reduce_mean(
                tf.keras.losses.MSE(advantage + v_old_n, v_new))

            # Entropy Lossの計算
            loss_entropy = -tf.reduce_mean(tf.reduce_sum(
                pi_new * tf.math.log(pi_new), axis=-1)) * self._entropy_coefficient

            loss_total = loss_policy + loss_value + loss_entropy

            # Apply Gradients
            gradient = tape.gradient(loss_total, model.trainable_variables)
            self._optimizer.apply_gradients(
                zip(gradient, model.trainable_variables))
        return loss_policy, loss_value, loss_entropy
