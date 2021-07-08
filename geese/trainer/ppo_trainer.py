from typing import Tuple
import numpy as np
import tensorflow as tf

from geese.trainer.trainer import Trainer
from geese.structure.sample import PPOSample
from geese.structure.parameter import PPOTrainerParameter
from geese.constants import ACTIONLIST
from geese.util.converter import type32
from geese.util.tensor_boad_logger import TensorBoardLogger

EPS = 1e-9


class PPOTrainer(Trainer):
    def __init__(self, parameter: PPOTrainerParameter, logger: TensorBoardLogger):
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=parameter.learning_rate
        )
        self._batch_size = parameter.batch_size
        self._num_epoch = parameter.num_epoch
        self._clip_eps = parameter.clip_eps
        self._entropy_coefficient = parameter.entropy_coefficient
        self._n_action = len(ACTIONLIST)
        self._logger = logger

    def train(self, model: tf.keras.models.Model, sample: PPOSample) -> None:
        sample_size = len(sample)
        assert self._batch_size < sample_size

        for _ in range(self._num_epoch):
            idx = np.random.randint(sample_size, size=self._batch_size)

            tmp_args = [
                sample.observation,
                sample.action,
                sample.gae,
                sample.v,
                sample.pi,
            ]

            def indexing(value: np.ndarray) -> np.ndarray:
                return value[idx]

            args = [model] + list(
                map(type32, map(tf.convert_to_tensor, map(indexing, tmp_args)))
            )
            loss_policy, loss_value, loss_entropy, entropy, v_targ = self._train(*args)
            self._logger.logging_scaler("loss_policy", loss_policy)
            self._logger.logging_scaler("loss_value", loss_value)
            self._logger.logging_scaler("loss_entropy", loss_entropy)
            self._logger.logging_scaler("entropy", entropy)
            self._logger.logging_scaler("v_targ", v_targ)

    @tf.function
    def _train(
        self,
        model: tf.keras.models.Model,
        observation: tf.Tensor,
        action: tf.Tensor,
        advantage: tf.Tensor,
        v_old: tf.Tensor,
        pi_old: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            # B, A and B
            pi_new, v_new = model(observation)
            # Policy Lossの計算
            # B, A
            action = tf.one_hot(action, depth=self._n_action, dtype=tf.float32)

            # B
            policy_rate = tf.reduce_sum(action * pi_new, axis=-1) / tf.reduce_sum(
                action * pi_old, axis=-1
            )
            # B
            clipped_advantage = tf.minimum(
                policy_rate * advantage,
                tf.clip_by_value(
                    policy_rate, (1 - self._clip_eps), (1 + self._clip_eps)
                )
                * advantage,
            )
            # TFは勾配降下しかできないので、最大化したい目的関数の逆符号の最小化を行う
            loss_policy = -tf.reduce_mean(clipped_advantage)

            # Valueの教師信号の計算
            v_targ = advantage + v_old

            # Value Lossの計算
            loss_value = tf.reduce_mean(tf.keras.losses.MSE(v_targ, v_new))

            # Entropyの計算
            entropy = -tf.reduce_mean(
                tf.reduce_sum(pi_new * tf.math.log(pi_new + EPS), axis=-1)
            )
            # Entropy Lossの計算
            loss_entropy = -entropy * self._entropy_coefficient

            loss_total = loss_policy + loss_value + loss_entropy

            # Apply Gradients
            gradient = tape.gradient(loss_total, model.trainable_variables)
            self._optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        return loss_policy, loss_value, loss_entropy, entropy, tf.reduce_mean(v_targ)
