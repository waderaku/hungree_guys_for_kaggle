import tensorflow as tf

from geese.trainer.trainer import Trainer
from geese.trainer.minibatch.ppo_minibatch import PPOMiniBatch
from geese.trainer.parameter import PPOTrainerParameter


class PPOTrainer(Trainer):
    def __init__(self, parameter: PPOTrainerParameter):
        self._clip_eps = parameter.clip_eps
        self._optimizer = tf.keras.optimizers.Adam(
            learning_rate=parameter.learning_rate)

    def train(self, model: tf.keras.models.Model, minibatch: PPOMiniBatch) -> None:
        pass

    @tf.function
    def _train(
        self,
        model: tf.keras.models.Model,
        observation: tf.Tensor,
        action: tf.Tensor,
        n_step_return: tf.Tensor,
        v_old: tf.Tensor,
        v_old_n: tf.Tensor,
        pi_old: tf.Tensor
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            # B, A and B
            pi_new, v_new = model(observation)
            # Compute Policy Loss
            # B, A
            advantage = n_step_return + v_old_n - v_old
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

            # Compute Value Loss
            loss_value = tf.reduce_mean(
                tf.keras.losses.MSE(advantage + v_old_n, v_new))

            # Compute Entropy Loss
            loss_entropy = tf.reduce_mean(tf.reduce_sum(
                pi_new * tf.math.log(pi_new), axis=-1))

            loss_total = loss_policy + loss_value + loss_entropy
            gradient = tape.gradient(loss_total, model.trainable_variables)
            self._optimizer.apply_gradients(
                zip(gradient, model.trainable_variables))
        return loss_total
