from conf.parameter import (
    NUM_PARALLELS,
    NUM_STEP,
    MIN_SAMPLE_SIZE,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCH,
    CLIP_EPS,
    GAMMA,
    LAMBDA,
    ENTROPY_COEFFICIENT,
    REWARD_FUNC,
    REWARD_LIST,
    NUM_LAYERS,
    NUM_FILTERS,
    KERNEL_SIZE,
    BATCH_NORMALIZATION,
    USE_GPU,
    SAVE_FREQ,
    AGAINST_GREEDY,
    REWARD_LOG_FREQ,
)
import os

if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from geese.controller.ppo_controller import PPOController
from geese.controller.ppo_solo_controller import PPOSoloController
from geese.constants import NO_GPU_MSG, RewardFunc, SAVE_DIR
from geese.structure.parameter import (
    AgentParameter,
    EnvParameter,
    BaseModelParameter,
    PPOTrainerParameter,
    PPOParameter,
)
from geese.agent.model import BaseModel


if __name__ == "__main__":
    if USE_GPU:
        import tensorflow as tf

        assert tf.test.is_gpu_available(), NO_GPU_MSG

    model = BaseModel(
        BaseModelParameter(
            num_layers=NUM_LAYERS,
            num_filters=NUM_FILTERS,
            kernel_size=KERNEL_SIZE,
            bn=BATCH_NORMALIZATION,
            use_gpu=USE_GPU,
        )
    )

    agent_parameter = AgentParameter(model=model)
    trainer_parameter = PPOTrainerParameter(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epoch=NUM_EPOCH,
        clip_eps=CLIP_EPS,
        entropy_coefficient=ENTROPY_COEFFICIENT,
    )

    if REWARD_FUNC == "RAW":
        reward_func = RewardFunc.RAW
    elif REWARD_FUNC == "RANK":
        reward_func = RewardFunc.RANK
    else:
        raise ValueError("Unexpected Reward Function")

    env_parameter = EnvParameter(reward_func=reward_func, reward_list=REWARD_LIST)

    parameter = PPOParameter(
        num_parallels=NUM_PARALLELS,
        num_step=NUM_STEP,
        gamma=GAMMA,
        param_lambda=LAMBDA,
        num_sample_size=MIN_SAMPLE_SIZE,
        save_freq=SAVE_FREQ,
        save_dir=SAVE_DIR,
        reward_log_freq=REWARD_LOG_FREQ,
        env_parameter=env_parameter,
        ppo_trainer_parameter=trainer_parameter,
        agent_parameter=agent_parameter,
    )

    if AGAINST_GREEDY:
        controller = PPOSoloController(parameter)
    else:
        controller = PPOController(parameter)
    controller.train()
