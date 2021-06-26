from geese.controller.cartpole_controller import CartPoleController
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
    USE_GPU,
    SAVE_FREQ,
    SAVE_DIR,
)
import os

if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from geese.constants import NO_GPU_MSG, RewardFunc
from geese.structure.parameter import (
    AgentParameter,
    EnvParameter,
    CPModelParameter,
    PPOTrainerParameter,
    PPOParameter,
)
from geese.agent.model import CPModel

NUM_ACTION = 2


if __name__ == "__main__":
    if USE_GPU:
        import tensorflow as tf

        assert tf.test.is_gpu_available(), NO_GPU_MSG

    model = CPModel(CPModelParameter(128, 4))

    agent_parameter = AgentParameter(model=model)
    trainer_parameter = PPOTrainerParameter(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epoch=NUM_EPOCH,
        clip_eps=CLIP_EPS,
        entropy_coefficient=ENTROPY_COEFFICIENT,
        num_action=NUM_ACTION,
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
        env_parameter=env_parameter,
        ppo_trainer_parameter=trainer_parameter,
        agent_parameter=agent_parameter,
    )

    controller = CartPoleController(parameter)
    controller.train()
