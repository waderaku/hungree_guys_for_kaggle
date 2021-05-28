from conf.parameter import (
    NUM_PARALLELS,
    NUM_STEP,
    MIN_SAMPLE_SIZE,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCH,
    CLIP_EPS,
    GAMMA,
    ENTROPY_COEFFICIENT,
    REWARD_LIST,
    NUM_LAYERS,
    NUM_FILTERS,
    KERNEL_SIZE,
    BATCH_NORMALIZATION,
    USE_GPU,
)
import os
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICE"] = "-1"
from geese.controller.ppo_controller import PPOController
from geese.structure.parameter import (
    AgentParameter,
    EnvParameter,
    BaseModelParameter,
    PPOTrainerParameter,
    PPOParameter
)
from geese.agent.model import BaseModel


if __name__ == "__main__":
    model = BaseModel(BaseModelParameter(
        num_layers=NUM_LAYERS,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        bn=BATCH_NORMALIZATION,
    ))

    agent_parameter = AgentParameter(model=model)
    trainer_parameter = PPOTrainerParameter(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epoch=NUM_EPOCH,
        clip_eps=CLIP_EPS,
        entropy_coefficient=ENTROPY_COEFFICIENT
    )
    env_parameter = EnvParameter(reward_list=REWARD_LIST)

    parameter = PPOParameter(
        num_parallels=NUM_PARALLELS,
        num_step=NUM_STEP,
        gamma=GAMMA,
        num_sample_size=MIN_SAMPLE_SIZE,
        env_parameter=env_parameter,
        ppo_trainer_parameter=trainer_parameter,
        agent_parameter=agent_parameter
    )

    controller = PPOController(parameter)
    controller.train()
