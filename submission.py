from geese.structure.parameter.agent_parameter import AgentParameter
from geese.structure.parameter.model_parameter import BaseModelParameter
from geese.agent.model.model import BaseModel
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation as KaggleObservation
from geese.agent import PPOAgent
from conf.parameter import (
    NUM_LAYERS,
    NUM_FILTERS,
    KERNEL_SIZE,
    BATCH_NORMALIZATION,
    USE_GPU,
)
model = BaseModel(BaseModelParameter(
    num_layers=NUM_LAYERS,
    num_filters=NUM_FILTERS,
    kernel_size=KERNEL_SIZE,
    bn=BATCH_NORMALIZATION,
    use_gpu=USE_GPU
))

agent_parameter = AgentParameter(model=model)
agent = PPOAgent(agent_parameter)


def our_agent(obs: KaggleObservation):
    return agent(obs)


env = make('hungry_geese')
env.reset()
env.run([our_agent, "greedy", "greedy", "greedy"])
env.render(mode="ipython")
