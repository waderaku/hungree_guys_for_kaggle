from kaggle_environments.envs.hungry_geese.hungry_geese import Action

# Field Size
FIELD_HEIGHT = 7
FIELD_WIDTH = 11

NUM_CHANNELS = 17

# ActionList
ACTIONLIST = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]

NUM_GEESE = 4

TIME_LIMIT = 1.0

NO_GPU_MSG = "GPU is not available."


# log directory
LOG_BASE_DIR = "logs"


from dataclasses import dataclass, asdict


@dataclass
class Configuration:
    act_timeout: int
    columns: int
    episode_steps: int
    hunger_rate: int
    max_length: int
    min_food: int
    rows: int
    run_timeout: int
    episodeSteps: int
    actTimeout: int
    runTimeout: int
    _is_protocol: bool


act_timeout = 1
columns = 11
episode_steps = 200
hunger_rate = 40
max_length = 99
min_food = 2
rows = 7
run_timeout = 1200
_is_protocol = False
episodeSteps = 200
actTimeout = 1
runTimeout = 1200

CONFIGURATION = asdict(
    Configuration(
        act_timeout,
        columns,
        hunger_rate,
        max_length,
        min_food,
        rows,
        episodeSteps,
        actTimeout,
        run_timeout,
        _is_protocol,
    )
)
