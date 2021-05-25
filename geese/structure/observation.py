from kaggle_environments.envs.hungry_geese.hungry_geese import Observation as Obs


class Observation:
    def __init__(self, now_obs: Obs, last_obs: Obs, index: int):
        self._now_obs = now_obs
        self._last_obs = last_obs
        self._index = index

    @property
    def now_obs(self):
        return self._now_obs

    @property
    def last_obs(self):
        return self._last_obs

    @property
    def index(self):
        return self._index
