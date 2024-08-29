from abc import ABC, abstractmethod
from typing import Any, SupportsFloat


class ObservationRewardSystem(ABC):
    @abstractmethod
    def get_reward(self, obs: dict[str, Any]) -> float:
        pass


class SatiationRewardSystem(ObservationRewardSystem):
    def __init__(
        self,
        keys: list[str],
        reward_value: SupportsFloat = 0.1,
        max_sat_value: SupportsFloat = 1000,
        sat_reset_floor: SupportsFloat = 500,
    ) -> None:
        self._prev = {key: max_sat_value for key in keys}
        self._reward_value = reward_value
        self._sat_reset_floor = sat_reset_floor

    def get_reward(self, obs: dict[str, Any]) -> float:
        reward = sum(self._reward_value for key in self._prev if self._prev[key] < obs[key] < self._sat_reset_floor)
        self._prev = {key: obs[key] for key in self._prev}
        return reward
