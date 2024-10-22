import itertools
import os
import tempfile
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Iterator, Optional, SupportsFloat

import gymnasium
import minetest.minetest_env
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType
from minetest.minetest_env import KEY_MAP as KEYBOARD_ACTION_KEYS
from .observation_reward_system import ObservationRewardSystem, SatiationRewardSystem

# Keyboard actions
N_KEYBOARD_ACTIONS = len(KEYBOARD_ACTION_KEYS)
KEYBOARD_NOOP = np.zeros(N_KEYBOARD_ACTIONS)
MOVEMENT_KEYS = ["forward", "left", "backward", "right"]
BOAD_KEYBOARD_ACTION_KEYS = MOVEMENT_KEYS + ["dig"]


def get_keyboard_actions(key: str, idx: int) -> Iterator[np.ndarray]:
    action_value = deepcopy(KEYBOARD_NOOP)
    action_value[idx] = 1
    if key in MOVEMENT_KEYS:
        return tuple(
            itertools.chain(
                itertools.repeat(action_value, times=32),
                itertools.repeat(KEYBOARD_NOOP, times=16),
            )
        )
    return (action_value,)


KEYBOARD_ACTIONS = OrderedDict([(k, get_keyboard_actions(k, idx)) for idx, k in enumerate(KEYBOARD_ACTION_KEYS)])

# Mouse actions
MOUSE_NOOP = [0, 0]
MOUSE_SCALE = 64
MOUSE_ACTIONS = OrderedDict(
    [
        ("mouse_left", [MOUSE_SCALE, 0]),
        ("mouse_right", [-MOUSE_SCALE, 0]),
        ("mouse_up", [0, MOUSE_SCALE]),
        ("mouse_down", [0, -MOUSE_SCALE]),
    ]
)
N_MOUSE_ACTIONS = len(MOUSE_ACTIONS)
BOAD_MOUSE_ACTION_KEYS = ["mouse_left", "mouse_right", "mouse_up", "mouse_down"]


def get_mouse_actions(unit_action: list[int], n_steps: int) -> Iterator[list[int]]:
    cum_sweep = 0
    for step in range(1, n_steps + 1):
        next_cum_sweep = MOUSE_SCALE * (step // n_steps)
        diff = next_cum_sweep - cum_sweep
        yield [x * diff for x in unit_action]
        cum_sweep = next_cum_sweep
    yield from itertools.repeat(MOUSE_NOOP)


# actions
NOOP_ACTION = {
    "keys": (KEYBOARD_NOOP,),
    "mouse": MOUSE_NOOP,
}
ACTION_KEYS = KEYBOARD_ACTION_KEYS + list(MOUSE_ACTIONS.keys())
REPEATED_ACTION_KEYS = set(MOVEMENT_KEYS + list(MOUSE_ACTIONS.keys()))
REPEATED_ACTION_IDXS = set([i for i, v in enumerate(ACTION_KEYS) if v in REPEATED_ACTION_KEYS])


def get_action_dicts(
    keyboard_action_keys: list[str] = KEYBOARD_ACTION_KEYS, mouse_action_keys: list[str] = list(MOUSE_ACTIONS.keys())
) -> list[dict[str, Any]]:
    n_keyboard_actions = len(keyboard_action_keys)
    n_mouse_actions = len(mouse_action_keys)
    action_dicts = [deepcopy(NOOP_ACTION) for _ in range(n_keyboard_actions + n_mouse_actions)]

    for action_idx, keyboard_key in enumerate(keyboard_action_keys):
        action_dicts[action_idx]["keys"] = KEYBOARD_ACTIONS[keyboard_key]

    mouse_action_values = [v for k, v in MOUSE_ACTIONS.items() if k in mouse_action_keys]
    for action_idx, mouse_value in enumerate(mouse_action_values, start=n_keyboard_actions):
        action_dicts[action_idx]["mouse"] = mouse_value

    return action_dicts


def _get_boad_config(hunger_rate: int = 20, thirst_rate: int = 20, allow_night: bool = False, apple_scale: float = 2.5, rose_scale: float = 1.5, **kwargs) -> str:
    return f"""STARVE_1_MUL={hunger_rate}
STARVE_2_MUL={thirst_rate}
ALLOW_NIGHT={int(allow_night)}
APPLE_SCALE={apple_scale}
ROSE_SCALE={rose_scale}
"""


def _write_boad_config(config: dict[str, Any], game_dir: str) -> None:
    if config is None:
        config = {}
    boad_config = _get_boad_config(**config)
    config_path = os.path.join(game_dir, "config.lua")
    with open(config_path, "w") as f:
        f.write(boad_config)


BOAD_ADDITIONAL_OBSERVATION_SPACES = {
    "health": gymnasium.spaces.Box(0, 20, (1,), dtype=np.float32),
    "hunger": gymnasium.spaces.Box(0, 1000, (1,), dtype=np.float32),
    "thirst": gymnasium.spaces.Box(0, 1000, (1,), dtype=np.float32),
}


class MinetestGymnasium(gymnasium.Wrapper):
    def __init__(self, game: str, screen_size: int = 128, config: Optional[dict[str, Any]] = None):
        """Wrapper for the MineRL environments.

        Args:
            game (str): the minetest game to play.
            screen_size (int): the height of the pixels observations.
                Default to 128.
            config (dict): game specific configuration
                Default to None
        """

        self._observation_reward_systems: list[ObservationRewardSystem] = []

        temp_dir = tempfile.mkdtemp(prefix="minetest_")
        game_dir = os.path.join(os.environ["CONDA_PREFIX"], "share/minetest/games/", game)

        if game == "boad":
            _write_boad_config(config, game_dir)
            additional_observation_spaces = BOAD_ADDITIONAL_OBSERVATION_SPACES
            keyboard_action_keys = BOAD_KEYBOARD_ACTION_KEYS
            mouse_action_keys = BOAD_MOUSE_ACTION_KEYS
            self._observation_reward_systems.append(SatiationRewardSystem(["hunger", "thirst"], reward_value=0.1))
        else:
            additional_observation_spaces = {}
            keyboard_action_keys = KEYBOARD_ACTION_KEYS
            mouse_action_keys = list(MOUSE_ACTIONS.keys())

        env = minetest.minetest_env.MinetestEnv(
            display_size=(screen_size, screen_size),
            artifact_dir=os.path.join(temp_dir, "artifacts"),
            game_dir=game_dir,
            additional_observation_spaces=additional_observation_spaces,
            verbose_logging=True,
        )
        super().__init__(env)
        self._action_dicts = get_action_dicts(keyboard_action_keys, mouse_action_keys)
        self.action_space = gymnasium.spaces.Discrete(len(keyboard_action_keys) + len(mouse_action_keys))

    def step(self, action_idx: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_dict = self._action_dicts[action_idx]
        cum_reward = 0
        for keyboard_action in action_dict["keys"]:
            action = {"keys": keyboard_action, "mouse": action_dict["mouse"]}
            obs, reward, terminated, truncated, info = self.env.step(action)
            cum_reward += reward
            cum_reward += sum(ors.get_reward(obs) for ors in self._observation_reward_systems)
            if terminated or truncated:
                break
        return obs, cum_reward, terminated, truncated, info

    @property
    def display_size(self):
        return self.env.display_size
