from gymnasium.envs.registration import register

from importlib import import_module

# Import environment classes to expose at package top level

BackgammonEnv = import_module("gym_backgammon.envs.backgammon_env").BackgammonEnv
BackgammonEnvPixel = import_module("gym_backgammon.envs.backgammon_env").BackgammonEnvPixel

__all__ = ["BackgammonEnv", "BackgammonEnvPixel"]

# Register environments under the new namespace.
register(
    id="gymnasium_backgammon/backgammon-v0",
    entry_point="gymnasium_backgammon.envs:BackgammonEnv",
    disable_env_checker=True,
)

register(
    id="gymnasium_backgammon/backgammon-pixel-v0",
    entry_point="gymnasium_backgammon.envs:BackgammonEnvPixel",
    disable_env_checker=True,
)
