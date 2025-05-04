from gymnasium.envs.registration import register

# Register environments under the new namespace.
register(
    id="backgammon-v0",
    entry_point="gymnasium_backgammon.envs:BackgammonEnv",
    disable_env_checker=True,
)

register(
    id="backgammon-pixel-v0",
    entry_point="gymnasium_backgammon.envs:BackgammonEnvPixel",
    disable_env_checker=True,
)
