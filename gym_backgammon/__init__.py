from gymnasium.envs.registration import register

register(
    id="backgammon-v0",
    entry_point="gym_backgammon.envs:BackgammonEnv",
    disable_env_checker=True,
)

register(
    id='backgammon-pixel-v0',
    entry_point='gym_backgammon.envs:BackgammonEnvPixel',
    disable_env_checker=True,
)
