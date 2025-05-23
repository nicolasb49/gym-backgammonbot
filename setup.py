from setuptools import setup, find_packages

setup(
    name="backgammonbot",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "torch",
        "stable-baselines3",
        "pyglet",
    ],
    entry_points={
        "gymnasium.envs": [
            "backgammon-v0 = gymnasium_backgammon.envs.backgammon_env:BackgammonEnv",
        ]
    },
)
