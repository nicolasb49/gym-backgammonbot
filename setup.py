from setuptools import setup

# Gymnasium is the maintained fork of OpenAI Gym providing the latest RL
# environment API (terminated / truncated step API, fixed render modes, etc.).
# Replace the old `gym` dependency with `gymnasium` so that the package works
# with the current ecosystem.  Most users can install `gymnasium` with
# `pip install gymnasium`.

setup(
    name="gymnasium_backgammon",
    version="0.0.2",
    install_requires=[
        "gymnasium>=0.29.1",  # current stable gymnasium release
        "numpy>=1.19",
    ],
)