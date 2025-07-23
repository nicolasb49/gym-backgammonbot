import argparse
import os

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gymnasium_backgammon.wrappers.action_wrapper import BackgammonActionWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--vec-type",
        choices=["dummy", "subproc"],
        default="dummy",
    )
    args = parser.parse_args()

    try:
        register(
            id="gymnasium_backgammon:backgammon-v0",
            entry_point="gymnasium_backgammon.envs:BackgammonEnv",
            disable_env_checker=True,
        )
    except Exception:
        # Environment might already be registered
        pass

    seed = 0

    def make_env(rank):
        def _init():
            env = gym.make("gymnasium_backgammon:backgammon-v0")
            env = BackgammonActionWrapper(env)
            env.seed(seed + rank)
            return env
        return _init

    VecEnvClass = SubprocVecEnv if args.vec_type == "subproc" else DummyVecEnv
    env = VecEnvClass([make_env(i) for i in range(args.num_envs)])

    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        tensorboard_log="logs",
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps)

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_backgammon")


if __name__ == "__main__":
    main()

