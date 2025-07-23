import os
import argparse

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import optuna

from gymnasium_backgammon.wrappers.action_wrapper import BackgammonActionWrapper


try:
    register(
        id="gymnasium_backgammon:backgammon-v0",
        entry_point="gymnasium_backgammon.envs:BackgammonEnv",
        disable_env_checker=True,
    )
except Exception:
    # Environment might already be registered
    pass


def make_env(seed: int = 0):
    """Utility function to create the Backgammon environment."""
    env = gym.make("gymnasium_backgammon:backgammon-v0")
    env = BackgammonActionWrapper(env)
    env.reset(seed=seed)
    return env


def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective for PPO hyperparameter search."""

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    train_env = DummyVecEnv([make_env])

    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        verbose=0,
        tensorboard_log="logs",
    )

    model.learn(total_timesteps=200_000)

    eval_env = make_env()
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)

    # Clean up environments
    train_env.close()
    eval_env.close()

    return mean_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backgammon PPO hyperparameter search")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments to use",
    )
    parser.add_argument(
        "--vec-type",
        type=str,
        choices=["dummy", "subproc"],
        default="dummy",
        help="Type of vectorized environment",
    )

    args = parser.parse_args()

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    seed = 0

    if args.vec_type == "subproc" and args.num_envs > 1:
        def make_env(rank):
            def _init():
                env = gym.make("gymnasium_backgammon:backgammon-v0")
                env = BackgammonActionWrapper(env)
                env.seed(seed + rank)
                return env
            return _init

        env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])
    else:
        env = DummyVecEnv([
            lambda: BackgammonActionWrapper(
                gym.make("gymnasium_backgammon:backgammon-v0")
            )
        ])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best params:", study.best_params)

