import os
import argparse

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import optuna

from gymnasium_backgammon.wrappers.action_wrapper import BackgammonActionWrapper

# Register the Backgammon environment
try:
    register(
        id="gymnasium_backgammon:backgammon-v0",
        entry_point="gymnasium_backgammon.envs:BackgammonEnv",
        disable_env_checker=True,
    )
except Exception:
    pass


def make_env(seed: int = 0):
    """Utility function to create the Backgammon environment."""
    env = gym.make("gymnasium_backgammon:backgammon-v0")
    env = BackgammonActionWrapper(env)
    env.reset(seed=seed)
    return env


def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective for PPO hyperparameter search."""
    # sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_steps       = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    gae_lambda    = trial.suggest_float("gae_lambda", 0.8, 0.99)

    # single-process vectorized env for search
    train_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        gae_lambda=gae_lambda,
        verbose=0,
        tensorboard_log="logs",
    )
    model.learn(total_timesteps=200_000)

    # evaluate with episode rewards
    eval_env = make_env()
    rewards, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=5,
        return_episode_rewards=True
    )
    mean_reward = float(np.mean(rewards))

    train_env.close()
    eval_env.close()
    return mean_reward


def main():
    parser = argparse.ArgumentParser(description="Backgammon PPO training and Optuna search")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "optuna"],
        default="train",
        help="Mode: 'train' to train a model, 'optuna' to run hyperparameter search",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        help="Total timesteps to train (required in train mode)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for PPO (required in train mode)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for PPO (only in train mode)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps to run for each environment per update (only in train mode)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda parameter (only in train mode)",
    )
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
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=10,
        help="Number of Optuna trials (only in optuna mode)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        # ensure required args are provided
        if args.timesteps is None or args.learning_rate is None:
            parser.error("--timesteps and --learning-rate are required in train mode")

        seed = 0
        # build vectorized environment
        if args.vec_type == "subproc" and args.num_envs > 1:
            def make_env_rank(rank):
                def _init():
                    env = gym.make("gymnasium_backgammon:backgammon-v0")
                    env = BackgammonActionWrapper(env)
                    env.reset(seed=seed + rank)
                    return env
                return _init

            env = SubprocVecEnv([make_env_rank(i) for i in range(args.num_envs)])
        else:
            env = DummyVecEnv([
                lambda: BackgammonActionWrapper(
                    gym.make("gymnasium_backgammon:backgammon-v0")
                )
            ])

        # train the PPO model with CLI hyperparameters
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            gae_lambda=args.gae_lambda,
            verbose=1,
        )
        model.learn(total_timesteps=args.timesteps)

        os.makedirs("models", exist_ok=True)
        model.save("models/ppo_backgammon")
        print(f"Training complete. Model saved to models/ppo_backgammon.zip")
        return

    # Optuna hyperparameter search
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.optuna_trials)
    print("Best parameters found:", study.best_params)


if __name__ == "__main__":
    main()
