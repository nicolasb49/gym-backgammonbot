import os
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from gymnasium_backgammon.wrappers.action_wrapper import BackgammonActionWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_backgammon.zip",
        help="Path to the trained PPO model",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval.csv",
        help="CSV file to write mean/std results",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Create and wrap environment
    env = gym.make("gymnasium_backgammon:backgammon-v0", render_mode=None)
    env = BackgammonActionWrapper(env)
    env = Monitor(env)  # ensures accurate episode lengths & rewards

    # Load the trained model
    model = PPO.load(args.model_path)

    # Evaluate
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.n_episodes,
        return_episode_rewards=True
    )
    mean_reward = float(np.mean(episode_rewards))
    std_reward  = float(np.std(episode_rewards))

    print(f"Evaluated over {args.n_episodes} episodes")
    print(f"Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mean_reward", "std_reward"])
        # seed is hardcoded 0 here; you could add a --seed arg if you like
        writer.writerow([0, mean_reward, std_reward])

    # Plot per-episode rewards
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards during Evaluation")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
