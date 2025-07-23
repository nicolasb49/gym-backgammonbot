import os
import csv
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium_backgammon.wrappers.action_wrapper import BackgammonActionWrapper


def main() -> None:
    seed = 0
    env = gym.make("gymnasium_backgammon:backgammon-v0")
    env = BackgammonActionWrapper(env)
    env.reset(seed=seed)

    model = PPO.load("models/ppo_backgammon")

    mean_reward, std_reward, episode_rewards, _ = evaluate_policy(
        model, env, n_eval_episodes=100, return_episode_rewards=True
    )

    print(f"mean_reward={mean_reward}, std_reward={std_reward}")

    os.makedirs("results", exist_ok=True)
    with open("results/eval.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mean_reward", "std_reward"])
        writer.writerow([seed, mean_reward, std_reward])

    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward during Evaluation")
    plt.show()


if __name__ == "__main__":
    main()
