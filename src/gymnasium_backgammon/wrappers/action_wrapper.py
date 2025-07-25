import gymnasium as gym
from random import randint
from gymnasium_backgammon.envs.backgammon import WHITE


def _unwrap_to_attr(env, attr: str):
    """Return the first environment in the wrapper chain that defines ``attr``."""
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        if hasattr(current, attr):
            return current
        visited.add(id(current))
        current = getattr(current, "env", getattr(current, "unwrapped", None))
    return current

class BackgammonActionWrapper(gym.ActionWrapper):
    """Wrap BackgammonEnv to map discrete indices to legal actions."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # keep the underlying discrete action space
        self.action_space = env.action_space
        self.roll = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        # store the roll for the first move
        self.roll = info.get("roll")
        # expose the current roll in info so agents can know the dice
        info["roll"] = self.roll
        return observation, info

    def step(self, action):
        # find the underlying env that exposes helper methods
        base_env = _unwrap_to_attr(self.env, "get_valid_actions")

        # find all valid moves for the current roll
        valid_moves = list(base_env.get_valid_actions(self.roll))
        # convert the discrete index into a specific play
        if 0 <= action < len(valid_moves):
            selected_action = valid_moves[action]
        else:
            selected_action = None

        obs, reward, terminated, truncated, info = self.env.step(selected_action)

        # prepare next step if the episode has not finished
        if not (terminated or truncated):
            # change player turn inside base env
            current = base_env.get_opponent_agent()
            # roll dice for next player
            dice = (randint(1, 6), randint(1, 6))
            if current == WHITE:
                dice = (-dice[0], -dice[1])
            self.roll = dice
        else:
            self.roll = None

        info["roll"] = self.roll
        return obs, reward, terminated, truncated, info
