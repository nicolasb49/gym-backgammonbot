import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium_backgammon.envs.backgammon import Backgammon as Game, WHITE, BLACK, COLORS
from random import randint
from gymnasium_backgammon.envs.rendering import Viewer
import numpy as np

STATE_W = 96
STATE_H = 96

SCREEN_W = 600
SCREEN_H = 500


class BackgammonEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "state_pixels"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: str | None = None):
        self.game = Game()
        self.current_agent = None
        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        ), f"Invalid render_mode {render_mode}"
        self.render_mode = render_mode

        low = np.zeros((198), dtype=np.float32)
        high = np.ones((198), dtype=np.float32)

        for i in range(3, 97, 4):
            high[i] = 6.0
        high[96] = 7.5

        for i in range(101, 195, 4):
            high[i] = 6.0
        high[194] = 7.5

        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.counter = 0
        self.max_length_episode = 10000
        self.viewer = None

        # Define a placeholder action space. Backgammon has a variable number of
        # legal actions per state so enumerating them upfront is infeasible.
        # We therefore define a large discrete action space that index-selects
        # among the legal moves returned by ``get_valid_actions``.
        # The chosen upper bound (1_000) is safely larger than the maximum
        # number of legal moves in backgammon positions.
        self.action_space = Discrete(1_000)

    def step(self, action):
        """Execute an *action* taken by the current agent and return the Gymnasium v0.29
        step tuple (observation, reward, terminated, truncated, info).

        The semantics are:
            terminated – The game finished naturally via one player bearing off all
                         checkers (``winner is not None``).
            truncated  – The episode exceeded ``max_length_episode``.

        The *info* dictionary always contains the keys:
            "winner" – ``WHITE``/``BLACK`` when *terminated* is True, otherwise None.
        """

        self.game.execute_play(self.current_agent, action)

        # Get the board representation from the opponent perspective (the
        # current player has already performed the move).
        observation = np.array(self.game.get_board_features(
            self.game.get_opponent(self.current_agent)
        ))

        reward = 0
        terminated = False
        truncated = False

        winner = self.game.get_winner()

        if winner is not None:
            # practical-issues-in-temporal-difference-learning, pag.3
            # …leading to a final reward signal z. In the simplest case,
            # z = 1 if White wins and z = 0 if Black wins
            if winner == WHITE:
                reward = 1
            terminated = True
        elif self.counter > self.max_length_episode:
            truncated = True

        self.counter += 1

        info = {"winner": winner if terminated else None}

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment and return *(observation, info)* as per the
        Gymnasium API.

        The *info* dict contains ``current_agent`` (whose turn it is first) and
        the initial dice ``roll`` that determines the starting player as in
        standard backgammon rules.
        """

        super().reset(seed=seed)

        # roll the dice
        roll = randint(1, 6), randint(1, 6)

        # roll the dice until they are different
        while roll[0] == roll[1]:
            roll = randint(1, 6), randint(1, 6)

        # set the current agent and orient dice from that player's point of view
        if roll[0] > roll[1]:
            self.current_agent = WHITE
            roll = (-roll[0], -roll[1])
        else:
            self.current_agent = BLACK

        self.game = Game()
        self.counter = 0

        observation = np.array(self.game.get_board_features(self.current_agent))
        info = {"current_agent": self.current_agent, "roll": roll}

        return observation, info

    def render(self, mode: str | None = None):
        """Render the environment.

        Gymnasium recommends configuring the render *mode* at environment
        creation time (`render_mode` argument of `__init__`).  To remain
        backwards-compatible with legacy code that passes a *mode* to `render`,
        we keep an optional *mode* argument.  If *mode* is not `None` then it
        overrides the instance `self.render_mode` for this call only.
        """

        if mode is None:
            mode = self.render_mode
        if mode is None:
            # Render was called but no mode specified; do nothing per API.
            return None

        if mode == "human":
            self.game.render()
            return True
        else:
            if self.viewer is None:
                self.viewer = Viewer(SCREEN_W, SCREEN_H)

            if mode == "rgb_array":
                width = SCREEN_W
                height = SCREEN_H
            elif mode == "state_pixels":
                width = STATE_W
                height = STATE_H
            else:
                raise ValueError(f"Unsupported render_mode {mode}")

            return self.viewer.render(
                board=self.game.board,
                bar=self.game.bar,
                off=self.game.off,
                state_w=width,
                state_h=height,
            )

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_valid_actions(self, roll):
        return self.game.get_valid_plays(self.current_agent, roll)

    def get_opponent_agent(self):
        self.current_agent = self.game.get_opponent(self.current_agent)
        return self.current_agent


class BackgammonEnvPixel(BackgammonEnv):

    def __init__(self, render_mode: str | None = None):
        # Force "state_pixels" render_mode if user passes None
        if render_mode is None:
            render_mode = "state_pixels"
        super().__init__(render_mode=render_mode)
        self.observation_space = Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        observation = self.render(mode="state_pixels")
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        observation = self.render(mode="state_pixels")
        return observation, info
