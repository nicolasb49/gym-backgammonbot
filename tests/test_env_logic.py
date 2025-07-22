import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from gymnasium_backgammon.envs.backgammon_env import BackgammonEnv, WHITE, BLACK
from gymnasium_backgammon.envs.backgammon import BAR


def orient_roll(player, base=(1, 2)):
    """Return roll oriented for the given player."""
    return tuple(-r for r in base) if player == WHITE else base


@pytest.fixture(params=[WHITE, BLACK])
def player(request):
    return request.param


@pytest.fixture
def env():
    return BackgammonEnv()


def test_get_valid_actions_start(env, player):
    env.reset()
    env.current_agent = player
    roll = orient_roll(player)
    actions = env.get_valid_actions(roll)
    assert len(actions) == 18


def test_get_valid_actions_bar_and_bearoff(env, player):
    env.reset()
    # Move one checker to the bar
    if player == WHITE:
        src = 23
        env.game.board[src] = (env.game.board[src][0] - 1, WHITE)
    else:
        src = 0
        env.game.board[src] = (env.game.board[src][0] - 1, BLACK)
    env.game.bar[player] = 1
    env.game.players_positions = env.game.get_players_positions()
    env.current_agent = player
    roll = orient_roll(player)
    actions = env.get_valid_actions(roll)
    assert any(move[0] == BAR for action in actions for move in action)

    # Bear-off scenario: one checker left on board
    env.reset()
    for i in range(24):
        env.game.board[i] = (0, None)
    if player == WHITE:
        env.game.board[0] = (1, WHITE)
    else:
        env.game.board[23] = (1, BLACK)
    env.game.off[player] = 14
    env.game.players_positions = env.game.get_players_positions()
    env.current_agent = player
    actions = env.get_valid_actions(roll)
    assert any(move[1] < 0 or move[1] >= 24 for action in actions for move in action)


def test_execute_play_consistency(env, player):
    env.reset()
    env.current_agent = player
    roll = orient_roll(player)
    actions = env.get_valid_actions(roll)
    action = sorted(actions)[0]

    before_white = sum(c for c, p in env.game.board if p == WHITE) + env.game.bar[WHITE] + env.game.off[WHITE]
    before_black = sum(c for c, p in env.game.board if p == BLACK) + env.game.bar[BLACK] + env.game.off[BLACK]

    env.game.execute_play(player, action)

    after_white = sum(c for c, p in env.game.board if p == WHITE) + env.game.bar[WHITE] + env.game.off[WHITE]
    after_black = sum(c for c, p in env.game.board if p == BLACK) + env.game.bar[BLACK] + env.game.off[BLACK]

    assert before_white == after_white == 15
    assert before_black == after_black == 15


def test_reset_returns_observation_and_info(env):
    obs, info = env.reset()
    assert len(obs) == 198
    assert "current_agent" in info and "roll" in info
