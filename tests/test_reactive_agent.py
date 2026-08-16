"""Tests for agents.reactive_agent.ReactiveAgent."""

from agents.reactive_agent import ReactiveAgent
from env.grid_env import ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY


def test_communicates_when_within_radius():
    agent = ReactiveAgent(agent_id=0, goal=(9, 9), comm_radius=2, randomness=0.0)
    own_position = (5, 5)
    other_positions = {1: (6, 5)}  # distance 1 <= comm_radius 2
    assert agent.should_communicate(own_position, other_positions) is True


def test_does_not_communicate_when_far():
    agent = ReactiveAgent(agent_id=0, goal=(9, 9), comm_radius=2, randomness=0.0)
    own_position = (0, 0)
    other_positions = {1: (9, 9)}  # far away
    assert agent.should_communicate(own_position, other_positions) is False


def test_action_moves_toward_goal():
    # Agent directly above its goal (same x, smaller y): moving DOWN reduces distance.
    agent = ReactiveAgent(agent_id=0, goal=(5, 5), comm_radius=2, randomness=0.0, seed=0)
    position = (5, 3)
    action = agent.act(position)
    assert action == ACTION_DOWN

    before = abs(position[0] - 5) + abs(position[1] - 5)
    deltas = {
        ACTION_UP: (0, -1), ACTION_DOWN: (0, 1),
        ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0), ACTION_STAY: (0, 0),
    }
    dx, dy = deltas[action]
    new_pos = (position[0] + dx, position[1] + dy)
    after = abs(new_pos[0] - 5) + abs(new_pos[1] - 5)
    assert after < before
