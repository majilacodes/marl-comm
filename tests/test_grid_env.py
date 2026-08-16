"""Tests for env.grid_env.GridEnv."""

import numpy as np
import pytest

from env.grid_env import GridEnv, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY


def test_reset_places_agents_within_bounds():
    env = GridEnv(grid_size=10, num_agents=4, seed=0)
    env.reset()
    for x, y in env.positions:
        assert 0 <= x < env.grid_size
        assert 0 <= y < env.grid_size


def test_reset_no_overlapping_start_positions():
    env = GridEnv(grid_size=10, num_agents=4, seed=1)
    env.reset()
    positions = [tuple(p) for p in env.positions]
    assert len(positions) == len(set(positions))


def test_step_respects_boundaries():
    env = GridEnv(grid_size=5, num_agents=1, seed=2)
    env.reset()
    env.positions[0] = np.array([0, 0])
    env.step([ACTION_LEFT])
    assert tuple(env.positions[0]) == (0, 0)

    env.positions[0] = np.array([0, 0])
    env.step([ACTION_UP])
    assert tuple(env.positions[0]) == (0, 0)

    env.positions[0] = np.array([4, 4])
    env.step([ACTION_RIGHT])
    assert tuple(env.positions[0]) == (4, 4)

    env.positions[0] = np.array([4, 4])
    env.step([ACTION_DOWN])
    assert tuple(env.positions[0]) == (4, 4)


def test_collision_detection_same_cell():
    env = GridEnv(grid_size=10, num_agents=2, seed=3)
    env.reset()
    env.positions[0] = np.array([2, 2])
    env.positions[1] = np.array([2, 4])
    env.history = [[(2, 2)], [(2, 4)]]
    _, _, _, info = env.step([ACTION_DOWN, ACTION_UP])
    assert info["collisions_this_step"] == 1
    assert tuple(env.positions[0]) == (2, 3)
    assert tuple(env.positions[1]) == (2, 3)


def test_collision_detection_swap():
    env = GridEnv(grid_size=10, num_agents=2, seed=4)
    env.reset()
    env.positions[0] = np.array([2, 2])
    env.positions[1] = np.array([3, 2])
    env.history = [[(2, 2)], [(3, 2)]]
    _, _, _, info = env.step([ACTION_RIGHT, ACTION_LEFT])
    assert info["collisions_this_step"] == 1


def test_goal_reached_flag():
    env = GridEnv(grid_size=10, num_agents=1, seed=5)
    env.reset()
    env.positions[0] = np.array([4, 4])
    env.goals[0] = np.array([4, 5])
    env.history = [[(4, 4)]]
    _, _, _, info = env.step([ACTION_DOWN])
    assert info["agents_at_goal"] == 1
    assert env.at_goal[0] is True


def test_episode_ends_at_max_steps():
    env = GridEnv(grid_size=10, num_agents=3, max_steps=5, seed=6)
    env.reset()
    done = False
    steps_taken = 0
    for _ in range(5):
        _, _, done, _ = env.step([ACTION_STAY] * 3)
        steps_taken += 1
        if done:
            break
    assert done is True
    assert steps_taken == 5
