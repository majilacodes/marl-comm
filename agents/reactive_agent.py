"""Reactive baseline agent for the predictive-communication MARL project.

``ReactiveAgent`` is deliberately simple: it is a heuristic (non-learned)
controller that moves greedily toward its goal and decides whether to
communicate purely from *current* proximity to other agents. It exists to
(a) generate realistic multi-agent trajectories and communication events for
the baseline experiment, and (b) serve as the point of comparison for the
predictive (ARIMA-forecast-based) communication trigger built in a later
phase. The agent is intentionally decoupled from ``GridEnv`` — it only reads
observation dicts and returns actions/decisions — so that a
``PredictiveAgent`` can later be swapped in without changing the environment.
"""

import numpy as np

from env.grid_env import ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY

_MOVE_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)


class ReactiveAgent:
    """A heuristic agent that moves toward its goal and reacts to nearby agents.

    Attributes:
        agent_id (int): This agent's index within the environment.
        goal (tuple[int, int]): The (x, y) cell this agent is trying to reach.
        comm_radius (int): Manhattan-distance threshold used by the reactive
            communication rule.
        randomness (float): Probability in [0, 1] of taking a random valid
            move instead of the greedy move, used to avoid deadlocks where
            two agents repeatedly block each other.
    """

    def __init__(self, agent_id, goal, comm_radius=2, randomness=0.1, seed=None):
        """Initializes the agent.

        Args:
            agent_id: Index of this agent in the environment.
            goal: (x, y) target cell for this agent.
            comm_radius: Manhattan distance within which the reactive rule
                triggers communication.
            randomness: Chance of ignoring the greedy heuristic and picking
                a random move instead, to reduce the odds of two agents
                getting stuck oscillating against each other.
            seed: Optional seed for this agent's private RNG.
        """
        self.agent_id = agent_id
        self.goal = tuple(goal)
        self.comm_radius = comm_radius
        self.randomness = randomness
        self.rng = np.random.default_rng(seed)

    def act(self, position):
        """Chooses a movement action using a greedy Manhattan-distance heuristic.

        With probability ``randomness`` a uniformly random move action is
        taken instead, to help break deadlocks. Otherwise, the agent picks
        the move (among up/down/left/right/stay) that most reduces the
        Manhattan distance to its goal; if already at the goal, it stays.

        Args:
            position: (x, y) current position of this agent.

        Returns:
            int: One of the ACTION_* constants from ``env.grid_env``.
        """
        position = tuple(position)
        if position == self.goal:
            return ACTION_STAY

        if self.rng.random() < self.randomness:
            return int(self.rng.choice(_MOVE_ACTIONS))

        best_action = ACTION_STAY
        best_distance = self._manhattan_distance(position, self.goal)

        for action in _MOVE_ACTIONS:
            candidate = self._apply_action(position, action)
            distance = self._manhattan_distance(candidate, self.goal)
            if distance < best_distance:
                best_distance = distance
                best_action = action

        return best_action

    def should_communicate(self, own_position, other_positions):
        """Reactive communication rule: broadcast if anyone is currently within range.

        This computes proximity itself (rather than trusting a pre-filtered
        list) so the rule is self-contained, independently testable, and a
        direct point of comparison for the later predictive rule, which will
        have the same signature but check *forecasted* positions instead.

        Args:
            own_position: (x, y) current position of this agent.
            other_positions: dict or iterable of other agents' (x, y)
                current positions (unfiltered — may include agents far away).

        Returns:
            bool: True if any other agent is within ``comm_radius`` (Manhattan
            distance) of this agent right now.
        """
        positions = other_positions.values() if isinstance(other_positions, dict) else other_positions
        own_position = tuple(own_position)
        for pos in positions:
            if self._manhattan_distance(own_position, tuple(pos)) <= self.comm_radius:
                return True
        return False

    @staticmethod
    def _manhattan_distance(pos_a, pos_b):
        """Returns the Manhattan distance between two (x, y) positions."""
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

    @staticmethod
    def _apply_action(position, action):
        """Returns the position that would result from applying an action, unclamped."""
        deltas = {
            ACTION_UP: (0, -1),
            ACTION_DOWN: (0, 1),
            ACTION_LEFT: (-1, 0),
            ACTION_RIGHT: (1, 0),
            ACTION_STAY: (0, 0),
        }
        dx, dy = deltas[action]
        return position[0] + dx, position[1] + dy
