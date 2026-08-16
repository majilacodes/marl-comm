"""Custom multi-agent grid environment for collision-avoidance experiments.

This module implements ``GridEnv``, a lightweight, dependency-free (beyond
numpy) 2D grid world used to study communication-triggering strategies in
multi-agent reinforcement learning. It intentionally avoids frameworks like
PettingZoo or Gymnasium so that the simulation loop, observation format, and
collision semantics are fully transparent and easy to extend later (e.g. to
plug in a predictive, ARIMA-based communication trigger).
"""

import numpy as np

# Action encoding used throughout the environment and agents.
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4

# Maps an action index to a (dx, dy) displacement on the grid.
ACTION_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
    ACTION_STAY: (0, 0),
}

STEP_PENALTY = -0.1
COLLISION_PENALTY = -10.0
GOAL_REWARD = 20.0


class GridEnv:
    """A discrete 2D grid world shared by multiple agents.

    Each agent occupies a cell and must navigate to its own goal cell.
    Agents can optionally communicate with each other; communication does
    not affect the physics of the environment (movement/collisions) but is
    tracked so that different triggering strategies (reactive vs. later
    predictive) can be compared on communication volume and effectiveness.

    Attributes:
        grid_size (int): Width and height of the square grid.
        num_agents (int): Number of agents in the environment.
        comm_radius (int): Manhattan-distance radius within which two
            agents are considered "in range" for communication.
        max_steps (int): Episode horizon; the episode ends after this many
            steps even if agents have not reached their goals.
        positions (np.ndarray): Shape (num_agents, 2) array of current
            (x, y) agent positions.
        goals (np.ndarray): Shape (num_agents, 2) array of goal (x, y)
            positions, fixed for the duration of an episode.
        history (list[list[tuple[int, int]]]): Per-agent list of
            (x, y) positions visited so far this episode, in order. This is
            the raw trajectory data the ARIMA forecaster will later consume.
    """

    def __init__(self, grid_size=10, num_agents=4, comm_radius=2, max_steps=100, seed=None):
        """Initializes environment configuration.

        Args:
            grid_size: Size of one side of the square grid.
            num_agents: Number of agents to simulate.
            comm_radius: Manhattan distance defining the communication range.
            max_steps: Maximum number of steps per episode.
            seed: Optional random seed for reproducible agent/goal placement.
        """
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.comm_radius = comm_radius
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.positions = None
        self.goals = None
        self.history = None
        self.at_goal = None
        self.step_count = 0

    def reset(self):
        """Randomly places agents and goals on distinct cells and resets state.

        Agent starting positions are drawn without replacement so no two
        agents begin on the same cell. Goal positions are drawn
        independently (also without replacement among themselves) and may
        coincide with another agent's start cell — that is not a collision,
        only same-timestep co-occupation of two agents counts as one.

        Returns:
            list[dict]: One observation dict per agent, each containing:
                - "position": (x, y) tuple, the agent's own position.
                - "visible_agents": dict mapping other agent index -> (x, y)
                  position, for every other agent currently within
                  ``comm_radius`` of this agent (available for the reactive
                  communication rule to consult).
        """
        cells = self.rng.choice(self.grid_size * self.grid_size, size=self.num_agents * 2, replace=False)
        agent_cells = cells[: self.num_agents]
        goal_cells = cells[self.num_agents:]

        self.positions = np.array(
            [(int(c % self.grid_size), int(c // self.grid_size)) for c in agent_cells], dtype=int
        )
        self.goals = np.array(
            [(int(c % self.grid_size), int(c // self.grid_size)) for c in goal_cells], dtype=int
        )

        self.at_goal = [False] * self.num_agents
        self.step_count = 0
        self.history = [[tuple(pos)] for pos in self.positions]

        return self._build_observations()

    def _build_observations(self):
        """Constructs the per-agent observation list from current state."""
        observations = []
        for i in range(self.num_agents):
            visible_agents = {}
            for j in range(self.num_agents):
                if i == j:
                    continue
                dist = self._manhattan_distance(self.positions[i], self.positions[j])
                if dist <= self.comm_radius:
                    visible_agents[j] = tuple(self.positions[j])
            observations.append({
                "position": tuple(self.positions[i]),
                "visible_agents": visible_agents,
            })
        return observations

    @staticmethod
    def _manhattan_distance(pos_a, pos_b):
        """Returns the Manhattan distance between two (x, y) positions."""
        return abs(int(pos_a[0]) - int(pos_b[0])) + abs(int(pos_a[1]) - int(pos_b[1]))

    def _clamp_move(self, pos, action):
        """Applies an action to a position, clamping the result to the grid."""
        dx, dy = ACTION_DELTAS[action]
        x = min(max(pos[0] + dx, 0), self.grid_size - 1)
        y = min(max(pos[1] + dy, 0), self.grid_size - 1)
        return x, y

    def step(self, actions):
        """Advances the environment by one timestep.

        Args:
            actions: Sequence (list/tuple/dict) of length ``num_agents``
                giving each agent's chosen action (see the ACTION_* / STAY
                constants). If a dict, keys must be agent indices.

        Returns:
            tuple: (observations, rewards, done, info) where
                - observations: list[dict], see ``reset``.
                - rewards: list[float], one scalar reward per agent.
                - done: bool, True if the episode has ended (max_steps
                  reached or all agents at their goals).
                - info: dict with "collisions_this_step" (int) and
                  "agents_at_goal" (int).
        """
        if isinstance(actions, dict):
            actions = [actions[i] for i in range(self.num_agents)]

        proposed = [self._clamp_move(self.positions[i], actions[i]) for i in range(self.num_agents)]
        old_positions = [tuple(p) for p in self.positions]

        colliding, num_incidents = self._detect_collisions(old_positions, proposed)

        rewards = [STEP_PENALTY] * self.num_agents
        for i in range(self.num_agents):
            self.positions[i] = proposed[i]
            self.history[i].append(tuple(self.positions[i]))
            if i in colliding:
                rewards[i] += COLLISION_PENALTY
            if tuple(self.positions[i]) == tuple(self.goals[i]):
                if not self.at_goal[i]:
                    rewards[i] += GOAL_REWARD
                self.at_goal[i] = True

        self.step_count += 1

        agents_at_goal = sum(self.at_goal)
        done = self.step_count >= self.max_steps or agents_at_goal == self.num_agents

        info = {
            "collisions_this_step": num_incidents,
            "agents_at_goal": agents_at_goal,
        }

        observations = self._build_observations()
        return observations, rewards, done, info

    def _detect_collisions(self, old_positions, new_positions):
        """Identifies which agents are involved in a collision this step.

        Two kinds of collisions are detected:
            1. Same-cell occupation: two or more agents end up on the same
               cell after moving.
            2. Direct swaps: agent A moves into agent B's old cell while B
               simultaneously moves into A's old cell.

        Args:
            old_positions: list of (x, y) tuples before the move.
            new_positions: list of (x, y) tuples after the move.

        Returns:
            tuple[set[int], int]: The set of agent indices involved in at
            least one collision (used to apply the per-agent collision
            penalty), and the number of distinct collision incidents (each
            same-cell pile-up or each swap pair counts as exactly one
            incident, regardless of how many agents it involves).
        """
        colliding = set()
        num_incidents = 0
        n = self.num_agents

        cell_occupants = {}
        for i in range(n):
            cell_occupants.setdefault(new_positions[i], []).append(i)
        for occupants in cell_occupants.values():
            if len(occupants) > 1:
                colliding.update(occupants)
                num_incidents += 1

        for i in range(n):
            for j in range(i + 1, n):
                if new_positions[i] == old_positions[j] and new_positions[j] == old_positions[i] \
                        and old_positions[i] != old_positions[j]:
                    colliding.add(i)
                    colliding.add(j)
                    num_incidents += 1

        return colliding, num_incidents

    def render(self):
        """Prints a simple text representation of the grid to stdout.

        Agents are drawn as their index number, goals as 'G', and empty
        cells as '.'. If an agent occupies its own goal cell, the agent
        index is shown (agent takes visual precedence). This is intended
        for debugging only, not for the report's figures.
        """
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for i in range(self.num_agents):
            gx, gy = self.goals[i]
            if grid[gy][gx] == '.':
                grid[gy][gx] = 'G'

        for i in range(self.num_agents):
            x, y = self.positions[i]
            grid[y][x] = str(i)

        lines = [' '.join(row) for row in grid]
        print('\n'.join(lines))
        print()
