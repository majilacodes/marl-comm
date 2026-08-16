"""Runs the reactive-baseline simulation and logs collision/communication metrics.

This script wires together ``GridEnv`` and ``ReactiveAgent`` to produce the
first empirical results for the project: how many collisions occur and how
much communication is triggered when agents use a purely reactive
(current-proximity-based) communication rule. These numbers are the
baseline that the later predictive (ARIMA-forecast-based) rule will be
compared against.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.grid_env import GridEnv
from agents.reactive_agent import ReactiveAgent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_baseline(grid_size=10, num_agents=4, comm_radius=2, max_steps=100, seed=0):
    """Runs one full episode of the reactive baseline and collects metrics.

    Args:
        grid_size: Size of one side of the square grid.
        num_agents: Number of agents to simulate.
        comm_radius: Manhattan-distance communication range.
        max_steps: Episode horizon.
        seed: Random seed for environment and agent reproducibility.

    Returns:
        dict: Logged per-step metrics and summary statistics:
            - "collisions_per_step": list[int]
            - "comm_events_per_step": list[int]
            - "cumulative_comm_events": list[int]
            - "agents_at_goal_per_step": list[int]
            - "total_collisions": int
            - "total_comm_events": int
            - "steps_run": int
            - "avg_steps_to_goal": float or None
    """
    env = GridEnv(grid_size=grid_size, num_agents=num_agents, comm_radius=comm_radius,
                  max_steps=max_steps, seed=seed)
    observations = env.reset()

    agents = [
        ReactiveAgent(agent_id=i, goal=env.goals[i], comm_radius=comm_radius, randomness=0.1, seed=seed + i)
        for i in range(num_agents)
    ]

    goal_reached_step = [None] * num_agents

    collisions_per_step = []
    comm_events_per_step = []
    agents_at_goal_per_step = []

    done = False
    step = 0
    while not done:
        actions = [agents[i].act(observations[i]["position"]) for i in range(num_agents)]

        comm_events = 0
        for i in range(num_agents):
            others = {j: pos for j, pos in observations[i]["visible_agents"].items()}
            if agents[i].should_communicate(observations[i]["position"], others):
                comm_events += 1

        observations, rewards, done, info = env.step(actions)
        step += 1

        for i in range(num_agents):
            if env.at_goal[i] and goal_reached_step[i] is None:
                goal_reached_step[i] = step

        collisions_per_step.append(info["collisions_this_step"])
        comm_events_per_step.append(comm_events)
        agents_at_goal_per_step.append(info["agents_at_goal"])

    cumulative_comm_events = []
    running = 0
    for c in comm_events_per_step:
        running += c
        cumulative_comm_events.append(running)

    reached = [s for s in goal_reached_step if s is not None]
    avg_steps_to_goal = sum(reached) / len(reached) if reached else None

    return {
        "collisions_per_step": collisions_per_step,
        "comm_events_per_step": comm_events_per_step,
        "cumulative_comm_events": cumulative_comm_events,
        "agents_at_goal_per_step": agents_at_goal_per_step,
        "total_collisions": sum(collisions_per_step),
        "total_comm_events": sum(comm_events_per_step),
        "steps_run": step,
        "avg_steps_to_goal": avg_steps_to_goal,
        "num_agents": num_agents,
    }


def save_plots(metrics, out_dir=RESULTS_DIR):
    """Saves the collisions-per-step and cumulative-communication plots.

    Args:
        metrics: The dict returned by ``run_baseline``.
        out_dir: Directory to write PNG files into (created if missing).
    """
    os.makedirs(out_dir, exist_ok=True)

    steps = list(range(1, metrics["steps_run"] + 1))

    plt.figure(figsize=(8, 4))
    plt.plot(steps, metrics["collisions_per_step"], color="crimson")
    plt.xlabel("Step")
    plt.ylabel("Collisions this step")
    plt.title("Reactive Baseline: Collisions per Step")
    plt.tight_layout()
    collisions_path = os.path.join(out_dir, "collisions_per_step.png")
    plt.savefig(collisions_path)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(steps, metrics["cumulative_comm_events"], color="steelblue")
    plt.xlabel("Step")
    plt.ylabel("Cumulative communication events")
    plt.title("Reactive Baseline: Cumulative Communication Events")
    plt.tight_layout()
    comm_path = os.path.join(out_dir, "cumulative_comm_events.png")
    plt.savefig(comm_path)
    plt.close()

    return collisions_path, comm_path


def main():
    """Runs the baseline simulation once, prints a summary, and saves plots."""
    metrics = run_baseline()

    print("=== Reactive Baseline Summary ===")
    print(f"Steps run: {metrics['steps_run']}")
    print(f"Total collisions: {metrics['total_collisions']}")
    print(f"Total communication events: {metrics['total_comm_events']}")
    if metrics["avg_steps_to_goal"] is not None:
        print(f"Average steps to goal (agents that reached it): {metrics['avg_steps_to_goal']:.2f}")
    else:
        print("Average steps to goal: no agents reached their goal")
    print(f"Agents at goal (final step): {metrics['agents_at_goal_per_step'][-1]} / {metrics['num_agents']}")

    collisions_path, comm_path = save_plots(metrics)
    print()
    print(f"Saved plot: {collisions_path}")
    print(f"Saved plot: {comm_path}")


if __name__ == "__main__":
    main()
