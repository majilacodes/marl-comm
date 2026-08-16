# Predictive State-Aware Communication Triggering in Multi-Agent Reinforcement Learning

Course project for BCSE334L (Predictive Analytics), DA1 — 30% implementation milestone.

## Problem

Existing communication-aware MARL systems trigger inter-agent communication
*reactively*: only after an agent's current state (e.g. proximity to
another agent) crosses a threshold. This project explores a *predictive*
alternative: each agent forecasts its own short-horizon future trajectory
(via ARIMA) and triggers communication when the *forecast* indicates an
impending collision — earlier than a reactive system would react.

## Scope of this submission (Phase 1 / DA1)

This milestone delivers:
1. A fully working multi-agent grid environment (`env/grid_env.py`) with
   collision detection (same-cell and swap), goal tracking, and per-agent
   position history logging.
2. A heuristic reactive baseline agent (`agents/reactive_agent.py`) that
   moves toward its goal and communicates using a *current*-proximity rule.
3. A standalone, tested proof-of-concept ARIMA trajectory forecaster
   (`forecasting/arima_forecaster.py`) that fits per-coordinate ARIMA models
   to a logged trajectory and forecasts N steps ahead.
4. An experiment runner (`experiments/run_baseline.py`) that simulates the
   reactive baseline end-to-end and produces the first metrics/plots.

**Not yet done (explicitly out of scope for this milestone):** the ARIMA
forecaster is not wired into the live agent decision loop, and there is no
predictive communication rule yet. See "Next phase" below.

## Project structure

```
predictive-marl-comm/
├── env/grid_env.py            # multi-agent grid environment
├── agents/reactive_agent.py   # heuristic agent + reactive comm rule
├── forecasting/arima_forecaster.py  # ARIMA forecaster (proof of concept)
├── experiments/run_baseline.py      # runs the reactive baseline, logs/plots metrics
├── tests/                     # pytest suite (14 tests, all passing)
├── results/                   # saved plots from run_baseline.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# Run the full reactive-baseline simulation and save plots to results/
python -m experiments.run_baseline

# Run the ARIMA forecaster demo (synthetic trajectory, standalone)
python -m forecasting.arima_forecaster

# Run the test suite
pytest tests/ -v
```

## Environment design

`GridEnv` is a plain Python class (no PettingZoo/Gymnasium) exposing
`reset()` / `step(actions)` / `render()`. Each agent gets an observation
containing its own position and the positions of any other agents
currently within `comm_radius`. Rewards are a small per-step penalty, a
collision penalty, and a one-time goal-reached bonus. Full position
history is retained per agent — this is the raw input the ARIMA forecaster
will later consume online.

## Reactive baseline

`ReactiveAgent` picks the move that most reduces Manhattan distance to its
goal (with a small random-move probability to avoid deadlocks), and
communicates whenever another agent is currently within `comm_radius`. It
has no dependency on `GridEnv` internals, so a `PredictiveAgent` can later
be substituted with the same interface (`act`, `should_communicate`).

## ARIMA forecaster

`ArimaTrajectoryForecaster` fits an independent ARIMA(1,1,0) model to the
x- and y-coordinate series of an agent's history and forecasts N steps
ahead. With fewer than 10 historical points, or if fitting fails, it falls
back to returning the last known position instead of crashing. Verified on
synthetic linear and constant trajectories (see `tests/test_arima_forecaster.py`
and the `__main__` demo).

## Test coverage

14 tests across the three modules (7 environment, 3 agent, 4 forecaster),
all passing — see the tests directory for exact scenarios covered
(boundary clamping, same-cell and swap collisions, goal detection, episode
termination, reactive communication radius, greedy movement, ARIMA forecast
shape/accuracy/fallback/constant-trajectory behavior).

## Next phase (not in this submission)

- Wire `ArimaTrajectoryForecaster` into the live agent decision loop so
  each agent forecasts its own position every step.
- Build a `PredictiveAgent` whose `should_communicate` compares *forecasted*
  positions (both agents' forecasts) against `comm_radius`, instead of
  current positions.
- Run controlled comparison experiments: predictive vs. reactive agents on
  the same seeds/scenarios, comparing total collisions and total
  communication volume (testing the hypothesis that predictive triggering
  reduces one or both).
- Tune ARIMA order / forecast horizon N and evaluate forecast accuracy
  against actual agent trajectories logged from the baseline runs.
