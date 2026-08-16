"""ARIMA-based trajectory forecaster (proof of concept).

This module is the seed of the project's core novelty: predicting where an
agent will be a few steps in the future, so that a later "predictive"
communication rule can trigger *before* two agents actually get close,
rather than reacting once they already are (see ``agents.reactive_agent``).

For this phase, the forecaster is deliberately standalone: it is not wired
into the live agent decision loop. It only needs to prove that, given a
logged (x, y) trajectory, ARIMA can be fit per-coordinate and produce a
sensible short-horizon forecast.

ARIMA (AutoRegressive Integrated Moving Average) is inherently a univariate
model, so a 2D trajectory is decomposed into two independent 1D series (the
x-coordinate over time and the y-coordinate over time) and a separate ARIMA
model is fit to each.
"""

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Below this many observations, ARIMA fits are unreliable or fail outright
# (too few points to estimate AR/MA/differencing terms), so we fall back to
# a naive "last known position" prediction instead of crashing.
MIN_HISTORY_FOR_ARIMA = 10

DEFAULT_ORDER = (1, 1, 0)


class ArimaTrajectoryForecaster:
    """Forecasts an agent's future (x, y) position from its position history.

    Attributes:
        order (tuple[int, int, int]): The (p, d, q) ARIMA order applied to
            each coordinate series. Defaults to (1, 1, 0): a first-difference
            with one autoregressive term, which is a reasonable, cheap
            default for short, roughly linear agent trajectories.
    """

    def __init__(self, order=DEFAULT_ORDER):
        """Initializes the forecaster.

        Args:
            order: (p, d, q) order passed to ``statsmodels`` ARIMA for both
                the x- and y-coordinate series.
        """
        self.order = order

    def forecast(self, history, n_steps=3):
        """Forecasts an agent's position ``n_steps`` into the future.

        Args:
            history: List of (x, y) tuples representing the agent's
                position at each past timestep, in chronological order.
            n_steps: Number of future steps to forecast.

        Returns:
            list[tuple[float, float]]: Exactly ``n_steps`` predicted (x, y)
            positions, one per future timestep. If there is insufficient
            history to fit ARIMA reliably (fewer than
            ``MIN_HISTORY_FOR_ARIMA`` points) or fitting fails for any
            reason, this falls back to repeating the last known position
            ``n_steps`` times rather than raising.
        """
        if history is None or len(history) < MIN_HISTORY_FOR_ARIMA:
            return self._fallback(history, n_steps)

        xs = np.array([p[0] for p in history], dtype=float)
        ys = np.array([p[1] for p in history], dtype=float)

        try:
            x_forecast = self._forecast_series(xs, n_steps)
            y_forecast = self._forecast_series(ys, n_steps)
        except Exception:
            return self._fallback(history, n_steps)

        return list(zip(x_forecast.tolist(), y_forecast.tolist()))

    def _forecast_series(self, series, n_steps):
        """Fits ARIMA to a single 1D series and forecasts ``n_steps`` ahead.

        Args:
            series: 1D numpy array of past values.
            n_steps: Number of future values to forecast.

        Returns:
            np.ndarray: Length-``n_steps`` array of forecasted values.
        """
        if np.all(series == series[0]):
            return np.full(n_steps, series[0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(series, order=self.order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=n_steps)
        return np.asarray(forecast)

    @staticmethod
    def _fallback(history, n_steps):
        """Returns the last known position repeated ``n_steps`` times.

        Used when history is too short or too sparse for ARIMA to fit
        reliably. If history is empty entirely, falls back to the origin.

        Args:
            history: List of (x, y) tuples, possibly short or empty.
            n_steps: Number of future steps to "forecast".

        Returns:
            list[tuple[float, float]]: The last known (x, y) position,
            repeated ``n_steps`` times.
        """
        last = tuple(history[-1]) if history else (0.0, 0.0)
        return [last] * n_steps


if __name__ == "__main__":
    # Demo: synthetic trajectory moving diagonally at a roughly constant
    # velocity (with a little noise), forecast ahead, and compare against
    # the true continuation of the pattern.
    rng = np.random.default_rng(42)
    steps = 20
    true_traj = [(1.0 * t, 0.5 * t) for t in range(steps)]
    noisy_traj = [
        (x + rng.normal(0, 0.05), y + rng.normal(0, 0.05)) for x, y in true_traj
    ]

    forecaster = ArimaTrajectoryForecaster()
    n_steps = 3
    predicted = forecaster.forecast(noisy_traj, n_steps=n_steps)
    actual_future = [(1.0 * t, 0.5 * t) for t in range(steps, steps + n_steps)]

    print("Logged trajectory (last 5 points):", [tuple(round(v, 2) for v in p) for p in noisy_traj[-5:]])
    print()
    for i in range(n_steps):
        px, py = predicted[i]
        ax, ay = actual_future[i]
        print(f"Step +{i + 1}: predicted=({px:.2f}, {py:.2f})  actual_if_pattern_continued=({ax:.2f}, {ay:.2f})")
