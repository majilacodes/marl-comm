"""Tests for forecasting.arima_forecaster.ArimaTrajectoryForecaster."""

from forecasting.arima_forecaster import ArimaTrajectoryForecaster


def test_forecast_returns_correct_shape():
    history = [(float(t), 0.0) for t in range(20)]
    forecaster = ArimaTrajectoryForecaster()
    result = forecaster.forecast(history, n_steps=3)
    assert len(result) == 3
    for point in result:
        assert len(point) == 2


def test_forecast_reasonable_on_linear_trajectory():
    history = [(float(t), 0.0) for t in range(20)]
    forecaster = ArimaTrajectoryForecaster()
    result = forecaster.forecast(history, n_steps=1)
    predicted_x, predicted_y = result[0]
    true_next_x = 20.0
    assert abs(predicted_x - true_next_x) <= 1.5
    assert abs(predicted_y - 0.0) <= 1.5


def test_handles_short_history_gracefully():
    history = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]  # fewer than 10 points
    forecaster = ArimaTrajectoryForecaster()
    result = forecaster.forecast(history, n_steps=3)
    assert len(result) == 3
    for point in result:
        assert point == (2.0, 2.0)


def test_handles_constant_trajectory():
    history = [(5.0, 5.0) for _ in range(15)]
    forecaster = ArimaTrajectoryForecaster()
    result = forecaster.forecast(history, n_steps=3)
    assert len(result) == 3
    for px, py in result:
        assert abs(px - 5.0) < 1e-6
        assert abs(py - 5.0) < 1e-6
