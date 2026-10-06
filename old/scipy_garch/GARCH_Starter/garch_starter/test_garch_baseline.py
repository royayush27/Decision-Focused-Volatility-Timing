"""Focused checks: no future input leakage, aggregation, and baseline alignment."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from garch_baseline import (Config, load_data, history_at_origin, forecast_one,
                            variance_path, walk_forward, score_forecasts)


class ForecastChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent
        cls.daily, cls.monthly = load_data(root / "datasets/market_daily.csv",
                                           root / "datasets/market_monthly.csv")

    def test_future_perturbations_do_not_change_forecast(self):
        cutoff = pd.Timestamp("2000-01-01")
        modified = self.daily.copy()
        modified.loc[modified.index >= cutoff, "ret"] = 0.5
        a = history_at_origin(self.daily, "2000-01")
        b = history_at_origin(modified, "2000-01")
        pd.testing.assert_series_equal(a, b)
        original, _ = forecast_one(a)
        changed, _ = forecast_one(b)
        self.assertEqual(original["garch_rv_forecast"], changed["garch_rv_forecast"])

    def test_monthly_sum_and_closed_form(self):
        history = history_at_origin(self.daily, "2000-01")
        row, fitted = forecast_one(history)
        path = variance_path(fitted, 21)
        self.assertAlmostEqual(row["garch_rv_forecast"], path.sum(), places=14)
        p, w, first = row["persistence"], row["omega_decimal"], row["next_day_variance"]
        if abs(1-p) > 1e-6:
            longrun = w/(1-p)
            closed = 21*longrun+(first-longrun)*(1-p**21)/(1-p)
            np.testing.assert_allclose(closed, path.sum(), rtol=1e-9)

    def test_baseline_alignment_and_failure_on_missing_month(self):
        cfg = Config(last_forecast_month="2000-02")
        out = walk_forward(self.daily, self.monthly, cfg)
        self.assertEqual(out.loc["2000-01", "origin_date"], "1999-12-31")
        self.assertEqual(out.loc["2000-02", "origin_date"], "2000-01-31")
        np.testing.assert_allclose(out.rv_baseline_forecast, out.rv_lag1)
        with self.assertRaises(ValueError):
            walk_forward(self.daily, self.monthly.drop(pd.Period("2000-01")), cfg)

    def test_perfect_forecasts_have_zero_loss(self):
        x = pd.DataFrame({"rv": [.01, .02], "rv_baseline_forecast": [.01, .02],
                          "garch_rv_forecast": [.01, .02]})
        scores = score_forecasts(x)
        np.testing.assert_allclose(scores[["QLIKE", "MSE"]], 0.)


if __name__ == "__main__":
    unittest.main()
