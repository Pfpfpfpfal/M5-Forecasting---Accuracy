from __future__ import annotations

from typing import Any, Protocol, cast


class Regressor(Protocol):
    def fit(self, x: Any, y: Any) -> Any: ...

    def predict(self, x: Any) -> Any: ...


def create_model(random_state: int = 42, **overrides: Any) -> Regressor:
    """Create the baseline regressor."""

    try:
        from lightgbm import LGBMRegressor

        params: dict[str, Any] = {
            "objective": "poisson",
            "n_estimators": 800,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "n_jobs": -1,
        }
        params.update(overrides)
        return cast_regressor(LGBMRegressor(**params))
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        params: dict[str, Any] = {
            "loss": "poisson",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 64,
            "random_state": random_state,
        }
        params.update(overrides)
        return cast_regressor(HistGradientBoostingRegressor(**params))


def cast_regressor(model: Any) -> Regressor:
    return cast(Regressor, model)
