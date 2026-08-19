"""
    python -m src.train --data-dir data --model-path artifacts/baseline.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import joblib  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.data_utils import (
    LAGS,
    ROLLING_WINDOWS,
    build_feature_frame,
    feature_columns,
    get_day_columns,
    read_m5_data,
)
from src.models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M5 baseline model.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sales-file", default="sales_train_validation.csv")
    parser.add_argument("--model-path", default="artifacts/baseline.joblib")
    parser.add_argument("--last-n-days", type=int, default=365)
    parser.add_argument("--valid-days", type=int, default=28)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    clipped_pred = np.clip(y_pred_arr, 0, None)
    return {
        "rmse": float(np.sqrt(np.mean((y_true_arr - clipped_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true_arr - clipped_pred))),
        "rmsle": float(np.sqrt(np.mean((np.log1p(y_true_arr) - np.log1p(clipped_pred)) ** 2))),
    }


def main() -> None:
    args = parse_args()
    sales, calendar, prices = read_m5_data(args.data_dir, args.sales_file)

    max_day = max(int(col[2:]) for col in get_day_columns(sales))
    start_day = max(1, max_day - args.last_n_days + 1)
    # Лаги считаются внутри выбранного окна, поэтому берём запас дней на "разогрев".
    warmup_days = max(LAGS) + max(ROLLING_WINDOWS)

    frame, encoders = build_feature_frame(
        sales=sales,
        calendar=calendar,
        prices=prices,
        start_day=max(1, start_day - warmup_days),
        end_day=max_day,
    )
    required_lags = [f"lag_{lag}" for lag in LAGS] + [f"rolling_mean_{window}" for window in ROLLING_WINDOWS]
    frame = frame.dropna(subset=["sales", *required_lags])
    frame = frame.loc[frame["d_num"] >= start_day]
    features = feature_columns(frame)

    valid_from = max_day - args.valid_days + 1
    train_frame: pd.DataFrame = frame.loc[frame["d_num"] < valid_from].copy()
    valid_frame: pd.DataFrame = frame.loc[frame["d_num"] >= valid_from].copy()

    if train_frame.empty:
        raise SystemExit(
            f"Обучающая выборка пуста: --last-n-days={args.last_n_days} <= --valid-days={args.valid_days}."
        )

    model = create_model(random_state=args.random_state)
    model.fit(train_frame[features], train_frame["sales"])

    metrics: dict[str, float] = {}
    if len(valid_frame):
        y_true: NDArray[np.float64] = np.asarray(valid_frame["sales"].to_numpy(), dtype=np.float64)
        y_pred = np.asarray(model.predict(valid_frame[features]))
        metrics = regression_metrics(y_true, y_pred)
        print("Validation metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.5f}")

    bundle: dict[str, Any] = {
        "model": model,
        "features": features,
        "encoders": encoders,
        "sales_file": args.sales_file,
        "last_train_day": max_day,
        "last_n_days": args.last_n_days,
        "valid_days": args.valid_days,
        "metrics": metrics,
    }

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    cast(Any, joblib).dump(bundle, model_path)
    print(f"Saved baseline bundle to {model_path}")


if __name__ == "__main__":
    main()
