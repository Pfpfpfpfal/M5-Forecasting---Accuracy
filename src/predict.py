"""
    python -m src.predict --data-dir data --model-path artifacts/baseline.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Protocol, cast

import joblib  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.data_utils import (
    ID_COLS,
    add_calendar_price_features,
    add_lag_features,
    apply_category_encoders,
    get_day_columns,
    read_m5_data,
    reduce_memory,
)


class PredictModel(Protocol):
    def predict(self, x: Any) -> Any: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict M5 baseline submission.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sales-file", default="sales_train_evaluation.csv")
    parser.add_argument("--model-path", default="artifacts/baseline.joblib")
    parser.add_argument("--output-path", default="submission.csv")
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--history-days", type=int, default=90)
    return parser.parse_args()


def make_history(sales: pd.DataFrame, history_days: int) -> pd.DataFrame:
    max_day = max(int(col[2:]) for col in get_day_columns(sales))
    start_day = max(1, max_day - history_days + 1)
    day_cols = [col for col in get_day_columns(sales) if int(col[2:]) >= start_day]
    history = sales.melt(id_vars=cast(Any, ID_COLS), value_vars=cast(Any, day_cols), var_name="d", value_name="sales")
    history["d_num"] = history["d"].astype(str).str[2:].astype("int16")
    return reduce_memory(history)


def forecast_horizon(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    model: PredictModel,
    features: list[str],
    encoders: dict[str, dict[str, int]],
    horizon: int,
    history_days: int,
) -> pd.DataFrame:
    max_day = max(int(col[2:]) for col in get_day_columns(sales))
    frame: pd.DataFrame = make_history(sales, history_days)
    base_ids: pd.DataFrame = sales.loc[:, ID_COLS].copy()
    predictions: list[NDArray[np.float64]] = []

    for step in range(1, horizon + 1):
        d_num = max_day + step
        future: pd.DataFrame = base_ids.copy()
        future["d"] = f"d_{d_num}"
        future["d_num"] = d_num
        future["sales"] = np.nan

        scoring: pd.DataFrame = pd.concat([frame, future], ignore_index=True)
        scoring = add_calendar_price_features(scoring, calendar, prices)
        scoring = add_lag_features(scoring)
        scoring = apply_category_encoders(scoring, encoders)
        current: pd.DataFrame = scoring.loc[scoring["d_num"].eq(d_num)].copy()

        pred = cast(NDArray[np.float64], np.clip(np.asarray(model.predict(current[features]), dtype=float), 0, None))
        future["sales"] = pred
        frame = pd.concat([frame, future], ignore_index=True)
        predictions.append(pred)

    forecast_cols = [f"F{i}" for i in range(1, horizon + 1)]
    pred_matrix: NDArray[np.float64] = np.vstack(predictions)
    pred_df = pd.DataFrame(pred_matrix.T, columns=forecast_cols)
    pred_df.insert(0, "id", sales["id"].astype(str))
    return pred_df


def add_known_validation_rows(
    submission: pd.DataFrame,
    sales_eval: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    day_cols = get_day_columns(sales_eval)
    validation_days = day_cols[-horizon:]
    validation_actuals: pd.DataFrame = sales_eval.loc[:, ["id", *validation_days]].copy()
    validation_actuals["id"] = (
        cast(Any, validation_actuals["id"]).astype(str).str.replace("_evaluation$", "_validation", regex=True)
    )
    validation_actuals = validation_actuals.rename(columns={day: f"F{i + 1}" for i, day in enumerate(validation_days)})

    forecast_cols = [f"F{i}" for i in range(1, horizon + 1)]
    actual_cols = {col: f"{col}_actual" for col in forecast_cols}
    validation_actuals = validation_actuals.rename(columns=actual_cols)
    submission = submission.merge(validation_actuals, on="id", how="left")
    for col in forecast_cols:
        submission[col] = submission[f"{col}_actual"].combine_first(submission[col])
    return submission[["id", *forecast_cols]]


def main() -> None:
    args = parse_args()
    bundle = cast(dict[str, Any], cast(Any, joblib).load(args.model_path))
    model = cast(PredictModel, bundle["model"])
    features = cast(list[str], bundle["features"])
    encoders = cast(dict[str, dict[str, int]], bundle["encoders"])

    sales, calendar, prices = read_m5_data(args.data_dir, args.sales_file)
    sample = pd.read_csv(Path(args.data_dir) / "sample_submission.csv")
    forecast_cols = [f"F{i}" for i in range(1, args.horizon + 1)]

    pred_df = forecast_horizon(
        sales=sales,
        calendar=calendar,
        prices=prices,
        model=model,
        features=features,
        encoders=encoders,
        horizon=args.horizon,
        history_days=args.history_days,
    )

    submission = sample[["id"]].merge(pred_df, on="id", how="left")
    if args.sales_file == "sales_train_evaluation.csv":
        submission = add_known_validation_rows(submission, sales, args.horizon)

    submission[forecast_cols] = submission[forecast_cols].fillna(0)
    submission.to_csv(args.output_path, index=False)
    print(f"Saved baseline submission to {args.output_path}")


if __name__ == "__main__":
    main()
