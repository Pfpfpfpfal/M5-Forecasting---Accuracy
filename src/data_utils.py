from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, cast

import pandas as pd


ID_COLS: list[str] = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
LAGS: list[int] = [7, 14, 28]
ROLLING_WINDOWS: list[int] = [7, 28]
CAT_COLS: list[str] = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "weekday",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]
DROP_FEATURE_COLS: list[str] = [
    "id",
    "d",
    "date",
    "sales",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]

CALENDAR_DTYPES: dict[str, str] = {
    "wm_yr_wk": "int16",
    "weekday": "category",
    "wday": "int8",
    "month": "int8",
    "year": "int16",
    "d": "category",
    "event_name_1": "category",
    "event_type_1": "category",
    "event_name_2": "category",
    "event_type_2": "category",
    "snap_CA": "int8",
    "snap_TX": "int8",
    "snap_WI": "int8",
}

PRICE_DTYPES: dict[str, str] = {
    "store_id": "category",
    "item_id": "category",
    "wm_yr_wk": "int16",
    "sell_price": "float32",
}


def day_number(day: str) -> int:
    return int(day.split("_")[1])


def get_day_columns(df: pd.DataFrame) -> list[str]:
    columns = [str(col) for col in df.columns]
    return sorted([col for col in columns if col.startswith("d_")], key=day_number)


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to keep the M5 long table manageable."""

    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def read_m5_data(data_dir: str | Path, sales_file: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    sales_header = [str(col) for col in pd.read_csv(data_path / sales_file, nrows=0).columns]
    sales_dtypes = {col: "category" for col in ID_COLS}
    sales_dtypes.update({col: "int16" for col in sales_header if col.startswith("d_")})

    sales = pd.read_csv(data_path / sales_file, dtype=cast(Any, sales_dtypes))
    calendar = pd.read_csv(data_path / "calendar.csv", dtype=cast(Any, CALENDAR_DTYPES), parse_dates=["date"])
    prices = pd.read_csv(data_path / "sell_prices.csv", dtype=cast(Any, PRICE_DTYPES))
    return sales, calendar, prices


def melt_sales(
    sales: pd.DataFrame,
    start_day: int | None = None,
    end_day: int | None = None,
) -> pd.DataFrame:
    day_cols = get_day_columns(sales)
    if start_day is not None:
        day_cols = [col for col in day_cols if day_number(col) >= start_day]
    if end_day is not None:
        day_cols = [col for col in day_cols if day_number(col) <= end_day]

    long_sales = sales.melt(
        id_vars=cast(Any, ID_COLS),
        value_vars=cast(Any, day_cols),
        var_name="d",
        value_name="sales",
    )
    long_sales["d_num"] = long_sales["d"].astype(str).str[2:].astype("int16")
    return reduce_memory(long_sales)


def add_calendar_price_features(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.merge(calendar, on="d", how="left")
    frame = frame.merge(prices, on=cast(Any, ["store_id", "item_id", "wm_yr_wk"]), how="left")

    snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    frame["snap"] = 0
    for state, col in snap_map.items():
        mask = frame["state_id"].eq(state)
        frame.loc[mask, "snap"] = frame.loc[mask, col].fillna(0).astype("int8")

    frame["sell_price"] = frame["sell_price"].fillna(0)
    frame["is_weekend"] = frame["wday"].isin([1, 2]).astype("int8")
    date_series = cast(Any, frame["date"]).dt
    frame["dayofyear"] = date_series.dayofyear.astype("int16")
    frame["weekofyear"] = date_series.isocalendar().week.astype("int16")
    frame["quarter"] = date_series.quarter.astype("int8")
    frame["has_event_1"] = frame["event_name_1"].notna().astype("int8")
    frame["has_event_2"] = frame["event_name_2"].notna().astype("int8")
    return reduce_memory(frame)


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["id", "d_num"]).copy()
    grouped = frame.groupby("id", observed=True)["sales"]

    for lag in LAGS:
        frame[f"lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(28)
    for window in ROLLING_WINDOWS:
        frame[f"rolling_mean_{window}"] = (
            cast(Any, shifted).groupby(frame["id"], observed=True)
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return reduce_memory(frame)


def fit_category_encoders(frame: pd.DataFrame, cat_cols: Iterable[str] = CAT_COLS) -> dict[str, dict[str, int]]:
    encoders: dict[str, dict[str, int]] = {}
    for col in cat_cols:
        values = cast(Any, frame[col]).astype("string").fillna("missing").unique()
        encoders[col] = {value: code for code, value in enumerate(sorted(values), start=1)}
    return encoders


def apply_category_encoders(
    frame: pd.DataFrame,
    encoders: dict[str, dict[str, int]],
) -> pd.DataFrame:
    for col, mapping in encoders.items():
        frame[col] = cast(Any, frame[col]).astype("string").fillna("missing").map(mapping).fillna(0).astype("int16")
    return frame


def build_feature_frame(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    start_day: int | None = None,
    end_day: int | None = None,
    encoders: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    frame = melt_sales(sales, start_day=start_day, end_day=end_day)
    frame = add_calendar_price_features(frame, calendar, prices)
    frame = add_lag_features(frame)

    if encoders is None:
        encoders = fit_category_encoders(frame)
    frame = apply_category_encoders(frame, encoders)
    return frame, encoders


def feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = set(DROP_FEATURE_COLS)
    return [col for col in frame.columns if col not in blocked]
