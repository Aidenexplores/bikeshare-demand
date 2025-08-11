# src/features.py
from pathlib import Path
import pandas as pd
import numpy as np
import importlib

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
MIN_PARQ = PROC / "trips_clean.parquet"
MIN_CSVGZ = PROC / "trips_clean.csv.gz"

OUT_PARQ = PROC / "daily_features.parquet"
OUT_CSVGZ = PROC / "daily_features.csv.gz"


def _has(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def _load_daily() -> pd.DataFrame:
    """Return per-station, per-day counts (rides). Chunked if reading CSV.gz."""
    # Parquet path (fast & low-memory if we select columns)
    if MIN_PARQ.exists():
        print("[features] Reading minimal parquet (trip_date, start_station_id)")
        trips = pd.read_parquet(
            MIN_PARQ, columns=["trip_date", "start_station_id"])
        trips["trip_date"] = pd.to_datetime(trips["trip_date"])
        daily = (trips.groupby(["start_station_id", "trip_date"])
                      .size().rename("rides").reset_index())
        return daily

    # CSV.gz path (stream in chunks)
    if MIN_CSVGZ.exists():
        print("[features] Reading minimal CSV.gz in chunks")
        pieces = []
        it = pd.read_csv(
            MIN_CSVGZ,
            compression="gzip",
            usecols=["trip_date", "start_station_id"],
            parse_dates=["trip_date"],
            chunksize=1_000_000
        )
        for i, ch in enumerate(it, 1):
            g = (ch.groupby(["start_station_id", "trip_date"])
                   .size().rename("rides").reset_index())
            pieces.append(g)
            if i % 5 == 0:
                print(f"[features] …processed {i} chunks")
        daily = pd.concat(pieces, ignore_index=True)
        daily = (daily.groupby(["start_station_id", "trip_date"])[
                 "rides"].sum().reset_index())
        return daily

    raise FileNotFoundError(
        "Missing cleaned file in data/processed/ (trips_clean.parquet or trips_clean.csv.gz)"
    )


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trip_date"] = pd.to_datetime(df["trip_date"])
    df["dow"] = df["trip_date"].dt.dayofweek
    df["month"] = df["trip_date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # optional stronger seasonality:
    doy = df["trip_date"].dt.dayofyear
    df["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    df["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    return df


def _add_city_context(df: pd.DataFrame) -> pd.DataFrame:
    """Citywide totals and per-day active-station count (useful context, no leakage)."""
    df = df.copy()
    city = df.groupby("trip_date")["rides"].sum().rename(
        "city_rides").reset_index()
    df = df.merge(city, on="trip_date", how="left")
    df["share"] = df["rides"] / df["city_rides"].clip(lower=1)

    active = (df.groupby("trip_date")["start_station_id"]
                .nunique().rename("active_stations").reset_index())
    df = df.merge(active, on="trip_date", how="left")
    return df


def _add_lags_rolls(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only features: lags & trailing rolling means (no target leakage)."""
    df = df.sort_values(["start_station_id", "trip_date"]).copy()

    # target lags
    for L in [1, 7, 14, 28]:
        df[f"lag_{L}"] = df.groupby("start_station_id")["rides"].shift(L)

    # trailing rolling means (shift(1) to ensure they use only prior days)
    for W in [7, 14, 28]:
        df[f"roll_mean_{W}"] = (
            df.groupby("start_station_id")["rides"]
              .shift(1).rolling(W, min_periods=2).mean()
        )

    # keep rows that have at least lag_1 available
    df = df.dropna(subset=["lag_1"])
    return df


def _save(df: pd.DataFrame):
    if _has("fastparquet"):
        df.to_parquet(OUT_PARQ, index=False, engine="fastparquet")
        print(f"[features] Saved Parquet -> {OUT_PARQ}")
    else:
        df.to_csv(OUT_CSVGZ, index=False, compression="gzip")
        print(f"[features] Saved CSV.gz -> {OUT_CSVGZ}")


if __name__ == "__main__":
    daily = _load_daily()
    print("[features] Base daily:", daily.shape)
    daily = _add_calendar(daily)
    daily = _add_city_context(daily)
    daily = _add_lags_rolls(daily)

    cols = [
        "start_station_id", "trip_date", "rides",
        "dow", "month", "is_weekend", "doy_sin", "doy_cos",
        "city_rides", "share", "active_stations",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_14", "roll_mean_28",
    ]
    daily = daily[cols]
    _save(daily)
    print("[features] Final shape:", daily.shape)
