from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def load_model_and_meta():
    model = load(MODELS / "model.joblib")
    meta = json.loads((MODELS / "feature_meta.json").read_text())
    return model, meta


def load_features_df():
    f = PROC / "daily_features.parquet"
    if f.exists():
        df = pd.read_parquet(f)
    else:
        f = PROC / "daily_features.csv.gz"
        df = pd.read_csv(f, parse_dates=["trip_date"])
    df = df.sort_values(["trip_date", "start_station_id"]
                        ).reset_index(drop=True)
    return df


def make_station_index(df):
    codes, uniques = pd.factorize(df["start_station_id"])
    m = pd.Series(codes, index=df["start_station_id"]
                  ).drop_duplicates().to_dict()
    return m


def calendar_features(dates):
    dow = dates.dt.dayofweek
    month = dates.dt.month
    is_weekend = (dow >= 5).astype(int)
    doy = dates.dt.dayofyear
    doy_sin = np.sin(2*np.pi*doy/365.25)
    doy_cos = np.cos(2*np.pi*doy/365.25)
    return pd.DataFrame({"dow": dow, "month": month, "is_weekend": is_weekend, "doy_sin": doy_sin, "doy_cos": doy_cos}, index=dates.index)


def forecast(station_id, horizon=14):
    model, meta = load_model_and_meta()
    df = load_features_df()
    try:
        sid = int(station_id)
    except:
        sid = station_id
    hist = df[df["start_station_id"] == sid].sort_values("trip_date").copy()
    if hist.empty:
        raise ValueError("station not found")
    rides_hist = hist["rides"].astype(float).tolist()
    dates_hist = hist["trip_date"].tolist()
    while len(rides_hist) < 28:
        rides_hist.insert(0, 0.0)
    sid_to_idx = make_station_index(df)
    station_idx = sid_to_idx.get(sid, 0)
    last_date = df["trip_date"].max()
    preds = []
    future_dates = []
    for k in range(1, horizon + 1):
        d = last_date + pd.Timedelta(days=k)
        cal = calendar_features(pd.Series([d]))
        feats = {
            "dow": int(cal.iloc[0]["dow"]),
            "month": int(cal.iloc[0]["month"]),
            "is_weekend": int(cal.iloc[0]["is_weekend"]),
            "doy_sin": float(cal.iloc[0]["doy_sin"]),
            "doy_cos": float(cal.iloc[0]["doy_cos"]),
            "lag_1": rides_hist[-1],
            "lag_7": rides_hist[-7],
            "lag_14": rides_hist[-14],
            "lag_28": rides_hist[-28],
            "roll_mean_7": float(np.mean(rides_hist[-7:])),
            "roll_mean_14": float(np.mean(rides_hist[-14:])),
            "roll_mean_28": float(np.mean(rides_hist[-28:])),
            "station_idx": station_idx,
        }
        X = np.array([[feats.get(c, 0.0)
                     for c in meta["features"]]], dtype=float)
        yhat = float(model.predict(X)[0])
        yhat = max(0.0, yhat)
        preds.append(yhat)
        future_dates.append(d)
        rides_hist.append(yhat)
    fcst = pd.DataFrame({"trip_date": future_dates, "pred_rides": preds})
    return fcst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--station", required=True)
    p.add_argument("--horizon", type=int, default=14)
    args = p.parse_args()
    df = forecast(args.station, args.horizon)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
