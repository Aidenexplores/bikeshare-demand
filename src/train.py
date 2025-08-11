# src/train.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
import sklearn

print(f"[info] scikit-learn version: {sklearn.__version__}")

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
FEAT_PARQ = PROC / "daily_features.parquet"
FEAT_CSVGZ = PROC / "daily_features.csv.gz"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_features() -> pd.DataFrame:
    if FEAT_PARQ.exists():
        df = pd.read_parquet(FEAT_PARQ)
    else:
        df = pd.read_csv(FEAT_CSVGZ, parse_dates=["trip_date"])
    df["trip_date"] = pd.to_datetime(df["trip_date"])
    df = df.sort_values(["trip_date", "start_station_id"]
                        ).reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame):
    train_end = pd.Timestamp("2023-12-31")
    valid_end = pd.Timestamp("2024-06-30")
    train = df[df["trip_date"] <= train_end]
    valid = df[(df["trip_date"] > train_end) & (df["trip_date"] <= valid_end)]
    test = df[df["trip_date"] > valid_end]
    return train, valid, test


def pick_features(df: pd.DataFrame):
    # drop same-day context to avoid leakage; we can add lagged versions later
    drop_now = {"rides", "trip_date", "start_station_id",
                "city_rides", "share", "active_stations"}
    X_cols = [c for c in df.columns if c not in drop_now]
    df = df.copy()
    df["station_idx"] = pd.factorize(df["start_station_id"])[0].astype("int32")
    if "station_idx" not in X_cols:
        X_cols.append("station_idx")
    return df, X_cols


def evaluate(y_true, y_pred, label):
    # robust RMSE + MAPE (manual; no version quirks)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float),
                        0.0)  # no negative forecasts
    mse = mean_squared_error(y_true, y_pred)   # MSE
    rmse = float(np.sqrt(mse))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)))
    print(f"[eval] {label} RMSE: {rmse:,.3f}  |  MAPE: {100*mape:,.2f}%")
    return {"rmse": rmse, "mape": mape}


def per_station_rmse(df, y_true, y_pred, label, n=10):
    s = pd.DataFrame({"sid": df["start_station_id"].values,
                      "y": y_true, "yhat": y_pred})
    g = s.groupby("sid").apply(
        lambda t: mean_squared_error(t["y"], t["yhat"]) ** 0.5
    ).sort_values(ascending=False)
    print(f"[eval] {label} worst {n} stations by RMSE:")
    print(g.head(n))
    return g


def main():
    df = load_features()
    df, X_cols = pick_features(df)
    df[X_cols] = df[X_cols].fillna(0)

    train, valid, test = make_splits(df)
    y_tr, y_va, y_te = train["rides"].values, valid["rides"].values, test["rides"].values
    X_tr, X_va, X_te = train[X_cols].values, valid[X_cols].values, test[X_cols].values
    print("[info] shapes ->", X_tr.shape, X_va.shape, X_te.shape)

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_leaf_nodes=31,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    pred_va = model.predict(X_va)
    pred_te = model.predict(X_te)

    metrics = {"valid": evaluate(y_va, pred_va, "VALID"),
               "test":  evaluate(y_te, pred_te, "TEST")}
    _ = per_station_rmse(valid, y_va, pred_va, "VALID", n=10)

    dump(model, MODELS_DIR / "model.joblib")
    meta = {
        "features": X_cols,
        "train_end": "2023-12-31",
        "valid_end": "2024-06-30",
        "n_rows": int(len(df)),
        "n_stations": int(df["start_station_id"].nunique()),
        "metrics": metrics,
    }
    (MODELS_DIR / "feature_meta.json").write_text(json.dumps(meta, indent=2))
    print("[save] models/model.joblib and models/feature_meta.json written.")


if __name__ == "__main__":
    main()
