# src/prepare_data.py
from pathlib import Path
import pandas as pd
import numpy as np
import zipfile
import sys
import importlib

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[^\w]+", "_", regex=True)
    )
    return df


def _guess_col(df, want):
    want = want.lower()
    cands = [c for c in df.columns if want in c]
    return min(cands, key=len) if cands else None


def _extract_zips():
    """Extract any .zip files in RAW_DIR into RAW_DIR (if not already extracted)."""
    for zf in sorted(RAW_DIR.glob("*.zip")):
        try:
            with zipfile.ZipFile(zf, "r") as z:
                members = z.namelist()
                need_extract = any(not (RAW_DIR / Path(m).name).exists()
                                   for m in members)
                if need_extract:
                    print(
                        f"[prepare_data] Extracting ZIP: {zf.name} -> {RAW_DIR}")
                    z.extractall(RAW_DIR)
        except zipfile.BadZipFile:
            print(
                f"[prepare_data] WARNING: Bad zip file skipped: {zf.name}", file=sys.stderr)


def _read_one_csv(path: Path) -> pd.DataFrame:
    """Robust CSV reader: try encodings and skip bad lines if needed."""
    errors = []
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            df = pd.read_csv(path, low_memory=False,
                             encoding=enc, on_bad_lines="skip")
            return _normalize_cols(df)
        except Exception as e:
            errors.append(f"{enc}: {e}")
    raise RuntimeError(
        f"Failed to read CSV {path.name} -> " + " | ".join(errors))


def _read_one_excel(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name=0)
        return _normalize_cols(df)
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel {path.name}: {e}")


def _has(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _save_with_fallbacks(df: pd.DataFrame, out_parquet: Path) -> Path:
    """
    Try Parquet with fastparquet; if not available, save CSV.gz.
    Returns the final path written.
    """
    # Prefer fastparquet (avoid pyarrow ABI issues)
    if _has("fastparquet"):
        df.to_parquet(out_parquet, index=False, engine="fastparquet")
        print(f"[prepare_data] Saved Parquet (fastparquet) -> {out_parquet}")
        return out_parquet

    # Fallback to CSV.gz
    csv_out = out_parquet.with_suffix(".csv.gz")
    df.to_csv(csv_out, index=False, compression="gzip")
    print(f"[prepare_data] Saved CSV.gz fallback -> {csv_out}")
    return csv_out

# ---------- pipeline steps ----------


def load_raw() -> pd.DataFrame:
    """Load and concatenate all raw files from RAW_DIR (csv/xlsx/xls, zip-aware)."""
    _extract_zips()
    csvs = sorted(RAW_DIR.glob("*.csv"))
    excels = sorted(list(RAW_DIR.glob("*.xlsx")) + list(RAW_DIR.glob("*.xls")))
    if not csvs and not excels:
        raise FileNotFoundError(
            f"No raw files in {RAW_DIR}. Put CSV/XLSX there (ZIPs also ok)."
        )
    dfs = []
    for f in csvs:
        print(f"[prepare_data] Reading CSV: {f.name}")
        dfs.append(_read_one_csv(f))
    for f in excels:
        print(f"[prepare_data] Reading Excel: {f.name}")
        dfs.append(_read_one_excel(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"[prepare_data] Loaded {len(df):,} rows from {len(dfs)} file(s).")
    return df


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize to daily trip rows with station metadata. Handles schema variants."""
    df = df.copy()

    start_time_col = _guess_col(
        df, "start_time") or _guess_col(df, "trip_start_time")
    end_time_col = _guess_col(
        df, "end_time") or _guess_col(df, "trip_end_time")
    dur_col = _guess_col(df, "duration") or _guess_col(df, "trip_duration")

    start_sid_col = (
        _guess_col(df, "start_station_id")
        or _guess_col(df, "from_station_id")
        or _guess_col(df, "start_station")
        or _guess_col(df, "start_station_number")
    )
    start_sname_col = (
        _guess_col(df, "start_station_name")
        or _guess_col(df, "from_station_name")
        or _guess_col(df, "start_station")
    )

    if not all([start_time_col, end_time_col, start_sid_col]):
        raise ValueError(
            "Missing required columns (start_time, end_time, start_station_id). "
            f"Columns present (first 20): {list(df.columns)[:20]}"
        )

    # Parse timestamps
    df[start_time_col] = pd.to_datetime(df[start_time_col], errors="coerce")
    df[end_time_col] = pd.to_datetime(df[end_time_col], errors="coerce")

    # Duration in minutes
    if dur_col is None:
        df["duration_min"] = (
            df[end_time_col] - df[start_time_col]).dt.total_seconds() / 60.0
    else:
        dtry = pd.to_numeric(df[dur_col], errors="coerce")
        if dtry.notna().any():
            med = dtry.dropna().median()
            df["duration_min"] = dtry / \
                60.0 if (pd.notna(med) and med > 300) else dtry
        else:
            df["duration_min"] = (
                df[end_time_col] - df[start_time_col]).dt.total_seconds() / 60.0

    # Basic cleaning
    before = len(df)
    df = df.dropna(subset=[start_time_col, end_time_col, "duration_min"])
    df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 240)]
    after = len(df)
    print(f"[prepare_data] Dropped {before - after:,} invalid rows.")

    # Daily grain + station fields
    df["trip_date"] = pd.to_datetime(df[start_time_col].dt.date)
    df["start_station_id"] = df[start_sid_col].astype(str).str.strip()
    df["start_station_name"] = (
        df[start_sname_col].astype(
            str).str.strip() if start_sname_col else np.nan
    )

    out = df[["trip_date", "start_station_id",
              "start_station_name", "duration_min"]].copy()
    print(f"[prepare_data] Cleaned shape: {out.shape}")
    return out


def main():
    print(f"[prepare_data] RAW_DIR = {RAW_DIR}")
    print(f"[prepare_data] PROC_DIR = {PROC_DIR}")
    df_raw = load_raw()
    df_clean = clean_trips(df_raw)

    # Save (Parquet via fastparquet if possible; else CSV.gz)
    out_parquet = PROC_DIR / "trips_clean.parquet"
    final_path = _save_with_fallbacks(df_clean, out_parquet)

    # ---------- verification ----------
    if final_path.exists():
        print(f"[prepare_data] ✅ File saved at: {final_path}")
        try:
            size = final_path.stat().st_size
            print(f"[prepare_data] File size: {size:,} bytes")
        except Exception:
            pass

        # Preview a small sample without loading everything
        try:
            if final_path.suffix == ".parquet":
                sample = pd.read_parquet(
                    final_path, engine="fastparquet").head(5)
                shape = pd.read_parquet(final_path, engine="fastparquet").shape
            else:
                sample = pd.read_csv(final_path, compression="gzip", nrows=5)
                # estimate shape cheaply by counting rows in chunks
                total = 0
                for ch in pd.read_csv(final_path, compression="gzip", usecols=["trip_date"], chunksize=1_000_000):
                    total += len(ch)
                shape = (total, 4)
            print("[prepare_data] Preview of cleaned data:")
            print(sample)
            print(f"[prepare_data] Shape (approx for CSV.gz): {shape}")
        except Exception as e:
            print(f"[prepare_data] NOTE: Could not preview saved file: {e}")
    else:
        print("[prepare_data] ❌ File was not created.")


if __name__ == "__main__":
    main()
