from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_BUCKET = "sourav-cost-high-20260328"

import boto3
s3 = boto3.client("s3")


def get_latest_raw_file():
    files = sorted(RAW_DIR.glob("metrics_*.json"))
    if not files:
        raise FileNotFoundError("No raw metric files found in data/raw")
    return files[-1]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["value"] = pd.to_numeric(df.get("value", 0), errors="coerce").fillna(0.0)
    # Mixed ISO formats can appear (with/without fractional seconds). Parse robustly.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["resource_type", "resource_id", "metric_name", "timestamp"])

    df["rolling_mean_3"] = (
        df.groupby(["resource_type", "resource_id", "metric_name"])["value"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    df["rolling_std_3"] = (
        df.groupby(["resource_type", "resource_id", "metric_name"])["value"]
        .transform(lambda s: s.rolling(3, min_periods=1).std().fillna(0))
    )

    df["pct_change"] = (
        df.groupby(["resource_type", "resource_id", "metric_name"])["value"]
        .transform(lambda s: s.pct_change().replace([float("inf"), float("-inf")], 0).fillna(0))
    )

    return df


def main():
    raw_file = get_latest_raw_file()

    with open(raw_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("rows", [])
    df = pd.DataFrame(rows)

    if df.empty:
        print("No rows found in raw payload.")
        return

    features_df = build_features(df)

    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    csv_file = PROCESSED_DIR / f"features_{ts}.csv"
    parquet_file = PROCESSED_DIR / f"features_{ts}.parquet"

    features_df.to_csv(csv_file, index=False)
    features_df.to_parquet(parquet_file, index=False)

    s3.upload_file(str(csv_file), PROCESSED_BUCKET, f"processed/{csv_file.name}")
    s3.upload_file(str(parquet_file), PROCESSED_BUCKET, f"processed/{parquet_file.name}")

    print(f"Read raw file: {raw_file}")
    print(f"Feature rows: {len(features_df)}")
    print(f"Saved CSV: {csv_file}")
    print(f"Saved Parquet: {parquet_file}")
    print(f"Uploaded to s3://{PROCESSED_BUCKET}/processed/")


if __name__ == "__main__":
    main()
