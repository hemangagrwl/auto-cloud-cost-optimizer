from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_BUCKET = "sourav-cost-high-20260328"

import boto3
s3 = boto3.client("s3")


def get_latest_feature_file():
    files = sorted(PROCESSED_DIR.glob("features_*.parquet"))
    if not files:
        raise FileNotFoundError("No processed feature files found")
    return files[-1]


def compute_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    pct_component = df["pct_change"].abs()
    cv_component = (df["rolling_std_3"] / (df["rolling_mean_3"].abs() + 1e-6))

    # Guard against noisy "drop to zero" spikes on EC2 network metrics.
    # We still keep anomaly sensitivity, but damp pct_change impact for hard-zero samples.
    network_metric_mask = (
        df["resource_type"].astype(str).str.lower().eq("ec2")
        & df["metric_name"].astype(str).isin(["NetworkIn", "NetworkOut"])
    )
    network_zero_mask = (
        network_metric_mask
        & pd.to_numeric(df["value"], errors="coerce").fillna(0.0).eq(0.0)
    )
    pct_component = pct_component.where(~network_zero_mask, pct_component * 0.2)
    # Additional stabilization for near-zero traffic baselines where CV can explode.
    near_zero_baseline_mask = network_metric_mask & pd.to_numeric(df["rolling_mean_3"], errors="coerce").fillna(0.0).abs().lt(1.0)
    cv_component = cv_component.where(~near_zero_baseline_mask, cv_component * 0.3)

    df["anomaly_score"] = (pct_component * 0.5) + (cv_component * 0.5)

    # Service-aware thresholds:
    # - Keep EC2/S3 at 0.8 to avoid over-alerting.
    # - Lower Lambda threshold so serverless anomalies are detected earlier.
    default_threshold = 0.8
    lambda_threshold = 0.4
    df["anomaly_threshold"] = default_threshold
    if "resource_type" in df.columns:
        df.loc[df["resource_type"] == "lambda", "anomaly_threshold"] = lambda_threshold

    df["anomaly_flag"] = (df["anomaly_score"] >= df["anomaly_threshold"]).astype(int)
    return df


def main():
    feature_file = get_latest_feature_file()
    df = pd.read_parquet(feature_file)

    scored_df = compute_anomaly_scores(df)

    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_csv = PROCESSED_DIR / f"anomaly_scores_{ts}.csv"
    out_parquet = PROCESSED_DIR / f"anomaly_scores_{ts}.parquet"

    scored_df.to_csv(out_csv, index=False)
    scored_df.to_parquet(out_parquet, index=False)

    s3.upload_file(str(out_csv), PROCESSED_BUCKET, f"processed/{out_csv.name}")
    s3.upload_file(str(out_parquet), PROCESSED_BUCKET, f"processed/{out_parquet.name}")

    print(f"Read feature file: {feature_file}")
    print(f"Scored rows: {len(scored_df)}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_parquet}")


if __name__ == "__main__":
    main()
