from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import boto3
import json

from cost_model import (
    get_ec2_hourly_cost,
    estimate_lambda_total_cost,
    estimate_s3_hourly_storage_cost_from_bytes,
    estimate_post_action_cost,
    estimate_savings,
    action_status_from_confidence,
    severity_from_score,
    normalize_action,
    is_optimization_action,
    enforce_service_action_policy,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "data" / "config"

PROCESSED_BUCKET = "sourav-cost-high-20260328"
s3 = boto3.client("s3")


def get_latest_anomaly_file():
    files = sorted(PROCESSED_DIR.glob("anomaly_scores_*.parquet"))
    if not files:
        raise FileNotFoundError("No anomaly files found")
    return files[-1]


def load_microservice_map() -> dict:
    config_path = CONFIG_DIR / "microservice_map.json"
    if not config_path.exists():
        return {"resource_id": {}, "resource_name": {}}
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "resource_id": data.get("resource_id", {}),
        "resource_name": data.get("resource_name", {}),
    }


def first_non_null(series: pd.Series, default):
    cleaned = series.dropna()
    if cleaned.empty:
        return default
    return cleaned.iloc[0]


def collapse_status(series: pd.Series) -> str:
    values = {str(v).strip().lower() for v in series.dropna().tolist()}
    if "complete" in values:
        return "Complete"
    if "carryforward" in values:
        return "CarryForward"
    if "nodata" in values:
        return "NoData"
    return "Complete"


def select_latest_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["is_nodata"] = (df["status_code"].astype(str).str.lower() == "nodata").astype(int)

    # Rule: pick newest timestamp first; if same timestamp exists, prefer real datapoint over NoData.
    ordered = df.sort_values(
        ["resource_type", "resource_id", "resource_name", "metric_name", "timestamp", "is_nodata"],
        ascending=[True, True, True, True, False, True],
    )
    return ordered.groupby(["resource_type", "resource_id", "resource_name", "metric_name"], as_index=False).head(1)


def append_s3_carryforward_rows(latest_file: Path, latest_df: pd.DataFrame):
    if "resource_type" in latest_df.columns and (latest_df["resource_type"] == "s3").any():
        return latest_df, None, 0

    files = sorted(PROCESSED_DIR.glob("anomaly_scores_*.parquet"))
    previous_files = [f for f in files if f != latest_file]

    for prev_file in reversed(previous_files):
        prev_df = pd.read_parquet(prev_file)
        if "resource_type" not in prev_df.columns:
            continue

        s3_prev = prev_df[prev_df["resource_type"] == "s3"].copy()
        if s3_prev.empty:
            continue

        if "status_code" not in s3_prev.columns:
            s3_prev["status_code"] = "Complete"

        s3_latest = select_latest_metric_rows(s3_prev)
        s3_latest["status_code"] = "CarryForward"
        merged = pd.concat([latest_df, s3_latest], ignore_index=True)
        return merged, prev_file, len(s3_latest)

    return latest_df, None, 0


def build_resource_level_df(latest_metric_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["resource_type", "resource_id", "resource_name"]

    latest_metric_df = latest_metric_df.copy()
    latest_metric_df["anomaly_score"] = pd.to_numeric(latest_metric_df["anomaly_score"], errors="coerce").fillna(0.0)
    latest_metric_df["anomaly_flag"] = pd.to_numeric(latest_metric_df["anomaly_flag"], errors="coerce").fillna(0).astype(int)
    latest_metric_df["value"] = pd.to_numeric(latest_metric_df["value"], errors="coerce").fillna(0.0)

    idx = latest_metric_df.groupby(keys)["anomaly_score"].idxmax()
    primary = latest_metric_df.loc[idx, keys + [
        "metric_name",
        "value",
        "rolling_mean_3",
        "rolling_std_3",
        "pct_change",
        "status_code",
    ]].rename(columns={"metric_name": "primary_metric_name", "value": "primary_value", "status_code": "primary_status_code"})

    metrics_wide = (
        latest_metric_df.pivot_table(index=keys, columns="metric_name", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    agg = latest_metric_df.groupby(keys, as_index=False).agg(
        anomaly_score=("anomaly_score", "max"),
        any_anomaly=("anomaly_flag", "max"),
        timestamp=("timestamp", "max"),
        instance_type=("instance_type", lambda s: first_non_null(s, "t3.micro")),
        lambda_memory_mb=("lambda_memory_mb", lambda s: first_non_null(pd.to_numeric(s, errors="coerce"), 128.0)),
        microservice=("microservice", lambda s: first_non_null(s, "unassigned")),
    )

    any_real = latest_metric_df.groupby(keys, as_index=False).agg(
        status_code=("status_code", collapse_status)
    )

    resource_df = agg.merge(metrics_wide, on=keys, how="left").merge(primary, on=keys, how="left").merge(any_real, on=keys, how="left")

    for col in [
        "CPUUtilization",
        "NetworkIn",
        "NetworkOut",
        "Invocations",
        "Errors",
        "Duration",
        "BucketSizeBytes",
        "NumberOfObjects",
    ]:
        if col not in resource_df.columns:
            resource_df[col] = 0.0
        resource_df[col] = pd.to_numeric(resource_df[col], errors="coerce").fillna(0.0)

    resource_df["error_rate"] = resource_df["Errors"] / resource_df["Invocations"].clip(lower=1.0)
    resource_df["ml_anomaly"] = resource_df["any_anomaly"].map({1: "yes", 0: "no"})
    resource_df["service"] = resource_df["resource_type"]

    # Features used by existing trained artifact (resource-level via primary metric row).
    resource_df["is_ec2"] = (resource_df["resource_type"] == "ec2").astype(int)
    resource_df["is_lambda"] = (resource_df["resource_type"] == "lambda").astype(int)
    resource_df["is_s3"] = (resource_df["resource_type"] == "s3").astype(int)
    resource_df["cost"] = np.select(
        [
            resource_df["resource_type"] == "ec2",
            resource_df["resource_type"] == "lambda",
            resource_df["resource_type"] == "s3",
        ],
        [0.0104, 0.0001, 0.00005],
        default=0.0001,
    )

    for col in ["primary_value", "rolling_mean_3", "rolling_std_3", "pct_change"]:
        resource_df[col] = pd.to_numeric(resource_df[col], errors="coerce").fillna(0.0)
    # Keep legacy model feature name compatibility ("value" was used at training time).
    resource_df["value"] = resource_df["primary_value"]

    return resource_df


def apply_microservice_mapping(resource_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    d = resource_df.copy()
    by_id = mapping.get("resource_id", {})
    by_name = mapping.get("resource_name", {})

    mapped_by_id = d["resource_id"].astype(str).map(by_id)
    mapped_by_name = d["resource_name"].astype(str).map(by_name)
    d["microservice"] = mapped_by_id.fillna(mapped_by_name).fillna(d["microservice"]).astype(str)
    return d


def estimate_current_cost(row: pd.Series) -> float:
    if row["service"] == "ec2":
        instance_type = str(row.get("instance_type", "t3.micro") or "t3.micro")
        return get_ec2_hourly_cost(instance_type)

    if row["service"] == "lambda":
        invocations = float(row.get("Invocations", 10.0))
        avg_duration_ms = float(row.get("Duration", 250.0))
        memory_mb = float(row.get("lambda_memory_mb", 128.0) or 128.0)
        return estimate_lambda_total_cost(
            avg_duration_ms=avg_duration_ms,
            memory_mb=memory_mb,
            invocations=max(invocations, 0.0),
        )

    if row["service"] == "s3":
        storage_bytes = float(row.get("BucketSizeBytes", 0.0))
        return estimate_s3_hourly_storage_cost_from_bytes(storage_bytes)

    return 0.0


def build_reason(row: pd.Series) -> str:
    action = str(row["final_action"])
    optimization_tag = "cost optimization" if is_optimization_action(action) else "investigation/monitoring"
    return (
        f"{row['service'].upper()} primary metric {row['metric_name']} had "
        f"value={round(float(row['value']), 4)}, "
        f"anomaly_score={round(float(row['anomaly_score']), 4)}, "
        f"status={row['status_code']}; "
        f"model predicted '{action}' ({optimization_tag}) "
        f"with model_confidence={float(row['model_confidence']):.4f}, "
        f"policy_confidence={float(row['policy_confidence']):.4f}, "
        f"final_confidence={float(row['final_confidence']):.4f}."
    )


def compute_policy_confidence(row: pd.Series) -> float:
    score = 0.55
    anomaly_score = float(row.get("anomaly_score", 0.0))
    if anomaly_score >= 1.2:
        score += 0.20
    elif anomaly_score >= 0.8:
        score += 0.15
    elif anomaly_score >= 0.4:
        score += 0.08

    status = str(row.get("status_code", "Complete")).strip().lower()
    if status == "complete":
        score += 0.10
    elif status == "carryforward":
        score -= 0.05
    elif status == "nodata":
        score -= 0.20

    if bool(row.get("is_optimization_action", False)):
        score += 0.08
    if float(row.get("estimated_savings", 0.0)) > 0:
        score += 0.07

    return float(np.clip(score, 0.05, 0.99))


def main():
    anomaly_file = get_latest_anomaly_file()
    df = pd.read_parquet(anomaly_file)

    if df.empty:
        print("No anomaly rows found.")
        return

    if "status_code" not in df.columns:
        df["status_code"] = "Complete"
    if "instance_type" not in df.columns:
        df["instance_type"] = "t3.micro"
    if "lambda_memory_mb" not in df.columns:
        df["lambda_memory_mb"] = 128.0
    if "microservice" not in df.columns:
        df["microservice"] = df.get("resource_name", "unassigned")

    df, carryforward_file, carryforward_rows = append_s3_carryforward_rows(anomaly_file, df)
    if carryforward_rows > 0:
        print(
            f"S3 carry-forward applied: {carryforward_rows} rows reused from {carryforward_file.name}"
        )

    latest_metric_df = select_latest_metric_rows(df)
    resource_df = build_resource_level_df(latest_metric_df)
    resource_df = apply_microservice_mapping(resource_df, load_microservice_map())

    model = joblib.load(MODELS_DIR / "model.joblib")
    labels = joblib.load(MODELS_DIR / "labels.joblib")

    candidate_features = [
        "value",
        "anomaly_score",
        "rolling_mean_3",
        "rolling_std_3",
        "pct_change",
        "is_ec2",
        "is_lambda",
        "is_s3",
        "cost",
    ]
    model_feature_count = int(getattr(model, "n_features_in_", len(candidate_features)))
    feature_cols = candidate_features[:model_feature_count]
    X = resource_df[feature_cols].fillna(0)

    pred = model.predict(X)
    proba = model.predict_proba(X)

    resource_df["model_action"] = [labels[int(p)] for p in pred]
    resource_df["final_action"] = resource_df["model_action"].map(normalize_action)
    resource_df["model_confidence"] = proba.max(axis=1).astype(float).round(4)
    resource_df["decision_scope"] = "resource_level_primary_metric"

    resource_df["final_action"] = resource_df.apply(
        lambda row: enforce_service_action_policy(str(row["service"]), str(row["final_action"])),
        axis=1,
    )
    resource_df["severity"] = resource_df["anomaly_score"].apply(severity_from_score)

    resource_df["estimated_cost_before"] = resource_df.apply(estimate_current_cost, axis=1)
    resource_df["estimated_cost_after"] = resource_df.apply(
        lambda row: estimate_post_action_cost(
            row["service"],
            float(row["estimated_cost_before"]),
            row["final_action"],
        ),
        axis=1,
    )
    resource_df["estimated_savings"] = resource_df.apply(
        lambda row: estimate_savings(
            float(row["estimated_cost_before"]),
            float(row["estimated_cost_after"]),
        ),
        axis=1,
    )

    resource_df["is_optimization_action"] = resource_df["final_action"].apply(is_optimization_action)
    resource_df["estimated_savings"] = resource_df["estimated_savings"].clip(lower=0.0)
    resource_df["estimated_cost_after"] = resource_df["estimated_cost_before"] - resource_df["estimated_savings"]
    resource_df["policy_confidence"] = resource_df.apply(compute_policy_confidence, axis=1).round(4)
    resource_df["final_confidence"] = (
        (resource_df["model_confidence"] * 0.6) + (resource_df["policy_confidence"] * 0.4)
    ).round(4)
    # Backward compatibility for downstream dashboard mapping.
    resource_df["confidence_score"] = resource_df["final_confidence"]
    resource_df["agreement_score"] = resource_df["final_confidence"]

    resource_df["impact"] = resource_df["estimated_savings"].apply(lambda x: "saving" if x > 0 else "neutral")
    resource_df["action_status"] = resource_df.apply(
        lambda row: (
            action_status_from_confidence(float(row["final_confidence"]))
            if bool(row["is_optimization_action"]) and float(row["estimated_savings"]) > 0
            else "recommendation_with_approval"
        ),
        axis=1,
    )

    resource_df["metric_name"] = resource_df["primary_metric_name"].fillna("unknown")
    resource_df["primary_trigger_metric"] = resource_df["metric_name"]
    resource_df["value"] = pd.to_numeric(resource_df["value"], errors="coerce").fillna(0.0)
    resource_df["reason"] = resource_df.apply(build_reason, axis=1)
    resource_df["timestamp"] = pd.to_datetime(resource_df["timestamp"], utc=True, errors="coerce").astype(str)

    ordered_cols = [
        # Legacy-first order requested by user
        "service",
        "resource_id",
        "resource_name",
        "ml_anomaly",
        "final_action",
        "severity",
        "confidence_score",
        "reason",
        "estimated_cost_before",
        "estimated_cost_after",
        "estimated_savings",
        "impact",
        "action_status",
        "timestamp",
        # Extended schema after legacy block
        "microservice",
        "primary_trigger_metric",
        "metric_name",
        "value",
        "CPUUtilization",
        "NetworkIn",
        "NetworkOut",
        "Invocations",
        "Errors",
        "Duration",
        "error_rate",
        "BucketSizeBytes",
        "NumberOfObjects",
        "anomaly_score",
        "status_code",
        "model_confidence",
        "policy_confidence",
        "final_confidence",
        "agreement_score",
        "decision_scope",
    ]
    ordered_cols = [c for c in ordered_cols if c in resource_df.columns]
    final_df = resource_df[ordered_cols].drop_duplicates(subset=["service", "resource_id"], keep="last")

    output_csv = PROCESSED_DIR / "final_output.csv"
    final_df.to_csv(output_csv, index=False)
    s3.upload_file(str(output_csv), PROCESSED_BUCKET, "processed/final_output.csv")

    print(f"Read anomaly file: {anomaly_file}")
    print(f"Final output rows: {len(final_df)}")
    print(f"Saved local file: {output_csv}")
    print(f"Uploaded to s3://{PROCESSED_BUCKET}/processed/final_output.csv")
    print(final_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
