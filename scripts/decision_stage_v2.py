from pathlib import Path
import pandas as pd
import boto3

from cost_model import (
    get_ec2_hourly_cost,
    estimate_lambda_total_cost,
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

PROCESSED_BUCKET = "sourav-cost-high-20260328"
s3 = boto3.client("s3")


def get_latest_anomaly_file():
    files = sorted(PROCESSED_DIR.glob("anomaly_scores_*.parquet"))
    if not files:
        raise FileNotFoundError("No anomaly score parquet files found in data/processed")
    return files[-1]


def choose_action(service: str, metric_name: str, anomaly_score: float, value: float) -> str:
    if anomaly_score < 0.4:
        return "monitor"

    if service == "ec2":
        if metric_name == "CPUUtilization":
            if value < 10:
                return "scale_down"
            if value > 75:
                return "investigate_high_compute"
            return "monitor"

        if metric_name in ["NetworkIn", "NetworkOut"]:
            return "investigate_network"

        if metric_name in ["DiskReadOps", "DiskWriteOps"]:
            return "investigate_network"

    if service == "lambda":
        if metric_name == "Errors" and value > 0:
            return "investigate_lambda_errors"
        if metric_name == "Duration":
            return "optimize_lambda"
        if metric_name == "Invocations":
            return "limit_lambda"

    return "monitor"


def confidence_from_inputs(anomaly_score: float, severity: str, metric_name: str, value: float) -> float:
    base = min(anomaly_score / 1.5, 1.0)

    if severity == "critical":
        base += 0.15
    elif severity == "high":
        base += 0.10
    elif severity == "medium":
        base += 0.05

    if metric_name == "Errors" and value > 0:
        base += 0.10

    return round(min(base, 0.99), 4)


def build_reason(service: str, metric_name: str, value: float, anomaly_score: float, action: str) -> str:
    return (
        f"{service.upper()} metric {metric_name} showed abnormal behavior "
        f"(value={round(float(value), 4)}, anomaly_score={round(float(anomaly_score), 4)}), "
        f"so action '{action}' was selected."
    )


def estimate_current_cost(row: pd.Series) -> float:
    if row["service"] == "ec2":
        instance_type = row.get("instance_type", "t3.micro")
        return get_ec2_hourly_cost(instance_type)

    if row["service"] == "lambda":
        metric_name = row["metric_name"]
        value = float(row["value"])

        invocations = value if metric_name == "Invocations" else 10.0
        avg_duration_ms = value if metric_name == "Duration" else 250.0
        memory_mb = 128.0

        return estimate_lambda_total_cost(
            avg_duration_ms=avg_duration_ms,
            memory_mb=memory_mb,
            invocations=invocations
        )

    return 0.0


def main():
    anomaly_file = get_latest_anomaly_file()
    df = pd.read_parquet(anomaly_file)

    if df.empty:
        print("No anomaly rows found.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "status_code" not in df.columns:
        df["status_code"] = "Complete"

    df["is_nodata"] = (df["status_code"].astype(str).str.lower() == "nodata").astype(int)
    latest_df = (
        df.sort_values(["is_nodata", "timestamp"])
        .groupby(["resource_type", "resource_id", "resource_name", "metric_name"], as_index=False)
        .tail(1)
        .copy()
    )

    latest_df["service"] = latest_df["resource_type"]
    latest_df["ml_anomaly"] = latest_df["anomaly_flag"].map({1: "yes", 0: "no"})
    latest_df["severity"] = latest_df["anomaly_score"].apply(severity_from_score)

    # Default instance type for EC2 since current pipeline doesn't yet carry it through
    latest_df["instance_type"] = "t3.micro"

    latest_df["final_action"] = latest_df.apply(
        lambda row: choose_action(
            row["service"],
            row["metric_name"],
            float(row["anomaly_score"]),
            float(row["value"])
        ),
        axis=1
    )
    latest_df["final_action"] = latest_df["final_action"].map(normalize_action)
    latest_df["final_action"] = latest_df.apply(
        lambda row: enforce_service_action_policy(str(row["service"]), str(row["final_action"])),
        axis=1,
    )

    latest_df["confidence_score"] = latest_df.apply(
        lambda row: confidence_from_inputs(
            float(row["anomaly_score"]),
            row["severity"],
            row["metric_name"],
            float(row["value"])
        ),
        axis=1
    )

    latest_df["reason"] = latest_df.apply(
        lambda row: build_reason(
            row["service"],
            row["metric_name"],
            float(row["value"]),
            float(row["anomaly_score"]),
            row["final_action"]
        ),
        axis=1
    )

    latest_df["estimated_cost_before"] = latest_df.apply(estimate_current_cost, axis=1)
    latest_df["estimated_cost_after"] = latest_df.apply(
        lambda row: estimate_post_action_cost(
            row["service"],
            float(row["estimated_cost_before"]),
            row["final_action"]
        ),
        axis=1
    )
    latest_df["estimated_savings"] = latest_df.apply(
        lambda row: estimate_savings(
            float(row["estimated_cost_before"]),
            float(row["estimated_cost_after"])
        ),
        axis=1
    )
    latest_df["is_optimization_action"] = latest_df["final_action"].apply(is_optimization_action)
    latest_df["estimated_savings"] = latest_df["estimated_savings"].clip(lower=0.0)
    latest_df["estimated_cost_after"] = latest_df["estimated_cost_before"] - latest_df["estimated_savings"]
    latest_df["impact"] = latest_df["estimated_savings"].apply(lambda x: "saving" if x > 0 else "neutral")
    latest_df["action_status"] = latest_df.apply(
        lambda row: (
            action_status_from_confidence(float(row["confidence_score"]))
            if bool(row["is_optimization_action"]) and float(row["estimated_savings"]) > 0
            else "recommendation_with_approval"
        ),
        axis=1,
    )
    latest_df["timestamp"] = latest_df["timestamp"].astype(str)

    final_df = latest_df[
        [
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
            "timestamp"
        ]
    ].drop_duplicates()

    output_csv = PROCESSED_DIR / "final_output.csv"
    final_df.to_csv(output_csv, index=False)

    s3.upload_file(str(output_csv), PROCESSED_BUCKET, "processed/final_output.csv")

    print(f"Read anomaly file: {anomaly_file}")
    print(f"Final output rows: {len(final_df)}")
    print(f"Saved local file: {output_csv}")
    print(f"Uploaded to s3://{PROCESSED_BUCKET}/processed/final_output.csv")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
