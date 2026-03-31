import pandas as pd
from pathlib import Path

# load latest anomaly file
file = sorted(Path("data/processed").glob("anomaly_scores_*.parquet"))[-1]
df = pd.read_parquet(file)

df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0)

def label(row):
    score = row["anomaly_score"]
    val = row["value"]

    if score < 0.3:
        return "monitor"

    if row["resource_type"] == "ec2":
        if row["metric_name"] == "CPUUtilization":
            if val < 15:
                return "scale_down"
            if val > 75:
                return "investigate_high_compute"
        if row["metric_name"] in ["NetworkIn", "NetworkOut", "DiskReadOps", "DiskWriteOps"]:
            return "investigate_network"

    if row["resource_type"] == "lambda":
        if row["metric_name"] == "Duration":
            return "optimize_lambda"
        if row["metric_name"] == "Invocations":
            return "limit_lambda"
        if row["metric_name"] == "Errors" and val > 0:
            return "investigate_lambda_errors"

    if row["resource_type"] == "s3":
        if row["metric_name"] == "BucketSizeBytes" and score >= 0.8:
            return "optimize_s3_lifecycle"
        return "investigate_s3_access"

    return "investigate"

df["label"] = df.apply(label, axis=1)

print("Class Distribution:\n")
print(df["label"].value_counts())
