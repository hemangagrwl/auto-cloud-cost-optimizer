from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def get_latest_anomaly_file():
    files = sorted(PROCESSED_DIR.glob("anomaly_scores_*.parquet"))
    if not files:
        raise FileNotFoundError("No anomaly files")
    return files[-1]

def main():
    df = pd.read_parquet(get_latest_anomaly_file())

    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0)

    df["is_ec2"] = (df["resource_type"] == "ec2").astype(int)
    df["is_lambda"] = (df["resource_type"] == "lambda").astype(int)
    df["is_s3"] = (df["resource_type"] == "s3").astype(int)

    df["cost"] = np.select(
        [df["resource_type"] == "ec2", df["resource_type"] == "lambda", df["resource_type"] == "s3"],
        [0.0104, 0.0001, 0.00005],
        default=0.0001,
    )

    # Rule-assisted labels used as bootstrap targets for the ML classifier.
    # Keep optimization labels strictly cost-reduction-oriented.
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

    X = df[[
    "value",
    "anomaly_score",
    "rolling_mean_3",
    "rolling_std_3",
    "pct_change",
    "is_ec2",
    "is_lambda",
    "is_s3",
    "cost"
	]].fillna(0)
    y = df["label"]

    classes = sorted(y.unique())
    label_map = {c: i for i, c in enumerate(classes)}
    inv_map = {i: c for c, i in label_map.items()}

    y_enc = y.map(label_map)

    model = HistGradientBoostingClassifier()
    model.fit(X, y_enc)

    joblib.dump(model, MODELS_DIR / "model.joblib")
    joblib.dump(inv_map, MODELS_DIR / "labels.joblib")

    print("Model trained successfully")
    print("Classes:", classes)

if __name__ == "__main__":
    main()
