from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import argparse
import os
import boto3

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_BUCKET = "sourav-cost-low-20260328"

ec2 = None
cloudwatch = None
lambda_client = None
s3 = None


def init_clients(region_name: str):
    global ec2, cloudwatch, lambda_client, s3
    if ec2 is None:
        ec2 = boto3.client("ec2", region_name=region_name)
    if cloudwatch is None:
        cloudwatch = boto3.client("cloudwatch", region_name=region_name)
    if lambda_client is None:
        lambda_client = boto3.client("lambda", region_name=region_name)
    if s3 is None:
        s3 = boto3.client("s3", region_name=region_name)


def validate_aws_preflight() -> str:
    session = boto3.session.Session()
    region = session.region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError(
            "AWS region is not configured. Set AWS_REGION or AWS_DEFAULT_REGION "
            "(for example: ap-south-1)."
        )

    creds = session.get_credentials()
    if creds is None:
        raise RuntimeError(
            "AWS credentials are not configured. Run `aws configure` or set "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
        )

    return region


def infer_microservice_from_tags(tags: dict, default: str) -> str:
    for key in ["Microservice", "Service", "Application", "App", "Project"]:
        value = tags.get(key)
        if value:
            return str(value)
    return default


def get_ec2_instances():
    data = ec2.describe_instances()
    rows = []

    for reservation in data.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            state = inst.get("State", {}).get("Name")
            if state != "running":
                continue

            tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])} if inst.get("Tags") else {}
            rows.append({
                "instance_id": inst["InstanceId"],
                "name": tags.get("Name", inst["InstanceId"]),
                "instance_type": inst.get("InstanceType", "t3.micro"),
                "microservice": infer_microservice_from_tags(tags, tags.get("Name", inst["InstanceId"])),
            })

    return rows


def get_lambda_functions():
    paginator = lambda_client.get_paginator("list_functions")
    rows = []

    for page in paginator.paginate():
        for fn in page.get("Functions", []):
            rows.append({
                "function_name": fn["FunctionName"],
                "memory_mb": float(fn.get("MemorySize", 128.0)),
                "microservice": fn["FunctionName"],
            })

    return rows


def get_s3_buckets():
    response = s3.list_buckets()
    rows = []
    for bucket in response.get("Buckets", []):
        name = bucket.get("Name")
        if not name:
            continue
        rows.append({"bucket_name": name})
    return rows


def build_ec2_queries(instances):
    queries = []
    query_map = {}

    metrics = [
        ("CPUUtilization", "Average"),
        ("NetworkIn", "Average"),
        ("NetworkOut", "Average"),
    ]

    idx = 0
    for inst in instances:
        for metric_name, stat in metrics:
            qid = f"ec2{idx}"
            idx += 1

            queries.append({
                "Id": qid,
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/EC2",
                        "MetricName": metric_name,
                        "Dimensions": [
                            {"Name": "InstanceId", "Value": inst["instance_id"]}
                        ]
                    },
                    "Period": 60,
                    "Stat": stat
                },
                "ReturnData": True
            })

            query_map[qid] = {
                "resource_type": "ec2",
                "resource_id": inst["instance_id"],
                "resource_name": inst["name"],
                "metric_name": metric_name,
                "instance_type": inst.get("instance_type", "t3.micro"),
                "lambda_memory_mb": None,
                "microservice": inst.get("microservice", inst["name"]),
            }

    return queries, query_map


def build_lambda_queries(functions):
    queries = []
    query_map = {}

    metrics = [
        ("Invocations", "Sum"),
        ("Errors", "Sum"),
        ("Duration", "Average")
    ]

    idx = 0
    for fn in functions:
        for metric_name, stat in metrics:
            qid = f"lam{idx}"
            idx += 1

            queries.append({
                "Id": qid,
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/Lambda",
                        "MetricName": metric_name,
                        "Dimensions": [
                            {"Name": "FunctionName", "Value": fn["function_name"]}
                        ]
                    },
                    "Period": 60,
                    "Stat": stat
                },
                "ReturnData": True
            })

            query_map[qid] = {
                "resource_type": "lambda",
                "resource_id": fn["function_name"],
                "resource_name": fn["function_name"],
                "metric_name": metric_name,
                "instance_type": None,
                "lambda_memory_mb": fn.get("memory_mb", 128.0),
                "microservice": fn.get("microservice", fn["function_name"]),
            }

    return queries, query_map


def build_s3_queries(buckets):
    queries = []
    query_map = {}

    # S3 storage metrics are daily; request metrics require additional per-bucket config.
    metrics = [
        ("BucketSizeBytes", "Average"),
        ("NumberOfObjects", "Average"),
    ]

    idx = 0
    for bucket in buckets:
        bucket_name = bucket["bucket_name"]
        for metric_name, stat in metrics:
            qid = f"s3{idx}"
            idx += 1

            queries.append({
                "Id": qid,
                "MetricStat": {
                    "Metric": {
                        "Namespace": "AWS/S3",
                        "MetricName": metric_name,
                        "Dimensions": [
                            {"Name": "BucketName", "Value": bucket_name},
                            {"Name": "StorageType", "Value": "StandardStorage" if metric_name == "BucketSizeBytes" else "AllStorageTypes"},
                        ],
                    },
                    "Period": 86400,
                    "Stat": stat,
                },
                "ReturnData": True,
            })

            query_map[qid] = {
                "resource_type": "s3",
                "resource_id": bucket_name,
                "resource_name": bucket_name,
                "metric_name": metric_name,
                "instance_type": None,
                "lambda_memory_mb": None,
                "microservice": bucket_name,
            }

    return queries, query_map


def fetch_metric_data(queries, lookback: timedelta):
    if not queries:
        return []

    end_time = datetime.now(timezone.utc)
    start_time = end_time - lookback

    all_results = []
    # CloudWatch GetMetricData supports up to 500 queries per call.
    for i in range(0, len(queries), 500):
        query_chunk = queries[i:i + 500]
        next_token = None

        while True:
            kwargs = {
                "MetricDataQueries": query_chunk,
                "StartTime": start_time,
                "EndTime": end_time,
                "ScanBy": "TimestampAscending"
            }

            if next_token:
                kwargs["NextToken"] = next_token

            response = cloudwatch.get_metric_data(**kwargs)
            all_results.extend(response.get("MetricDataResults", []))
            next_token = response.get("NextToken")

            if not next_token:
                break

    return all_results


def normalize_results(results, query_map):
    rows = []
    seen_ids = set()
    now_utc = datetime.now(timezone.utc)

    for result in results:
        qid = result.get("Id")
        meta = query_map.get(qid, {})
        if qid:
            seen_ids.add(qid)

        timestamps = result.get("Timestamps", [])
        values = result.get("Values", [])
        status_code = result.get("StatusCode", "")

        for ts, val in zip(timestamps, values):
            # S3 metrics are often daily snapshots; represent them at collection time for real-time view.
            effective_ts = now_utc.isoformat() if meta.get("resource_type") == "s3" else ts.isoformat()
            rows.append({
                "collected_at": now_utc.isoformat(),
                "resource_type": meta.get("resource_type"),
                "resource_id": meta.get("resource_id"),
                "resource_name": meta.get("resource_name"),
                "metric_name": meta.get("metric_name"),
                "timestamp": effective_ts,
                "value": val,
                "status_code": status_code,
                "instance_type": meta.get("instance_type"),
                "lambda_memory_mb": meta.get("lambda_memory_mb"),
                "microservice": meta.get("microservice"),
            })

    # Ensure resources still appear when CloudWatch has no recent points.
    now_iso = now_utc.isoformat()
    for qid, meta in query_map.items():
        if qid in seen_ids:
            # If the result existed but had zero datapoints, we still add fallback.
            matched = any(r.get("resource_id") == meta.get("resource_id") and r.get("metric_name") == meta.get("metric_name") for r in rows)
            if matched:
                continue
        rows.append({
            "collected_at": now_iso,
            "resource_type": meta.get("resource_type"),
            "resource_id": meta.get("resource_id"),
            "resource_name": meta.get("resource_name"),
            "metric_name": meta.get("metric_name"),
            "timestamp": now_iso,
            "value": 0.0,
            "status_code": "NoData",
            "instance_type": meta.get("instance_type"),
            "lambda_memory_mb": meta.get("lambda_memory_mb"),
            "microservice": meta.get("microservice"),
        })

    return rows


def main(include_s3: bool = True):
    region = validate_aws_preflight()
    init_clients(region_name=region)
    ec2_instances = get_ec2_instances()
    lambda_functions = get_lambda_functions()
    s3_buckets = get_s3_buckets() if include_s3 else []

    ec2_queries, ec2_query_map = build_ec2_queries(ec2_instances)
    lambda_queries, lambda_query_map = build_lambda_queries(lambda_functions)
    s3_queries, s3_query_map = build_s3_queries(s3_buckets)

    ec2_map = {**ec2_query_map}
    lambda_map = {**lambda_query_map}
    s3_map = {**s3_query_map}

    # EC2 should feel real-time (minute-level windows).
    ec2_results = fetch_metric_data(ec2_queries, lookback=timedelta(minutes=20))
    # Lambda can be sparse/bursty; keep a wider window to retain meaningful anomaly context.
    lambda_results = fetch_metric_data(lambda_queries, lookback=timedelta(hours=24))
    # S3 storage metrics are daily; use a wider lookback and allow slower scheduling.
    s3_results = fetch_metric_data(s3_queries, lookback=timedelta(days=2)) if include_s3 else []

    all_results = ec2_results + lambda_results + s3_results
    query_map = {**ec2_map, **lambda_map, **s3_map}
    normalized_rows = normalize_results(all_results, query_map)

    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "ec2_instances": ec2_instances,
        "lambda_functions": lambda_functions,
        "s3_buckets": s3_buckets,
        "rows": normalized_rows
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_file = RAW_DIR / f"metrics_{timestamp}.json"

    with open(local_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    s3_key = f"raw/metrics_{timestamp}.json"
    s3.upload_file(str(local_file), RAW_BUCKET, s3_key)

    print(f"Collected EC2 instances: {len(ec2_instances)}")
    print(f"Collected Lambda functions: {len(lambda_functions)}")
    print(f"Collected S3 buckets: {len(s3_buckets)}")
    print(f"AWS region: {region}")
    print(f"S3 collection enabled: {include_s3}")
    print(f"Metric rows collected: {len(normalized_rows)}")
    print(f"Local file saved: {local_file}")
    print(f"Uploaded to s3://{RAW_BUCKET}/{s3_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect AWS metrics for the real-time pipeline.")
    parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Skip S3 metric collection for this cycle (useful for fast-lane minute loops).",
    )
    args = parser.parse_args()
    try:
        main(include_s3=not args.skip_s3)
    except RuntimeError as e:
        print(f"Preflight check failed: {e}")
        raise SystemExit(2)
