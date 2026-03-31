from typing import Dict


# Approximate hourly on-demand prices for demo purposes.
# Keep these easy to update.
EC2_HOURLY_PRICES = {
    "t3.micro": 0.0104
}

# Lambda pricing proxies for demo.
# Request price: $0.20 per 1M requests
LAMBDA_REQUEST_PRICE_PER_MILLION = 0.20

# Compute price approximation per GB-second
# Public AWS pricing varies by region; this is a hackathon-friendly proxy.
LAMBDA_GB_SECOND_PRICE = 0.0000166667

# S3 Standard storage proxy: $0.023 per GB-month.
S3_STANDARD_GB_MONTH_PRICE = 0.023


# Action policy:
# - "optimize" actions are allowed to reduce cost.
# - "investigate"/"monitor" actions never modify cost directly.
ACTION_COST_MULTIPLIERS: Dict[str, float] = {
    "scale_down": 0.50,
    "optimize_lambda": 0.60,
    "limit_lambda": 0.70,
    "optimize_s3_lifecycle": 0.75,
}

INCIDENT_OR_MONITOR_ACTIONS = {
    "monitor",
    "investigate",
    "investigate_network",
    "investigate_high_compute",
    "investigate_lambda_errors",
    "investigate_s3_access",
}

# Legacy labels that may still appear from older trained artifacts.
LEGACY_ACTION_ALIASES = {
    "network_spike": "investigate_network",
    "high_compute": "investigate_high_compute",
    "optimize_lambda_execution": "optimize_lambda",
    "limit_repeated_serverless_invocations": "limit_lambda",
    "s3_spike": "investigate_s3_access",
}

SERVICE_ALLOWED_ACTIONS = {
    "ec2": {"monitor", "investigate", "investigate_network", "investigate_high_compute", "scale_down"},
    "lambda": {"monitor", "investigate", "investigate_lambda_errors", "optimize_lambda", "limit_lambda"},
    "s3": {"monitor", "investigate", "investigate_s3_access", "optimize_s3_lifecycle"},
}


def get_ec2_hourly_cost(instance_type: str) -> float:
    return float(EC2_HOURLY_PRICES.get(instance_type, 0.0104))


def estimate_lambda_request_cost(invocations: float) -> float:
    return float(invocations) / 1_000_000.0 * LAMBDA_REQUEST_PRICE_PER_MILLION


def estimate_lambda_compute_cost(avg_duration_ms: float, memory_mb: float, invocations: float) -> float:
    gb_seconds = (memory_mb / 1024.0) * (avg_duration_ms / 1000.0) * float(invocations)
    return gb_seconds * LAMBDA_GB_SECOND_PRICE


def estimate_lambda_total_cost(avg_duration_ms: float, memory_mb: float, invocations: float) -> float:
    return estimate_lambda_request_cost(invocations) + estimate_lambda_compute_cost(
        avg_duration_ms=avg_duration_ms,
        memory_mb=memory_mb,
        invocations=invocations
    )


def estimate_s3_hourly_storage_cost_from_bytes(storage_bytes: float) -> float:
    gb = float(storage_bytes) / (1024.0 ** 3)
    monthly = gb * S3_STANDARD_GB_MONTH_PRICE
    return monthly / (30.0 * 24.0)


def normalize_action(action: str) -> str:
    raw = str(action or "").strip().lower()
    if raw in LEGACY_ACTION_ALIASES:
        return LEGACY_ACTION_ALIASES[raw]
    if raw in ACTION_COST_MULTIPLIERS or raw in INCIDENT_OR_MONITOR_ACTIONS:
        return raw
    return "investigate"


def is_optimization_action(action: str) -> bool:
    return normalize_action(action) in ACTION_COST_MULTIPLIERS


def enforce_service_action_policy(service: str, action: str) -> str:
    svc = str(service or "").strip().lower()
    normalized = normalize_action(action)
    allowed = SERVICE_ALLOWED_ACTIONS.get(svc)
    if not allowed:
        return "investigate"
    if normalized in allowed:
        return normalized
    # If model predicts cross-service action (e.g. scale_down for lambda), downgrade safely.
    return "monitor"


def estimate_post_action_cost(service, current_cost, action):
    _ = service  # Kept for compatibility/future service-specific policies.
    normalized = normalize_action(action)
    before = float(current_cost)

    if normalized in ACTION_COST_MULTIPLIERS:
        after = before * ACTION_COST_MULTIPLIERS[normalized]
        # Evaluation safety: optimization should never increase cost.
        return min(after, before)

    # Investigate/monitor actions are not optimization moves.
    return before


def estimate_savings(before, after):
    return round(before - after, 6)


def action_status_from_confidence(confidence_score: float) -> str:
    if confidence_score >= 0.95:
        return "executed"
    if confidence_score >= 0.85:
        return "auto_fix_with_rollback"
    if confidence_score >= 0.70:
        return "recommendation_with_approval"
    return "monitor_only"


def severity_from_score(score: float) -> str:
    if score >= 1.2:
        return "critical"
    if score >= 0.8:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"
