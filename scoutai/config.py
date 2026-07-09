"""
Configuration loader for ScoutAI.

Loads config.yaml and resolves environment variables for secrets.
All configuration is read once at startup and injected via dependency injection —
never accessed via global state in business logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


# Load .env at import time so os.getenv picks up values.
load_dotenv()

_CONFIG_PATH_DEFAULT = Path(__file__).parent.parent / "config.yaml"


# ── Sub-models ────────────────────────────────────────────────────────────────


class ModelRoleConfig(BaseModel):
    """Configuration for a single model role (fast_structured or high_context)."""

    primary: str
    fallback: str
    fallback_2: str | None = None
    provider_env_vars: dict[str, str] = Field(default_factory=dict)


class ModelRolesConfig(BaseModel):
    fast_structured: ModelRoleConfig
    high_context: ModelRoleConfig


class RetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_ms: list[int] = Field(default_factory=lambda: [250, 500, 1000])
    timeout_seconds: int = 30


class CircuitBreakerConfig(BaseModel):
    consecutive_failures_threshold: int = 5
    open_window_seconds: int = 60


class AgentConfig(BaseModel):
    max_iterations: int = 8
    force_finalize_recommendation: str = "hold"


class GraphConfig(BaseModel):
    recursion_limit: int = 40
    interview_rounds_cap: int = 1


class RubricConfig(BaseModel):
    config_version: str = "1.0.0"
    default_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "required_skills": 0.40,
            "experience": 0.30,
            "education": 0.15,
            "soft_skills": 0.15,
        }
    )
    unknown_confidence_score_floor: float = 0.0
    unknown_confidence_score_cap: float = 70.0

    @field_validator("default_weights")
    @classmethod
    def weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"rubric.default_weights must sum to 1.0, got {total:.4f}")
        return v


class SecurityConfig(BaseModel):
    sensitive_attribute_patterns: list[str] = Field(default_factory=list)
    injection_patterns: list[str] = Field(default_factory=list)
    max_jd_chars: int = 10000
    max_resume_chars: int = 15000
    secret_redaction_patterns: list[str] = Field(default_factory=list)


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_seconds: int = 86400
    backend: str = "memory"


class SchedulingConfig(BaseModel):
    backend: str = "mock"
    calendar_env_var: str = "GOOGLE_CALENDAR_CREDENTIALS_JSON"


class AuditConfig(BaseModel):
    backend: str = "jsonl"
    output_dir: str = "audit_logs"
    filename_template: str = "run_{run_id}.jsonl"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    include_node_timings: bool = True
    include_model_usage: bool = True


# ── Root config ───────────────────────────────────────────────────────────────


class ScoutAIConfig(BaseModel):
    """Root configuration object for ScoutAI.

    Loaded from config.yaml. Secrets (API keys) are NOT stored here;
    they are resolved from environment variables at runtime by the model router.
    """

    schema_version: str = "1.0.0"
    model_roles: ModelRolesConfig
    retry: RetryConfig = Field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    rubric: RubricConfig = Field(default_factory=RubricConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    tool_versions: dict[str, str] = Field(default_factory=dict)


def load_config(path: Path | str | None = None) -> ScoutAIConfig:
    """Load and validate ScoutAI configuration from a YAML file.

    Args:
        path: Path to config.yaml. Defaults to the bundled config.yaml at the
              project root. Can also be overridden via the SCOUTAI_CONFIG_PATH
              environment variable.

    Returns:
        A fully validated ScoutAIConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the config is structurally invalid.
    """
    if path is None:
        env_path = os.getenv("SCOUTAI_CONFIG_PATH")
        path = Path(env_path) if env_path else _CONFIG_PATH_DEFAULT

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return ScoutAIConfig.model_validate(raw)
