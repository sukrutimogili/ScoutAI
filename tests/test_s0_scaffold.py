"""
S0 smoke tests: verify the repo scaffold is importable and config loads cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scoutai import __version__
from scoutai.config import ScoutAIConfig, load_config


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def test_package_version_defined() -> None:
    assert __version__ == "0.1.0"


def test_config_yaml_exists() -> None:
    assert CONFIG_PATH.exists(), "config.yaml must exist at project root"


def test_config_loads_successfully() -> None:
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, ScoutAIConfig)


def test_config_schema_version() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.schema_version == "1.0.0"


def test_config_model_roles_defined() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.model_roles.fast_structured.primary
    assert cfg.model_roles.high_context.primary


def test_config_agent_max_iterations() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.agent.max_iterations == 8


def test_config_graph_recursion_limit() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.graph.recursion_limit == 40


def test_config_rubric_weights_sum_to_one() -> None:
    cfg = load_config(CONFIG_PATH)
    total = sum(cfg.rubric.default_weights.values())
    assert abs(total - 1.0) < 0.001, f"Rubric weights sum to {total}, expected 1.0"


def test_config_security_patterns_nonempty() -> None:
    cfg = load_config(CONFIG_PATH)
    assert len(cfg.security.sensitive_attribute_patterns) > 0
    assert len(cfg.security.injection_patterns) > 0


def test_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_directory_layout_exists() -> None:
    """All required package sub-directories must exist after scaffold."""
    root = Path(__file__).parent.parent / "scoutai"
    required_dirs = ["schemas", "graph", "agent", "capabilities", "audit"]
    for d in required_dirs:
        assert (root / d).is_dir(), f"scoutai/{d}/ directory is missing"


def test_env_example_exists() -> None:
    root = Path(__file__).parent.parent
    assert (root / ".env.example").exists()


def test_gitignore_excludes_env() -> None:
    root = Path(__file__).parent.parent
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert ".env.*" in gitignore or ".env*" in gitignore
