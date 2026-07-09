"""
S4 tests: extract_role_requirements, generate_rubric, and SessionCache.

Tests cover:
- Happy path with mocked LLM response
- Schema validation: missing fields, empty criteria
- Caching: cache hit avoids LLM call, cache key includes version strings
- Cache miss after version change
- Cache expiry
- Empty JD raises ValueError
- JD truncation warning
- Fail closed on LLM failure (ProviderError raised)
- Rubric has unique criterion names
- Schema repair-retry on first parse failure
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, call, patch

import pytest

from scoutai.capabilities.role_requirements import (
    SessionCache,
    _build_role_profile,
    _build_rubric,
    _parse_json_response,
    _parse_role_profile_response,
    _parse_rubric_response,
    _role_profile_cache_key,
    _rubric_cache_key,
    _sha256,
    extract_role_requirements,
    generate_rubric,
)
from scoutai.capabilities.model_router import ModelRouter, ProviderError
from scoutai.config import load_config
from scoutai.schemas import RequiredSkill, RoleProfile

CONFIG_PATH = "config.yaml"

SAMPLE_JD = """
We are looking for a Senior Python Engineer to join our platform team.

Requirements:
- 5+ years Python experience (required)
- Strong distributed systems knowledge (required)
- Experience with Kubernetes and Docker (required)
- PostgreSQL or equivalent database experience (required)
- Experience with FastAPI or similar frameworks (nice-to-have)
- Bachelor's degree in Computer Science or related field (preferred)

Responsibilities:
- Design and implement backend microservices
- Collaborate with senior engineers on architecture
- Mentor junior team members
"""

SAMPLE_ROLE_PROFILE_JSON = {
    "title": "Senior Python Engineer",
    "summary": "Backend engineer for distributed systems",
    "required_skills": [
        {"name": "Python", "description": "5+ years", "required": True, "weight": 5.0},
        {"name": "Distributed Systems", "description": "Strong knowledge", "required": True, "weight": 4.0},
        {"name": "Kubernetes", "description": "Container orchestration", "required": True, "weight": 3.0},
    ],
    "experience_requirements": ["5+ years Python", "Distributed systems experience"],
    "education_requirements": ["Bachelor's in CS or related"],
    "soft_skills": ["Mentoring", "Collaboration"],
}

SAMPLE_RUBRIC_JSON = {
    "criteria": [
        {
            "name": "Python Proficiency",
            "description": "5+ years production Python",
            "weight": 5.0,
            "category": "required_skills",
            "examples_of_sufficient_evidence": ["5+ years listed", "production Python code"],
        },
        {
            "name": "Distributed Systems",
            "description": "Microservices, message queues, consensus",
            "weight": 4.0,
            "category": "required_skills",
            "examples_of_sufficient_evidence": ["Led distributed system design"],
        },
    ]
}


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def mock_router(config):
    return ModelRouter(config)


@pytest.fixture
def sample_role_profile(config):
    return _build_role_profile(SAMPLE_ROLE_PROFILE_JSON, _sha256(SAMPLE_JD))


# ── SessionCache ──────────────────────────────────────────────────────────────


class TestSessionCache:
    def test_set_and_get(self):
        cache = SessionCache(ttl_seconds=60)
        cache.set("key1", {"value": 42})
        assert cache.get("key1") == {"value": 42}

    def test_miss_returns_none(self):
        cache = SessionCache(ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_disabled_cache_always_misses(self):
        cache = SessionCache(ttl_seconds=60, enabled=False)
        cache.set("key1", "value")
        assert cache.get("key1") is None

    def test_ttl_expiry(self):
        cache = SessionCache(ttl_seconds=1)
        cache.set("key1", "value")
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_clear(self):
        cache = SessionCache(ttl_seconds=60)
        cache.set("k1", 1)
        cache.set("k2", 2)
        assert cache.size() == 2
        cache.clear()
        assert cache.size() == 0

    def test_overwrite(self):
        cache = SessionCache(ttl_seconds=60)
        cache.set("key", "old")
        cache.set("key", "new")
        assert cache.get("key") == "new"

    def test_cache_key_includes_prompt_version(self):
        k1 = _role_profile_cache_key("hash123", "1.0.0", "1.0.0")
        k2 = _role_profile_cache_key("hash123", "1.1.0", "1.0.0")
        assert k1 != k2, "Different prompt versions must produce different cache keys"

    def test_cache_key_includes_schema_version(self):
        k1 = _role_profile_cache_key("hash123", "1.0.0", "1.0.0")
        k2 = _role_profile_cache_key("hash123", "1.0.0", "2.0.0")
        assert k1 != k2

    def test_rubric_cache_key_includes_rubric_config_version(self):
        k1 = _rubric_cache_key("hash123", "1.0.0", "1.0.0", "1.0.0")
        k2 = _rubric_cache_key("hash123", "2.0.0", "1.0.0", "1.0.0")
        assert k1 != k2


# ── JSON parsing helpers ──────────────────────────────────────────────────────


class TestParseJsonResponse:
    def test_valid_json(self):
        raw = '{"title": "Engineer", "skills": []}'
        result = _parse_json_response(raw, "test")
        assert result["title"] == "Engineer"

    def test_json_with_code_fence(self):
        raw = '```json\n{"title": "Engineer"}\n```'
        result = _parse_json_response(raw, "test")
        assert result["title"] == "Engineer"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object"):
            _parse_json_response("plain text response", "test")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            _parse_json_response("{broken: json}", "test")


class TestParseRoleProfileResponse:
    def test_valid_response(self):
        raw = json.dumps(SAMPLE_ROLE_PROFILE_JSON)
        result = _parse_role_profile_response(raw)
        assert result["title"] == "Senior Python Engineer"

    def test_missing_title_raises(self):
        raw = json.dumps({"summary": "no title here"})
        with pytest.raises(ValueError, match="missing fields"):
            _parse_role_profile_response(raw)


class TestParseRubricResponse:
    def test_valid_response(self):
        raw = json.dumps(SAMPLE_RUBRIC_JSON)
        result = _parse_rubric_response(raw)
        assert len(result["criteria"]) == 2

    def test_missing_criteria_raises(self):
        raw = json.dumps({"something_else": []})
        with pytest.raises(ValueError, match="missing 'criteria'"):
            _parse_rubric_response(raw)

    def test_empty_criteria_raises(self):
        raw = json.dumps({"criteria": []})
        with pytest.raises(ValueError, match="non-empty"):
            _parse_rubric_response(raw)


# ── _build_role_profile ───────────────────────────────────────────────────────


class TestBuildRoleProfile:
    def test_builds_valid_profile(self):
        rp = _build_role_profile(SAMPLE_ROLE_PROFILE_JSON, "abc123")
        assert rp.title == "Senior Python Engineer"
        assert len(rp.required_skills) == 3

    def test_malformed_skill_is_skipped(self):
        data = dict(SAMPLE_ROLE_PROFILE_JSON)
        data["required_skills"] = [
            {"name": "Python", "weight": 5.0, "required": True},
            "not_a_dict",  # malformed — should be skipped
        ]
        rp = _build_role_profile(data, "abc123")
        assert len(rp.required_skills) == 1  # only the valid one

    def test_jd_hash_preserved(self):
        rp = _build_role_profile(SAMPLE_ROLE_PROFILE_JSON, "myhash")
        assert rp.jd_hash == "myhash"


# ── _build_rubric ─────────────────────────────────────────────────────────────


class TestBuildRubric:
    def test_builds_valid_rubric(self):
        rubric = _build_rubric(SAMPLE_RUBRIC_JSON, "abc123", "1.0.0")
        assert len(rubric.criteria) == 2
        assert rubric.criteria[0].name == "Python Proficiency"

    def test_duplicate_criteria_deduplicated(self):
        data = {
            "criteria": [
                {"name": "Python", "weight": 1.0},
                {"name": "Python", "weight": 2.0},  # duplicate — should be skipped
            ]
        }
        rubric = _build_rubric(data, "hash", "1.0.0")
        assert len(rubric.criteria) == 1

    def test_empty_result_raises(self):
        data = {"criteria": [{"not_a_name": "x"}]}
        with pytest.raises(ValueError, match="No valid criteria"):
            _build_rubric(data, "hash", "1.0.0")


# ── extract_role_requirements (mocked LLM) ───────────────────────────────────


class TestExtractRoleRequirements:
    def _make_mock_model(self, response_data: dict) -> MagicMock:
        mock = MagicMock()
        resp = MagicMock()
        resp.content = json.dumps(response_data)
        mock.invoke.return_value = resp
        return mock

    def test_happy_path(self, config, mock_router):
        mock_model = self._make_mock_model(SAMPLE_ROLE_PROFILE_JSON)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            rp = extract_role_requirements(SAMPLE_JD, config, mock_router)
        assert rp.title == "Senior Python Engineer"
        assert len(rp.required_skills) == 3

    def test_empty_jd_raises(self, config, mock_router):
        with pytest.raises(ValueError, match="must not be empty"):
            extract_role_requirements("", config, mock_router)

    def test_whitespace_only_jd_raises(self, config, mock_router):
        with pytest.raises(ValueError):
            extract_role_requirements("   \n  ", config, mock_router)

    def test_jd_truncated_if_too_long(self, config, mock_router, caplog):
        import logging
        mock_model = self._make_mock_model(SAMPLE_ROLE_PROFILE_JSON)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            with caplog.at_level(logging.WARNING):
                extract_role_requirements("A" * (config.security.max_jd_chars + 100), config, mock_router)
        assert "truncated" in caplog.text.lower()

    def test_cache_hit_avoids_llm_call(self, config, mock_router):
        """Cache hit must not trigger a second LLM call (§9, §8)."""
        cache = SessionCache(ttl_seconds=60)
        mock_model = self._make_mock_model(SAMPLE_ROLE_PROFILE_JSON)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            # First call — cache miss
            rp1 = extract_role_requirements(SAMPLE_JD, config, mock_router, cache=cache)
            # Second call — should be a cache hit
            rp2 = extract_role_requirements(SAMPLE_JD, config, mock_router, cache=cache)

        assert mock_model.invoke.call_count == 1, "LLM should only be called once — second was a cache hit"
        assert rp1 == rp2

    def test_different_jd_different_cache_entry(self, config, mock_router):
        cache = SessionCache(ttl_seconds=60)
        mock_model = self._make_mock_model(SAMPLE_ROLE_PROFILE_JSON)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            extract_role_requirements(SAMPLE_JD, config, mock_router, cache=cache)
            extract_role_requirements(SAMPLE_JD + " EXTRA TEXT", config, mock_router, cache=cache)
        assert mock_model.invoke.call_count == 2

    def test_llm_failure_raises_provider_error(self, config, mock_router):
        """§6, Principle #10: LLM failure must raise ProviderError (fail closed)."""
        mock_model = MagicMock()
        mock_model.invoke.side_effect = TimeoutError("timeout")
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            with pytest.raises(ProviderError):
                extract_role_requirements(SAMPLE_JD, config, mock_router)

    def test_schema_invalid_output_retried(self, config, mock_router):
        """§7.3: Invalid output gets one repair retry before failing."""
        mock_model = MagicMock()
        bad_resp = MagicMock()
        bad_resp.content = "Not valid JSON at all"
        good_resp = MagicMock()
        good_resp.content = json.dumps(SAMPLE_ROLE_PROFILE_JSON)
        # First call: bad; second call: good
        mock_model.invoke.side_effect = [bad_resp, good_resp]
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            rp = extract_role_requirements(SAMPLE_JD, config, mock_router)
        assert rp.title == "Senior Python Engineer"
        assert mock_model.invoke.call_count == 2


# ── generate_rubric (mocked LLM) ─────────────────────────────────────────────


class TestGenerateRubric:
    def _make_mock_model(self, response_data: dict) -> MagicMock:
        mock = MagicMock()
        resp = MagicMock()
        resp.content = json.dumps(response_data)
        mock.invoke.return_value = resp
        return mock

    def test_happy_path(self, config, mock_router, sample_role_profile):
        mock_model = self._make_mock_model(SAMPLE_RUBRIC_JSON)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            rubric = generate_rubric(sample_role_profile, config, mock_router)
        assert len(rubric.criteria) == 2
        assert rubric.jd_hash == sample_role_profile.jd_hash

    def test_cache_hit_avoids_llm_call(self, config, mock_router, sample_role_profile):
        """§9: Rubric should be cached per (jd_hash, rubric_config_version) (§8 cost)."""
        cache = SessionCache(ttl_seconds=60)
        mock_model = self._make_mock_model(SAMPLE_RUBRIC_JSON)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            r1 = generate_rubric(sample_role_profile, config, mock_router, cache=cache)
            r2 = generate_rubric(sample_role_profile, config, mock_router, cache=cache)
        assert mock_model.invoke.call_count == 1, "Rubric LLM should only be called once — second was cache hit"
        assert r1 == r2

    def test_rubric_config_version_change_invalidates_cache(self, config, mock_router, sample_role_profile):
        """A rubric_config_version change must force a cache miss (§9)."""
        cache = SessionCache(ttl_seconds=60)
        mock_model = self._make_mock_model(SAMPLE_RUBRIC_JSON)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            generate_rubric(sample_role_profile, config, mock_router, cache=cache)

        # Modify config to simulate a rubric_config_version bump
        import copy
        config2 = copy.deepcopy(config)
        config2.rubric.config_version = "2.0.0"

        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            generate_rubric(sample_role_profile, config2, mock_router, cache=cache)

        assert mock_model.invoke.call_count == 2, "Config version change must invalidate cache"

    def test_llm_failure_raises_provider_error(self, config, mock_router, sample_role_profile):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = ConnectionError("connection refused")
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            with pytest.raises(ProviderError):
                generate_rubric(sample_role_profile, config, mock_router)

    def test_rubric_criteria_unique(self, config, mock_router, sample_role_profile):
        data = {
            "criteria": [
                {"name": "Python", "weight": 1.0},
                {"name": "Python", "weight": 2.0},
            ]
        }
        mock_model = self._make_mock_model(data)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            rubric = generate_rubric(sample_role_profile, config, mock_router)
        names = [c.name for c in rubric.criteria]
        assert len(names) == len(set(names)), "Criteria names must be unique"
