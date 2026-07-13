"""
Mock data for ScoutAI Streamlit UI.

All field names match the real backend schemas (§5) so swapping to real data later
is a data-source change only, never a UI rewrite.
"""

from __future__ import annotations

from typing import Any

# ── Mock GraphState ───────────────────────────────────────────────────────────

MOCK_RUN_ID = "run_abc123"
MOCK_RUN_NAME = "Senior Backend Engineer – Q3"

MOCK_CANDIDATES: list[dict[str, Any]] = [
    {
        "candidate_id": "c001",
        "recommendation": "strong_interview",
        "finalized": True,
        "interview_rounds": 0,
        "scorecard": {"system_design": 92, "python": 88, "aws": 85, "leadership": 78},
        "capabilities": {
            "system_design": {"score": 92, "confidence": "high"},
            "python": {"score": 88, "confidence": "high"},
            "aws": {"score": 85, "confidence": "medium"},
            "leadership": {"score": 78, "confidence": "medium"},
        },
        "evidence_buckets": {
            "system_design": [
                "Designed and led migration of monolith to microservices (12 services)",
                "Architected real-time data pipeline handling 50k events/sec",
            ],
            "python": [
                "8 years professional Python experience",
                "Core contributor to open-source async framework",
            ],
            "aws": [
                "AWS Solutions Architect certification",
                "Built CI/CD pipeline on ECS + Fargate",
            ],
        },
        "injection_flag": False,
        "leakage_flag": False,
        "remaining_uncertainties": [],
        "rationale": "Strong match across all criteria. Deep system design experience and proven Python expertise. AWS certification verified.",
        "strengths": ["System design", "Python expertise", "AWS certification"],
    },
    {
        "candidate_id": "c002",
        "recommendation": "interview",
        "finalized": True,
        "interview_rounds": 1,
        "scorecard": {"system_design": 65, "python": 82, "aws": 70, "leadership": 60},
        "capabilities": {
            "system_design": {"score": 65, "confidence": "medium"},
            "python": {"score": 82, "confidence": "high"},
            "aws": {"score": 70, "confidence": "low"},
            "leadership": {"score": 60, "confidence": "low"},
        },
        "evidence_buckets": {
            "system_design": [
                "Designed REST API for payment processing system",
                "Some experience with distributed systems",
            ],
            "python": [
                "5 years professional Python experience",
                "Built Django-based web applications",
            ],
            "aws": [
                "Basic familiarity with EC2 and S3",
                "No AWS certifications",
            ],
        },
        "injection_flag": False,
        "leakage_flag": False,
        "remaining_uncertainties": ["aws_depth", "leadership_scope"],
        "rationale": "Strong Python skills. System design experience is moderate. AWS depth is uncertain — recommend interview to assess.",
        "strengths": ["Python proficiency", "API design"],
    },
    {
        "candidate_id": "c003",
        "recommendation": "reject",
        "finalized": True,
        "interview_rounds": 0,
        "scorecard": {"system_design": 35, "python": 45, "aws": 20, "leadership": 40},
        "capabilities": {
            "system_design": {"score": 35, "confidence": "high"},
            "python": {"score": 45, "confidence": "medium"},
            "aws": {"score": 20, "confidence": "high"},
            "leadership": {"score": 40, "confidence": "medium"},
        },
        "evidence_buckets": {
            "system_design": [
                "Junior-level experience with monolithic applications",
            ],
            "python": [
                "2 years Python experience",
                "Basic scripting only",
            ],
            "aws": [
                "No AWS experience documented",
            ],
        },
        "injection_flag": False,
        "leakage_flag": False,
        "remaining_uncertainties": ["system_design", "python_depth", "aws", "leadership"],
        "rationale": "Does not meet minimum requirements. Insufficient experience across all criteria.",
        "strengths": [],
    },
    {
        "candidate_id": "c004",
        "recommendation": "interview",
        "finalized": True,
        "interview_rounds": 0,
        "scorecard": {"system_design": 72, "python": 75, "aws": 68, "leadership": 70},
        "capabilities": {
            "system_design": {"score": 72, "confidence": "medium"},
            "python": {"score": 75, "confidence": "high"},
            "aws": {"score": 68, "confidence": "medium"},
            "leadership": {"score": 70, "confidence": "medium"},
        },
        "evidence_buckets": {
            "system_design": [
                "Led team of 5 engineers on microservices migration",
                "Event-driven architecture experience with Kafka",
            ],
            "python": [
                "6 years Python experience",
                "Flask and FastAPI expertise",
            ],
            "aws": [
                "AWS Developer Associate certification",
                "Serverless applications with Lambda + DynamoDB",
            ],
        },
        "injection_flag": False,
        "leakage_flag": False,
        "remaining_uncertainties": ["leadership_scale"],
        "rationale": "Solid across the board. Leadership experience is at team level — unclear if ready for org-wide scope.",
        "strengths": ["Python", "System design", "AWS certified"],
    },
    {
        "candidate_id": "c005",
        "recommendation": "strong_interview",
        "finalized": True,
        "interview_rounds": 0,
        "scorecard": {"system_design": 90, "python": 85, "aws": 80, "leadership": 82},
        "capabilities": {
            "system_design": {"score": 90, "confidence": "high"},
            "python": {"score": 85, "confidence": "high"},
            "aws": {"score": 80, "confidence": "medium"},
            "leadership": {"score": 82, "confidence": "high"},
        },
        "evidence_buckets": {
            "system_design": [
                "Architected multi-region disaster recovery system",
                "Designed system handling 100k concurrent users",
            ],
            "python": [
                "7 years Python experience",
                "Published PyPI packages with 5k+ downloads",
            ],
            "aws": [
                "AWS Professional Solutions Architect certification",
                "Cost optimization expertise — reduced infra spend by 40%",
            ],
        },
        "injection_flag": False,
        "leakage_flag": False,
        "remaining_uncertainties": [],
        "rationale": "Top-tier candidate. Deep expertise across all criteria with verifiable certifications and published work.",
        "strengths": ["System design", "Python", "AWS certification", "Leadership"],
    },
]

MOCK_SHORTLIST: list[dict[str, Any]] = [
    {
        "candidate": "c001",
        "recommendation": "strong_interview",
        "weighted_score": 88.0,  # 0-100 scale, matching ShortlistEntry schema
        "confidence_summary": {"high": 3, "medium": 1, "low": 0},
        "strengths": ["System design", "Python expertise", "AWS certification"],
        "remaining_uncertainties": [],
        "evidence_refs": ["ev-001", "ev-002", "ev-003"],
    },
    {
        "candidate": "c005",
        "recommendation": "strong_interview",
        "weighted_score": 85.0,  # 0-100 scale, matching ShortlistEntry schema
        "confidence_summary": {"high": 3, "medium": 1, "low": 0},
        "strengths": ["System design", "Python", "AWS certification", "Leadership"],
        "remaining_uncertainties": [],
        "evidence_refs": ["ev-011", "ev-012", "ev-013"],
    },
    {
        "candidate": "c002",
        "recommendation": "interview",
        "weighted_score": 68.0,  # 0-100 scale, matching ShortlistEntry schema
        "confidence_summary": {"high": 1, "medium": 1, "low": 2},
        "strengths": ["Python proficiency", "API design"],
        "remaining_uncertainties": ["aws_depth", "leadership_scope"],
        "evidence_refs": ["ev-004", "ev-005"],
    },
    {
        "candidate": "c004",
        "recommendation": "interview",
        "weighted_score": 72.0,  # 0-100 scale, matching ShortlistEntry schema
        "confidence_summary": {"high": 1, "medium": 3, "low": 0},
        "strengths": ["Python", "System design", "AWS certified"],
        "remaining_uncertainties": ["leadership_scale"],
        "evidence_refs": ["ev-008", "ev-009", "ev-010"],
    },
]

MOCK_BIAS_REPORTS: list[dict[str, Any]] = [
    {
        "candidate_a": "c001",
        "candidate_b": "c003",
        "severity": "low",
        "indicators": ["gender_parity"],
        "recommendation": "no_action",
        "rationale": "No significant bias detected. Score differential is consistent with evidence quality.",
    },
]

MOCK_INITIAL_STATE: dict[str, Any] = {
    "run_id": MOCK_RUN_ID,
    "step_count": 42,
    "run_name": MOCK_RUN_NAME,
    "jd": "Senior Backend Engineer with strong system design, Python, and AWS experience. Leadership skills preferred.",
    "candidates": MOCK_CANDIDATES,
    "shortlist": MOCK_SHORTLIST,
    "bias_reports": MOCK_BIAS_REPORTS,
    "trajectory": [],
}


def get_recommendation_tint(recommendation: str) -> str:
    """Return the tint background color for a recommendation status pill."""
    tints = {
        "strong_interview": "#E9F5EC",
        "interview": "#EFEFEF",
        "reject": "#FBEAE9",
        "hold": "#FBF3DC",
    }
    return tints.get(recommendation, "#EFEFEF")


def get_recommendation_label(recommendation: str) -> str:
    """Return the human-readable label for a recommendation."""
    labels = {
        "strong_interview": "Strong Shortlist",
        "interview": "Interview",
        "reject": "Reject",
        "hold": "Needs Review",
    }
    return labels.get(recommendation, recommendation.replace("_", " ").title())