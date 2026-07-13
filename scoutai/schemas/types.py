"""
Pydantic schemas for all ScoutAI data types.

This module is the canonical source for every schema used throughout the system.
All schemas are validated at construction time. External inputs (LLM responses,
file contents) must be coerced through these models before being written to graph
state — never passed around as raw dicts.

Spec reference: §5 (State Schema).
ADR references: ADR-1 (Confidence enum), ADR-3 (interview_rounds Literal[0,1]),
                ADR-6 (Confidence lives on CapabilityAssessment), ADR-8 (TrajectoryEntry
                auto-appended, never agent-written).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enum types (§5) ───────────────────────────────────────────────────────────

Confidence = Literal["unknown", "low", "medium", "high"]
Relevance = Literal["high", "medium", "low"]
Recommendation = Literal["reject", "hold", "interview", "strong_interview"]
VerificationVerdict = Literal["sufficient", "insufficient"]
TrajectoryStatus = Literal["success", "retried", "failed_closed", "escalated"]
ModelRole = Literal["fast_structured", "high_context"]


# ── Evidence & scoring types (§5) ─────────────────────────────────────────────


class JDRelevance(BaseModel):
    """Relevance of a piece of evidence to the job description.

    Per ADR-1 spirit: 'rationale' is required so this is reviewable, not just a number.
    """

    level: Relevance
    rationale: str = Field(
        min_length=1,
        description="Short justification for the relevance level — reviewable, not a bare score.",
    )


class EvidenceItem(BaseModel):
    """A single piece of evidence extracted from a résumé or interview answer.

    Each item is source-tagged and JD-aligned so every claim is traceable.
    """

    value: str = Field(min_length=1, description="The extracted evidence text.")
    source: str = Field(
        min_length=1,
        description='Source tag, e.g. "Resume:Projects", "Interview:Q1".',
    )
    jd_relevance: JDRelevance


class CapabilityAssessment(BaseModel):
    """Assessment of a single capability dimension.

    Confidence is an enum (ADR-1) — never a float. evidence_refs must trace back
    to EvidenceItem.source values in the candidate's evidence_buckets.
    """

    confidence: Confidence
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="References to EvidenceItem.source values that support this assessment.",
    )


class ScoreEntry(BaseModel):
    """Scored assessment of a single rubric criterion.

    score is normalized 0-100:
      - 0   = no evidence supports this criterion at all
      - 100 = criterion fully and directly evidenced

    Per §12 Open Items: weighted sum aggregation. Any criterion at 'unknown' confidence
    caps the maximum achievable weighted_score (see RubricConfig.unknown_confidence_score_cap).
    """

    score: float = Field(
        ge=0.0,
        le=100.0,
        description="Normalized 0-100. 0 = no supporting evidence; 100 = fully evidenced.",
    )
    confidence: Confidence
    evidence_refs: list[str] = Field(default_factory=list)


# ── Candidate state (§5) ──────────────────────────────────────────────────────


class CandidateState(BaseModel):
    """Per-candidate evaluation state, carried through the candidate_agent loop.

    Notes:
    - resume_text: the raw (pre-sanitization) text. Dropped from working memory after
      sanitization per §7.6 — retained here only until screen_resume completes.
    - sanitized_resume: identity-scrubbed text that flows into evidence/scoring.
    - interview_rounds is Literal[0, 1] — enforced both in schema and by the agent
      harness (ADR-3).
    - recommendation/rationale/slot are None until finalize_candidate writes them.
    """

    resume_text: str = Field(default="", description="Raw résumé text (pre-sanitization).")
    injection_flag: bool = Field(default=False)
    leakage_flag: bool = Field(default=False)
    sanitized_resume: str = Field(default="", description="Identity-scrubbed résumé text.")
    evidence_buckets: list[EvidenceItem] = Field(default_factory=list)
    capabilities: dict[str, CapabilityAssessment] = Field(default_factory=dict)
    scorecard: dict[str, ScoreEntry] = Field(default_factory=dict)
    verification_verdict: Optional[VerificationVerdict] = Field(default=None)
    interview_rounds: Literal[0, 1] = Field(
        default=0,
        description="How many interview clarification rounds have been used. Max 1 (ADR-3).",
    )
    qa_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Log of Q&A exchanges from ask_candidate calls.",
    )
    recommendation: Optional[Recommendation] = Field(default=None)
    rationale: Optional[str] = Field(default=None)
    slot: Optional[dict[str, Any]] = Field(default=None)

    # Internal tracking — set by graph, not agent
    candidate_id: str = Field(
        default="",
        description="Unique identifier for this candidate within the run.",
    )
    finalized: bool = Field(
        default=False,
        description="True once finalize_candidate has been called for this candidate.",
    )
    remaining_uncertainties: list[str] = Field(
        default_factory=list,
        description="Unresolved gaps at finalization time (populated on force-finalize too).",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="High-confidence capability names — copied from ShortlistEntry on finalization.",
    )


# ── Shortlist entry (§5) ──────────────────────────────────────────────────────


class ShortlistEntry(BaseModel):
    """Summary entry written to the shortlist when a candidate is finalized.

    weighted_score uses the same 0-100 normalization as ScoreEntry.score.
    All claims must have resolvable evidence_refs (§4.1, compose_decision_summary).
    """

    candidate: str = Field(description="Candidate identifier.")
    recommendation: Recommendation
    weighted_score: float = Field(ge=0.0, le=100.0)
    confidence_summary: dict[str, Confidence] = Field(
        default_factory=dict,
        description="Per-criterion confidence levels at finalization.",
    )
    strengths: list[str] = Field(default_factory=list)
    remaining_uncertainties: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="All evidence references cited for this candidate's recommendation.",
    )

    @field_validator("evidence_refs")
    @classmethod
    def evidence_refs_nonempty_for_positive_recommendation(
        cls, v: list[str], info: Any
    ) -> list[str]:
        """Positive recommendations must have at least one evidence reference."""
        rec = info.data.get("recommendation") if hasattr(info, "data") else None
        if rec in ("interview", "strong_interview") and not v:
            raise ValueError(
                f"evidence_refs must not be empty for recommendation='{rec}'. "
                "Every claim must be backed by evidence (§4.1)."
            )
        return v


# ── Trajectory (audit log) entry (§5, ADR-8) ─────────────────────────────────


class TrajectoryEntry(BaseModel):
    """One row per node execution or agent tool-call in the audit log.

    Auto-appended by the graph runtime (ADR-8). NEVER written by a node or
    the agent itself — this is a non-negotiable compliance requirement.

    input_hash / output_hash are sha256 of the validated payload (hex string).
    """

    node: str = Field(description='e.g. "candidate_agent", "screen_resume"')
    tool_used: Optional[str] = Field(
        default=None,
        description='e.g. "extract_evidence"; None for fixed-node entries.',
    )
    timestamp: str = Field(description="ISO 8601 UTC timestamp.")
    input_hash: str = Field(description="sha256 hex of the validated input payload.")
    output_hash: str = Field(description="sha256 hex of the validated output payload.")
    latency_ms: int = Field(ge=0)
    model: Optional[str] = Field(
        default=None,
        description="Resolved provider/model id actually used, e.g. 'gemini/gemini-2.0-flash'.",
    )
    model_role: Optional[ModelRole] = Field(
        default=None,
        description='"fast_structured" | "high_context" (§6).',
    )
    prompt_version: str = Field(description="Semver of the prompt used.")
    tool_version: str = Field(description="Semver of the tool/capability used.")
    schema_version: str = Field(description="Semver of the output schema version.")
    status: TrajectoryStatus


# ── Role profile and rubric (§4.1) ────────────────────────────────────────────


class RequiredSkill(BaseModel):
    """A single skill or competency extracted from the job description."""

    name: str = Field(min_length=1)
    description: str = Field(default="")
    required: bool = Field(
        default=True,
        description="True = must-have; False = nice-to-have.",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative importance weight for this skill within its category.",
    )


class RoleProfile(BaseModel):
    """Structured role profile extracted from the job description (§4.1).

    Output of extract_role_requirements. Cached per jd_hash (§9).
    """

    title: str = Field(min_length=1)
    summary: str = Field(default="")
    required_skills: list[RequiredSkill] = Field(default_factory=list)
    experience_requirements: list[str] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    jd_hash: str = Field(description="sha256 of the original JD text used for caching.")
    extracted_at: str = Field(description="ISO 8601 UTC timestamp.")


class RubricCriterion(BaseModel):
    """A single assessable criterion in the rubric."""

    name: str = Field(min_length=1)
    description: str = Field(default="")
    weight: float = Field(ge=0.0, description="Relative weight within weighted_score.")
    category: str = Field(
        default="required_skills",
        description="Maps to rubric.default_weights category.",
    )
    examples_of_sufficient_evidence: list[str] = Field(default_factory=list)


class Rubric(BaseModel):
    """Structured evaluation rubric generated from a RoleProfile (§4.1).

    Cached per (jd_hash, rubric_config_version) (§9).
    """

    criteria: list[RubricCriterion] = Field(min_length=1)
    jd_hash: str
    rubric_config_version: str
    generated_at: str = Field(description="ISO 8601 UTC timestamp.")

    @field_validator("criteria")
    @classmethod
    def criteria_names_unique(cls, v: list[RubricCriterion]) -> list[RubricCriterion]:
        names = [c.name for c in v]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Rubric criteria names must be unique. Duplicates: {duplicates}")
        return v


# ── Evidence buckets (agent tool output) ──────────────────────────────────────


class EvidenceBuckets(BaseModel):
    """Output of extract_evidence tool — JD-aligned, source-tagged evidence.

    Each item must have a source tag so evidence is traceable to the résumé section.
    """

    items: list[EvidenceItem] = Field(default_factory=list)


# ── Capability hypotheses (agent tool output) ─────────────────────────────────


class CapabilityHypotheses(BaseModel):
    """Output of assess_capabilities tool.

    Maps capability dimension names to their assessments.
    confidence == 'unknown' must not be used when evidence exists — it means
    the capability has not been assessed at all, not that evidence is weak (ADR-1).
    """

    assessments: dict[str, CapabilityAssessment] = Field(default_factory=dict)


# ── Verification verdict (agent tool output) ──────────────────────────────────


class VerificationResult(BaseModel):
    """Output of verify_evidence tool.

    Per-gap reasoning is required so the human reviewer can understand WHY
    a candidate was flagged as insufficient rather than just seeing the verdict.
    """

    verdict: VerificationVerdict
    per_gap_reasoning: dict[str, str] = Field(
        default_factory=dict,
        description="Maps criterion name → reason why evidence is sufficient or insufficient.",
    )


# ── Interview question set (agent tool output) ────────────────────────────────


class InterviewQuestion(BaseModel):
    """A single interview question."""

    question: str = Field(min_length=1)
    target_criterion: str = Field(description="Rubric criterion this question addresses.")
    rationale: str = Field(description="Why this gap warrants this question.")
    priority_score: float = Field(
        ge=0.0,
        description="weight × (1 − confidence_weight), higher = more important to ask.",
    )


class QuestionSet(BaseModel):
    """Output of generate_interview_questions tool.

    Ranked by priority_score; only top 1-2 questions should be surfaced (§4.2).
    """

    questions: list[InterviewQuestion] = Field(default_factory=list)

    @property
    def top_questions(self) -> list[InterviewQuestion]:
        """Return the top 2 questions ranked by priority_score."""
        return sorted(self.questions, key=lambda q: q.priority_score, reverse=True)[:2]


# ── Updated scorecard after reevaluation (agent tool output) ─────────────────


class UpdatedScoreCard(BaseModel):
    """Output of reevaluate_candidate tool after receiving an interview answer."""

    scorecard: dict[str, ScoreEntry]
    capabilities: dict[str, CapabilityAssessment]
    changes_summary: str = Field(
        description="Short summary of what changed after incorporating the new answer."
    )


# ── Bias report (§9, ADR-7) ───────────────────────────────────────────────────


class BiasIndicator(BaseModel):
    """A single potential bias signal detected by run_fairness_probe."""

    criterion: str
    description: str
    severity: Literal["low", "medium", "high"]
    counterfactual_delta: Optional[float] = Field(
        default=None,
        description="Score difference between identity-swapped variants (0-100 scale).",
    )


class BiasReport(BaseModel):
    """Output of run_fairness_probe (§4.1, ADR-7).

    run_fairness_probe runs on shortlisted candidates only (ADR-7 — expensive check
    reserved for where it's needed).
    """

    candidate_a: str
    candidate_b: str
    indicators: list[BiasIndicator] = Field(default_factory=list)
    overall_risk: Literal["low", "medium", "high"]
    summary: str
    probe_timestamp: str = Field(description="ISO 8601 UTC timestamp.")


# ── Decision summary (§9) ─────────────────────────────────────────────────────


class RecruiterSummary(BaseModel):
    """Output of compose_decision_summary (§4.1).

    Every claim must cite evidence_refs so the summary is verifiable, not just
    a generated narrative. This is a hard requirement (§4.1, §11 hallucination metric).
    """

    shortlist: list[ShortlistEntry]
    bias_reports: list[BiasReport] = Field(default_factory=list)
    overall_recommendation: str
    evidence_refs: list[str] = Field(
        description="All evidence references cited in this summary — must be non-empty.",
    )
    generated_at: str
    run_id: str

    @field_validator("evidence_refs")
    @classmethod
    def evidence_refs_must_exist(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError(
                "RecruiterSummary.evidence_refs must not be empty. "
                "Every claim must cite at least one evidence source (§4.1)."
            )
        return v


# ── Scheduling types (§10, S10) ───────────────────────────────────────────────


class AvailabilitySlot(BaseModel):
    """A single available interview time slot."""

    start_iso: str = Field(description="ISO 8601 UTC start datetime.")
    end_iso: str = Field(description="ISO 8601 UTC end datetime.")
    timezone: str = Field(default="UTC")


class PendingConfirmation(BaseModel):
    """Output of propose_interview — awaiting candidate/interviewer confirmation."""

    candidate: str
    slot: AvailabilitySlot
    confirmation_id: str
    proposed_at: str = Field(description="ISO 8601 UTC timestamp.")
    status: Literal["pending", "confirmed", "declined"] = Field(default="pending")


# ── Graph state (§5) ──────────────────────────────────────────────────────────


class GraphState(BaseModel):
    """Root state for the LangGraph outer graph.

    Every field is typed and validated. State transitions write to this object
    via LangGraph's state update mechanism — nodes never receive raw dicts.

    trajectory is auto-appended by the graph runtime (ADR-8) — never written
    by a node or the agent.
    """

    jd: str = Field(default="", description="Original job description text.")
    role_profile: Optional[RoleProfile] = Field(default=None)
    rubric: Optional[Rubric] = Field(default=None)
    candidates: list[CandidateState] = Field(default_factory=list)
    current_idx: int = Field(default=0, ge=0)
    shortlist: list[ShortlistEntry] = Field(default_factory=list)
    trajectory: list[TrajectoryEntry] = Field(
        default_factory=list,
        description="Auto-appended by graph runtime. NEVER written by nodes or the agent (ADR-8).",
    )
    step_count: int = Field(default=0, ge=0)
    run_id: str = Field(default="", description="Unique identifier for this pipeline run.")

    @model_validator(mode="after")
    def current_idx_within_bounds(self) -> "GraphState":
        """current_idx must be a valid index into candidates, or 0 when list is empty."""
        if self.candidates and self.current_idx >= len(self.candidates):
            raise ValueError(
                f"current_idx={self.current_idx} is out of bounds for "
                f"candidates list of length {len(self.candidates)}."
            )
        return self
