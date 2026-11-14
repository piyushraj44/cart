"""Pydantic response/request schemas - the contract between backend and frontend."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.intelligence.schemas import (CDIResult, CounterfactualResult, FSPMResult, FusionGroup, IntelligenceReport,
                                          MCAIResult, ReplayRecord, TemporalResult)

SourceType = Literal["text", "html", "screenshot_ocr"]
RiskLevel = Literal["high", "medium", "low", "none"]


class TextAuditRequest(BaseModel):
    text: str = Field(..., description="Pasted webpage text")


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class SourceLocation(BaseModel):
    source_type: SourceType
    # text: character offsets into the submitted text; html: CSS selector + tag; screenshot: bbox in image pixels
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    line: Optional[int] = None
    css_selector: Optional[str] = None
    html_tag: Optional[str] = None
    html_snippet: Optional[str] = None
    bbox: Optional[BBox] = None
    ocr_confidence: Optional[float] = None


class SimilarExample(BaseModel):
    text: str
    label: str
    similarity: float


class ClaimCheck(BaseModel):
    """Rule-based claim specificity (backend/claim_check.py). Never changes a model score."""
    verdict: Literal["checkable", "vague", "mixed", "none"]
    checkable: list[str] = []        # "label: matched text" - deadline, reference price, offer terms...
    vague: list[str] = []            # "label: matched text" - hurry, undated countdown, stock count...
    note: str = ""


EvidenceTier = Literal["observed", "pressure_claim"]


class Finding(BaseModel):
    id: str
    category: str                    # machine key: urgency | scarcity | social_proof | misdirection | preselected_option
    category_label: str              # human readable
    evidence: str                    # exact text from the user's input
    explanation: str
    detection_method: Literal["ml_classifier", "dom_observation"]
    model_score: Optional[float]     # calibrated P(dark pattern) in [0,1]; null for DOM observations
    confidence_pct: Optional[int]    # round(model_score*100)
    confidence_statement: str        # e.g. "ClearCart is 87% sure this is a fraud/scam (dark pattern) tactic: Misleading urgency."
    category_probabilities: Optional[dict[str, float]] = None
    risk_level: RiskLevel
    key_terms: list[str] = []        # words in the evidence that pushed the model towards the category
    similar_examples: list[SimilarExample] = []  # nearest labelled TRAINING examples
    source_location: SourceLocation
    ethical_recommendation: str
    suggested_rewrite: Optional[str] = None
    short_reason: str = ""           # one-line reason shown in the annotated preview callout
    # observed = a fact read directly from the input (e.g. a checked box in the HTML);
    # pressure_claim = the text model recognises pressure language, which ClearCart cannot verify as true or false.
    # (Withheld elements form the third tier, `uncertain_elements`.)
    evidence_tier: EvidenceTier = "pressure_claim"
    claim_check: Optional[ClaimCheck] = None


class CategoryBreakdown(BaseModel):
    category: str
    category_label: str
    count: int
    max_score: Optional[float]


class AuditSummary(BaseModel):
    elements_analysed: int
    suspicious_findings: int
    categories_detected: list[CategoryBreakdown]
    overall_risk: RiskLevel
    overall_score: Optional[float]   # max calibrated P(dark) over all analysed elements
    overall_confidence_statement: str
    no_concern: bool
    summary_text: str
    # 0-99 gauge value = floor(100 * highest calibrated P(dark) among analysed elements), capped at 99
    risk_score: Optional[int] = None
    threshold: Optional[float] = None    # flagging threshold used for this input type
    # counts per evidence tier: observed, pressure_claim (findings) and review (uncertain_elements)
    tier_counts: Optional[dict[str, int]] = None


class AnalysedElement(BaseModel):
    text: str
    p_dark: float
    flagged: bool
    source_location: SourceLocation
    # Set only on `uncertain_elements`: scores above threshold withheld for review
    # by an evidence/uncertainty gate (see backend/auditor.py).
    gate: Optional[Literal["quoted_mention", "out_of_distribution", "structured_promotion", "neutral_choice",
                           "checkable_claim"]] = None
    gate_note: Optional[str] = None
    masked_text: Optional[str] = None            # quoted_mention: the text with quoted spans masked
    masked_p_dark: Optional[float] = None        # quoted_mention: calibrated P(dark) of masked_text
    max_train_similarity: Optional[float] = None  # max cosine similarity to any training text (MiniLM space)
    ood_threshold: Optional[float] = None         # out_of_distribution: the validation-chosen cut-off
    claim_check: Optional[ClaimCheck] = None      # checkable_claim (and other gated urgency/scarcity elements)


class VisualSignal(BaseModel):
    """EXPERIMENTAL visual pattern-type signal (opt-in, CLEARCART_VISUAL=1). Indicates a pattern type, not deception;
    never part of findings, the risk summary or the intelligence layer."""
    label: str
    probability: float                      # temperature-calibrated, 0..1
    threshold: float                        # per-label VAL-selected threshold the probability met
    bbox: Optional[BBox] = None             # Grad-CAM evidence box in original image pixels
    provenance: str = "model (visual, experimental)"


class AuditResponse(BaseModel):
    audit_id: str
    input_type: SourceType
    created_at: str
    processing_ms: int
    summary: AuditSummary
    findings: list[Finding]
    benign_elements: list[AnalysedElement]   # up to 50, lowest risk first
    disclaimer: str
    # Elements whose score crossed the threshold but were withheld for review by
    # a false-positive gate. Highest score first.
    uncertain_elements: list[AnalysedElement] = []
    image_size: Optional[dict[str, int]] = None   # screenshot only: {"width":..,"height":..}
    warnings: list[str] = []
    # NEW (intel-v1.0, optional, backward compatible): CDI, MCAI, counterfactual; FSPM/temporal
    # are marked unavailable for single-page audits.
    intelligence: Optional[IntelligenceReport] = None
    # model / calibration / scoring versions used for this audit (reproducibility)
    versions: Optional[dict[str, str]] = None
    # EXPERIMENTAL (screenshot only, opt-in): None = feature off; [] = on but nothing above threshold.
    visual_signals: Optional[list[VisualSignal]] = None


class ModelInfo(BaseModel):
    name: str
    version: str
    architecture: str
    embedder: str
    categories: list[dict[str, str]]
    threshold: float
    temperature: float
    training_data: str
    test_metrics: dict
    limitations: list[str]
    ocr_available: bool


class SampleInput(BaseModel):
    id: str
    title: str
    input_type: Literal["text", "html"]
    content: str
    provenance: str
    expected: str


class JourneyStageResult(BaseModel):
    index: int
    name: str
    stage_type: Optional[str] = None
    position: int
    input_type: Literal["text", "html", "screenshot"]
    captured_at: Optional[str] = None
    synthetic: bool = False
    content: Optional[str] = None            # echoed text/html (needed for previews); None for screenshots
    audit: AuditResponse
    pressure: float                          # p_i used by FSPM
    fused_into: Optional[int] = None         # set when this stage was merged into an earlier same-page stage


class JourneyResponse(BaseModel):
    journey_id: str
    created_at: str
    processing_ms: int
    stages: list[JourneyStageResult]
    fspm: FSPMResult
    temporal: TemporalResult
    cdi: CDIResult
    mcai: MCAIResult                         # choice sets across all stages (ids prefixed s<k>-)
    counterfactual: CounterfactualResult
    config_version: str
    disclaimer: str
    warnings: list[str] = []
    versions: Optional[dict[str, str]] = None
    fusion: list[FusionGroup] = []
    replay: Optional[ReplayRecord] = None


class ReplayResult(BaseModel):
    identical: bool
    differences: list[str]
    journey: JourneyResponse
