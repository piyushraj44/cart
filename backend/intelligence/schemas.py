"""Pydantic schemas for the intelligence layer (all optional additions to existing responses)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field

Provenance = Literal["model", "measured", "default", "declared", "observed", "inferred", "by_construction", "unavailable"]


class EvidenceRef(BaseModel):
    stage_index: Optional[int] = None     # journey stage (0-based); None for single audits
    finding_id: Optional[str] = None      # existing Finding.id (provenance preserved)
    text: str

    @computed_field
    @property
    def evidence_id(self) -> Optional[str]:
        """Globally unique evidence id: 's<stage>:<finding>' in journeys (1-based stage), '<finding>' in single audits."""
        if self.finding_id is None:
            return None
        return self.finding_id if self.stage_index is None else f"s{self.stage_index + 1}:{self.finding_id}"


# ------------------------------------------------------------------ FSPM
class FSPMStage(BaseModel):
    stage_index: int
    name: str
    position: int                          # 1-based position used for weighting
    pressure: float                        # p_i in [0,1]
    pressure_sources: list[EvidenceRef]
    weights: dict[str, float]              # scheme -> normalised weight
    contributions: dict[str, float]        # scheme -> 100 * w_i * p_i


class FSPMResult(BaseModel):
    available: bool
    reason: Optional[str] = None
    formula: str = "FSPM = 100 * sum(F_pos(i) * p_i) / sum(F_pos(i))"
    stages: list[FSPMStage] = []
    scores: dict[str, float] = {}          # scheme -> 0..100
    top_stage_index: Optional[int] = None  # by fibonacci contribution
    fibonacci_numbers: list[int] = []
    positions_mode: Literal["canonical", "submission_order"] = "submission_order"


# ------------------------------------------------------------------ MCAI
class FeatureComparison(BaseModel):
    feature: str
    accept_value: Optional[float]
    decline_value: Optional[float]
    accept_provenance: Provenance
    decline_provenance: Provenance
    asymmetry: Optional[float]             # in [0,1]; None if unavailable
    note: Optional[str] = None


class DimensionScore(BaseModel):
    name: Literal["visual", "interaction", "semantic"]
    available: bool
    score: Optional[float]                 # in [0,1]
    provenance: Provenance
    reason: Optional[str] = None
    features: list[FeatureComparison] = []


class ChoiceOption(BaseModel):
    role: Literal["accept", "decline"]
    text: str
    css_selector: Optional[str] = None
    bbox: Optional[dict] = None
    finding_id: Optional[str] = None


class WeightSensitivity(BaseModel):
    one_at_a_time: list[dict]              # [{component, low, high}] each weight x(1 +/- 0.5), renormalised
    dirichlet_p05: Optional[float]
    dirichlet_p95: Optional[float]
    samples: int


class ChoiceSetResult(BaseModel):
    id: str
    accept: Optional[ChoiceOption]
    decline: ChoiceOption
    dimensions: list[DimensionScore]
    mcai: Optional[float]                  # 0..100 over available dimensions
    coverage: float                        # share of MCAI weight that was available
    role_detection: str = "rule-based lexicon (candidate extraction only)"
    dimension_contributions: dict[str, float] = {}   # points of the 0..100 MCAI per available dimension
    top_dimension: Optional[str] = None               # dimension contributing most
    alternatives: dict[str, float] = {}               # same inputs under alternative aggregators (study only)
    sensitivity: Optional[WeightSensitivity] = None


class MCAIResult(BaseModel):
    available: bool
    reason: Optional[str] = None
    formula: str = "MCAI = 100 * sum_{d in available} w_d * D_d / sum_{d in available} w_d"
    weights: dict[str, float]
    choice_sets: list[ChoiceSetResult] = []
    mcai: Optional[float] = None           # max over choice sets (worst case)


# ------------------------------------------------------------------ CDI
class CDIComponent(BaseModel):
    key: str
    label: str
    available: bool
    value: Optional[float]                 # in [0,1]
    weight: float                          # configured weight
    effective_weight: Optional[float]      # renormalised over available components
    contribution: Optional[float]          # points of the 0..100 CDI
    provenance: str
    explanation: str
    evidence: list[EvidenceRef] = []


class CDISensitivity(BaseModel):
    one_at_a_time: list[dict]              # [{component, low, high}]
    dirichlet_p05: Optional[float]
    dirichlet_p95: Optional[float]
    samples: int


class CDIConfidence(BaseModel):
    """Confidence is reported SEPARATELY from severity (cdi) and coverage."""
    evidence_by_provenance: dict[str, int]          # how many evidence items are model / observed / measured / declared ...
    model_evidence: int
    mean_model_confidence: Optional[float]          # mean calibrated P(dark) of model evidence used by the CDI
    min_model_confidence: Optional[float]
    level: Literal["high", "medium", "low", "none"]  # rule: see engine._confidence
    note: str = "Confidence describes how sure the evidence is, not how severe the deception is."


class CDIResult(BaseModel):
    cdi: Optional[float]                   # 0..100, None when no component is available
    band: Optional[str]
    weights_version: str
    components: list[CDIComponent]
    coverage: float                        # sum of configured weights that were available
    evidence_items: int
    sensitivity: Optional[CDISensitivity] = None
    confidence: Optional[CDIConfidence] = None
    alternatives: dict[str, float] = {}    # same components under alternative aggregators (transparency; not used)
    formula: str = "CDI = 100 * sum_{j in available} w_j * x_j / sum_{j in available} w_j"
    note: str = "Interpretable screening index. Not a probability that a merchant is fraudulent."


# ------------------------------------------------------------------ counterfactual
class DiffSegment(BaseModel):
    op: Literal["equal", "delete", "insert"]
    text: str


class RewriteCandidate(BaseModel):
    variant: str                           # e.g. minimal, neutral_statement, drop_pressure_sentence, original
    text: Optional[str]
    score: Optional[float]                 # calibrated P(dark) (model-estimated)
    feasible: bool
    violated: list[str] = []               # constraints violated (facts_added, facts_removed, offer_words, empty, unchanged)
    edit_distance: Optional[int] = None    # word-level edits from the original


class CounterfactualEdit(BaseModel):
    target: str                            # finding id or choice-set id
    stage_index: Optional[int] = None
    category: str
    action: Literal["rewrite", "uncheck_default", "equalise_choice", "no_safe_rewrite", "remove_unverifiable_claim"]
    original: str
    redesigned: Optional[str]
    diff: list[DiffSegment] = []
    original_score: Optional[float] = None     # calibrated P(dark)
    redesigned_score: Optional[float] = None   # calibrated P(dark) of the rewrite (model-estimated)
    facts_preserved: bool = True
    explanation: str
    candidates: list[RewriteCandidate] = []   # every candidate considered by the optimiser (transparency)


class CounterfactualResult(BaseModel):
    edits: list[CounterfactualEdit]
    original_cdi: Optional[float]
    redesigned_cdi: Optional[float]
    risk_reduction_pct: Optional[float]
    original_components: list[CDIComponent] = []
    redesigned_components: list[CDIComponent] = []
    original_semantic_pressure: Optional[float] = None
    redesigned_semantic_pressure: Optional[float] = None
    original_mcai: Optional[float] = None
    redesigned_mcai: Optional[float] = None
    original_fspm: Optional[float] = None
    redesigned_fspm: Optional[float] = None
    optimisation: Optional[dict] = None    # method, objective, constraints, optimality argument
    label: str = ("Model-estimated reduction: the redesigned text is re-scored by the same classifier. "
                  "This is not a measured change in consumer behaviour.")


# ------------------------------------------------------------------ temporal
class TimelineEvent(BaseModel):
    stage_index: int
    stage_name: str
    timestamp: Optional[str]
    kind: Literal["observed_fact", "observation", "model_inference"]
    category: str
    description: str
    evidence: list[EvidenceRef] = []
    pressure: Optional[float] = None       # stage pressure p_i at this point
    deceptive_signal: bool = False         # True only for facts that enter FSPM/CDI
    previous_state: Optional[str] = None   # observed/declared state at the earlier stage (change events)
    current_state: Optional[str] = None    # state at this stage
    pressure_delta: Optional[float] = None # p_i - p_(i-1) in timeline order (filled by the engine)


class TemporalResult(BaseModel):
    available: bool
    reason: Optional[str] = None
    ordering: Literal["timestamp", "submission_order"] = "submission_order"
    events: list[TimelineEvent] = []
    pressure_progression: list[float] = []
    progression_stage_indices: list[int] = []   # stage index for each pressure_progression point (0-based)
    metrics: Optional[dict] = None              # persistence / repetition / pressure-change metrics (engine)


# ------------------------------------------------------------------ containers
class IntelligenceReport(BaseModel):
    config_version: str
    cdi: CDIResult
    mcai: MCAIResult
    counterfactual: CounterfactualResult
    fspm: FSPMResult
    temporal: TemporalResult
    status_note: str = ("New in intel-v1.0: implemented and unit-tested; evaluated only on synthetic, explicitly "
                        "labelled benchmarks (no real-world ground truth exists for these indices).")


class ChoicePathDeclaration(BaseModel):
    label: str = Field(..., max_length=200)
    accept_steps: int = Field(..., ge=1, le=50)
    decline_steps: int = Field(..., ge=1, le=50)


class JourneyStageInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    stage_type: Optional[str] = None        # one of config.CANONICAL_STAGES
    input_type: Literal["text", "html", "screenshot"]
    content: Optional[str] = None           # text / html
    file_field: Optional[str] = None        # multipart field holding the screenshot
    captured_at: Optional[str] = None       # ISO-8601 timestamp
    choice_path: Optional[ChoicePathDeclaration] = None
    synthetic: bool = False                 # authored demo content flag (shown in UI)
    fuse_with: Optional[int] = Field(None, ge=0)  # 0-based index of an EARLIER stage showing the same page in another modality


class FusionGroup(BaseModel):
    """Stages that show the SAME page in different modalities (declared via fuse_with)."""
    stage_indices: list[int]
    representative: int                     # stage whose position is used for FSPM
    duplicate_pairs: list[dict] = []        # [{a: EvidenceRef, b: EvidenceRef, similarity}]
    fused_pressure: float
    note: str = ("Findings were merged across modalities: each category counts once (strongest calibrated score), "
                 "so the same manipulation seen as text, HTML and screenshot is not counted three times.")


class ReplayRecord(BaseModel):
    """Everything needed to reconstruct and re-verify a journey audit."""
    inputs: list[JourneyStageInput]         # text/html stored verbatim; screenshot stages store only their hash
    input_sha256: list[str]
    versions: dict[str, str]
    expected: dict                          # key deterministic outputs to compare on replay
