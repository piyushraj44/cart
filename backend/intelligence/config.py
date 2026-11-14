"""Versioned constants for the ClearCart intelligence layer.

Every number the new indices depend on is defined here, with its rationale, so results are
reproducible and reviewable. These are DESIGN CHOICES, not values learned from data: no labelled
dataset of journeys, choice sets or composite deception levels exists for this project. They are
reported as such in the API response (`config_version`) and in the technical addendum.
"""
from __future__ import annotations

CONFIG_VERSION = "intel-v1.1"   # v1.1: inferred interaction, constrained optimiser, fusion, confidence

# ---------------------------------------------------------------- FSPM
FSPM_SCHEMES = ("fibonacci", "linear", "equal")
# Canonical order used when a stage declares `stage_type`. A journey that omits some stages keeps
# the canonical positions of the stages it has (missing stages are NOT invented or zero-filled).
CANONICAL_STAGES = ("landing", "product", "cart", "checkout", "payment", "confirmation", "exit")

# ---------------------------------------------------------------- stage / content pressure
# Intensity-breadth blend: pressure = lam * max_k c_k + (1 - lam) * mean_k c_k over the four
# classifier categories. A pure noisy-OR/max saturates at ~1.0 for any page with one flagged
# element (the calibrated classifier is near-binary on in-distribution strings), which made
# journey stages indistinguishable; the blend keeps a single certain tactic high (~0.62) while
# rewarding breadth (all four tactics ~1.0).
PRESSURE_INTENSITY_WEIGHT = 0.5
# Pressure value assigned to directly OBSERVED (non-model) facts when they enter a pressure
# aggregate. A pre-selected option that mentions cost/consent, a countdown observed to increase,
# or a newly introduced pre-selected add-on are facts, so they enter at full strength; a
# pre-selected option without a cost/consent cue enters at half strength.
OBSERVED_FACT_STRENGTH = 1.0
OBSERVED_WEAK_FACT_STRENGTH = 0.5

# ---------------------------------------------------------------- MCAI
# Dimension weights (sum to 1). Unavailable dimensions are dropped and the remaining weights
# renormalised; the dropped share is reported as `coverage`.
MCAI_WEIGHTS = {"visual": 0.4, "interaction": 0.3, "semantic": 0.3}
# Semantic sub-score weights (sum to 1).
SEMANTIC_WEIGHTS = {"guilt": 0.5, "accept_pressure": 0.25, "wording_imbalance": 0.25}
# Interaction asymmetry is only computed from declared/observed step counts.
# Visual feature list (each compared as a relative difference accept-vs-decline).
VISUAL_FEATURES_HTML = ("control_kind", "font_size_px", "font_weight", "background_fill", "contrast_ratio", "opacity")
VISUAL_FEATURES_SCREENSHOT = ("text_height_px", "text_box_area_px", "contrast_ratio", "background_fill")

# ---------------------------------------------------------------- CDI
# Component weights (sum to 1), version-tagged. Content pressure dominates because it is the only
# component backed by a validated classifier; asymmetry and obstruction are observational.
CDI_WEIGHTS_VERSION = "cdi-w1"
CDI_WEIGHTS = {"content_pressure": 0.55, "choice_asymmetry": 0.25, "observed_obstruction": 0.20}
# Screening bands for display only (NOT calibrated probabilities).
CDI_BANDS = ((75, "high"), (50, "elevated"), (25, "moderate"), (0, "low"))
# Sensitivity analysis (deterministic).
SENSITIVITY_ONE_AT_A_TIME = 0.5          # each weight scaled by (1 +/- 0.5), then renormalised
SENSITIVITY_DIRICHLET_SAMPLES = 1000
SENSITIVITY_DIRICHLET_CONCENTRATION = 20.0
SENSITIVITY_SEED = 7

# ---------------------------------------------------------------- temporal
COUNTDOWN_TOLERANCE_SECONDS = 60          # slack when comparing timestamped countdown readings
