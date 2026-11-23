"""Pure, auditable display helpers for learner-facing reasons and research strategies.

No model fitting, no user profiling beyond saved historical model explanations.
Strategies are descriptive rules/hypotheses; they are not causal estimates.
"""
from __future__ import annotations
from collections import defaultdict
import math

import pandas as pd

AGGREGATE_FEATURE = "Other features (aggregated)"

FEATURE_GROUPS = {
    "Selection History": (
        "history_count", "history_span_days", "provider_history_count",
        "course_code_history_count", "provider_affinity", "course_code_affinity",
    ),
    "Selection Timing": ("days_since_last_start", "user_starts_30d"),
    "Selection Diversity": ("provider_diversity", "course_code_diversity"),
}

STRATEGY_PLANS = {
    "Selection History": {
        "title": "History-Based Personalized Course Pathway",
        "summary": "Highlight relevant follow-on courses connected with previously selected providers or course families, and offer an alternative path for discovery.",
        "dimension": "Course Participation (historical course selection)",
        "action": "Present a contextual learning-path message alongside the recommended courses; evaluate uptake in a future controlled study.",
    },
    "Selection Timing": {
        "title": "Timing-Aware Course Discovery",
        "summary": "Consider a well-timed, non-intrusive course-discovery prompt that reflects previous selection recency, with opt-out options.",
        "dimension": "Course Participation (historical start timing)",
        "action": "Test an optional reminder or course-preview placement; do not infer the learner's present availability from 2013 data.",
    },
    "Selection Diversity": {
        "title": "Diversified Course Discovery",
        "summary": "Present a balanced selection of familiar and less familiar course families so the learner can explore alternatives.",
        "dimension": "Course Participation (historical choice diversity)",
        "action": "Evaluate whether a more diversified presentation improves discovery, rather than assuming diversity causes engagement.",
    },
}


def _validated_shap(pair, expected_score=None) -> pd.DataFrame | None:
    """Return the saved *individual* features if the SHAP reconstruction matches."""
    if pair is None:
        return None
    try:
        score = float(pair["score"])
        baseline = float(pair["baseline"])
        frame = pair["shap"]
        if expected_score is not None and not math.isclose(score, float(expected_score), abs_tol=1e-5, rel_tol=0):
            return None
        if not math.isclose(baseline + float(frame["SHAP_Value"].sum()), score, abs_tol=1e-4, rel_tol=0):
            return None
        individual = frame.loc[frame["Feature"] != AGGREGATE_FEATURE].copy()
        individual["SHAP_Value"] = pd.to_numeric(individual["SHAP_Value"], errors="coerce")
        individual = individual.dropna(subset=["SHAP_Value"])
        individual["Abs_SHAP"] = individual["SHAP_Value"].abs()
        return individual.sort_values("Abs_SHAP", ascending=False)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def student_reason(pair, expected_score, friendly_feature, max_positive=2) -> dict:
    """Plain-language display reason, derived only from verified saved SHAP values.

    A negative contribution is NOT a claim that the learner dislikes a course.
    """
    values = _validated_shap(pair, expected_score)
    if values is None or values.empty:
        return {"available": False,
                "main": "A validated explanation is not available for this recommendation.",
                "secondary": "The course ranking is based on a saved historical model score."}
    positive = values.loc[values["SHAP_Value"] > 0].head(max_positive)
    negative = values.loc[values["SHAP_Value"] < 0].head(1)
    if positive.empty:
        main = ("No individual positive factor appeared among the saved leading features. "
                "The course was nevertheless in the model's top recommendations.")
    else:
        factors = ", ".join(friendly_feature(f).lower() for f in positive["Feature"])
        main = f"The model gave this course a higher ranking score partly because of {factors}."
    # Keep technical/negative attributions for the Platform Intelligence view.
    # The student asked for *only why* a course was suggested.
    secondary = "This explains the historical suggestion; it does not guarantee a match to your interests."
    return {"available": True, "main": main, "secondary": secondary}


def recommend_strategy(pairs, friendly_feature) -> dict:
    """Propose ONE strategy for selected learner from SHAP history-feature salience.

    pairs: iterable of (saved_explanation_or_None, expected_model_score).
    Uses sum of absolute SHAP contributions across up to 3 top-ranked courses.
    This is a TRANSPARENT HEURISTIC, not a learned marketing-impact model.
    """
    totals = defaultdict(float)
    evidence = defaultdict(lambda: defaultdict(float))
    valid = 0
    for pair, expected_score in pairs:
        features = _validated_shap(pair, expected_score)
        if features is None:
            continue
        valid += 1
        for feat, weight in features[["Feature", "SHAP_Value"]].itertuples(index=False, name=None):
            for group, names in FEATURE_GROUPS.items():
                if feat in names:
                    totals[group] += abs(float(weight))
                    evidence[group][feat] += abs(float(weight))
    if not valid or not totals or max(totals.values()) <= 0:
        return {"available": False, "title": "Not enough explanation evidence",
                "summary": "A personalized strategy cannot be justified from the saved explanations for these courses.",
                "dimension": "Not determined", "action": "Review the available historical data.",
                "valid_recommendations": valid, "group_scores": {}, "top_factors": []}
    winning = max(FEATURE_GROUPS, key=lambda name: totals[name])
    plan = STRATEGY_PLANS[winning]
    top = sorted(evidence[winning].items(), key=lambda item: -item[1])[:3]
    return {"available": True, "title": plan["title"], "summary": plan["summary"],
            "dimension": plan["dimension"], "action": plan["action"],
            "group": winning, "valid_recommendations": valid,
            "group_scores": {key: round(float(totals[key]), 6) for key in FEATURE_GROUPS},
            "top_factors": [friendly_feature(name) for name, _ in top]}
