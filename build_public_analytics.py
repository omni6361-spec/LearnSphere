"""Validate full saved SHAP/LIME export and generate compact public dashboard statistics.

This script reads existing results; NEVER runs a model, SHAP, or LIME.
It does not publish data or change the original SQLite databases.

Usage (from the deployment directory):
    python build_public_analytics.py --expected 177247
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3

import pandas as pd

AGGREGATE_LABEL = "Other features (aggregated)"


def unpack(raw: str, exclude_aggregate: bool = True) -> list[tuple[str, float]]:
    items = json.loads(raw)
    if not isinstance(items, list) or not items:
        raise ValueError("An explanation is missing or not a list")
    values = []
    seen = set()
    for pair in items:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("An explanation has a malformed feature pair")
        feature, magnitude = str(pair[0]), float(pair[1])
        if not math.isfinite(magnitude) or feature in seen:
            raise ValueError("An explanation has invalid or repeated feature values")
        seen.add(feature)
        if exclude_aggregate and feature == AGGREGATE_LABEL:
            continue
        values.append((feature, magnitude))
    values.sort(key=lambda item: abs(item[1]), reverse=True)
    if not values:
        raise ValueError("An explanation has no individual feature values")
    return values


def group_name(r2: float) -> str:
    if r2 < 0.5:
        return "Low (<0.50)"
    if r2 < 0.75:
        return "Moderate (0.50–0.75)"
    return "Higher (≥0.75)"


def percent(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 4) if denominator else 0.0


def aggregate(data_dir: Path, expected_total: int | None = None) -> dict:
    rec_path = data_dir / "recommendations.csv"
    shards = sorted((data_dir / "local_xai").glob("xai_batch_*.sqlite"))
    if not rec_path.is_file() or not shards:
        raise FileNotFoundError("Required data/recommendations.csv or data/local_xai/xai_batch_*.sqlite is missing")
    rec = pd.read_csv(rec_path, dtype={"userid_DI": str, "course_id": str})
    required = {"userid_DI", "course_id", "rank", "recommendation_score"}
    if not required.issubset(rec.columns):
        raise ValueError(f"Recommendations missing columns: {sorted(required-set(rec.columns))}")
    if rec[["userid_DI", "course_id", "rank", "recommendation_score"]].isna().any().any():
        raise ValueError("Missing recommendation identifiers, ranks or scores")
    expected_keys = set(zip(rec.userid_DI, rec.course_id))
    if len(expected_keys) != len(rec):
        raise ValueError("Duplicate learner-course recommendation pairs")
    if expected_total is not None and len(rec) != expected_total:
        raise ValueError(f"Expected {expected_total:,} recommendations, found {len(rec):,}")
    for account in rec.userid_DI:
        if not str(account).startswith("Learner "):
            raise ValueError("This export contains a non-pseudonymous learner ID. Do not publish it.")

    score_lookup = dict(zip(zip(rec.userid_DI, rec.course_id), pd.to_numeric(rec.recommendation_score, errors="raise")))
    all_keys = set()
    courses = defaultdict(lambda: {"count": 0, "score_sum": 0.0, "fidelity_sum": 0.0,
                                  "overlap_sum": 0.0, "jaccard_sum": 0.0, "zero": 0,
                                  "low": 0, "sign_matches": 0, "shared_signs": 0})
    groups = defaultdict(lambda: {"count": 0, "fidelity_sum": 0.0,
                                 "overlap_sum": 0.0, "jaccard_sum": 0.0})
    overlap_freq, shap_top_freq, lime_top_freq = Counter(), Counter(), Counter()
    fidelity_hist = Counter()
    total_overlap = total_jaccard = total_fidelity = 0.0
    total_shared = total_sign_matches = total_sign_eligible = total_zero = 0
    shard_counts = {}

    for shard in shards:
        count = 0
        with closing(sqlite3.connect(f"file:{shard.as_posix()}?mode=ro", uri=True)) as db:
            cursor = db.execute("SELECT learner,course,score,baseline,shap_json,lime_json,lime_fidelity,samples FROM explanations")
            for learner, course, score, baseline, shap_raw, lime_raw, fidelity, samples in cursor:
                count += 1
                key = (learner, course)
                if key not in expected_keys or key in all_keys:
                    raise ValueError(f"Missing source recommendation or duplicate SQLite key: {key}")
                all_keys.add(key)
                score, baseline, fidelity = float(score), float(baseline), float(fidelity)
                if not all(math.isfinite(v) for v in (score, baseline, fidelity)):
                    raise ValueError(f"Non-finite metric: {key}")
                if abs(score - float(score_lookup[key])) > 1e-5:
                    raise ValueError(f"Model score mismatch: {key}")
                if samples is None or int(samples) <= 0:
                    raise ValueError(f"Missing LIME samples: {key}")
                shap_full = unpack(shap_raw, exclude_aggregate=False)
                if abs(baseline + sum(v for _, v in shap_full) - score) > 1e-4:
                    raise ValueError(f"SHAP attribution reconstruction mismatch: {key}")
                shap_top = unpack(shap_raw, exclude_aggregate=True)[:5]
                lime_top = unpack(lime_raw, exclude_aggregate=True)[:5]
                names_sh, names_li = dict(shap_top), dict(lime_top)
                intersection = set(names_sh) & set(names_li)
                union = set(names_sh) | set(names_li)
                overlap = len(intersection)
                jac = overlap / len(union) if union else 0.0
                signs = sum((names_sh[f] > 0) == (names_li[f] > 0) for f in intersection
                              if names_sh[f] != 0 and names_li[f] != 0)
                eligible_signs = sum(names_sh[f] != 0 and names_li[f] != 0 for f in intersection)
                total_shared += eligible_signs
                total_sign_matches += signs
                if eligible_signs:
                    total_sign_eligible += 1
                overlap_freq[overlap] += 1
                shap_top_freq[shap_top[0][0]] += 1
                lime_top_freq[lime_top[0][0]] += 1
                fidelity_hist[f"{math.floor(fidelity*10)/10:.1f}"] += 1
                total_overlap += overlap
                total_jaccard += jac
                total_fidelity += fidelity
                if not overlap:
                    total_zero += 1
                row = courses[course]
                row["count"] += 1
                row["score_sum"] += score
                row["fidelity_sum"] += fidelity
                row["overlap_sum"] += overlap
                row["jaccard_sum"] += jac
                row["zero"] += overlap == 0
                row["low"] += fidelity < 0.5
                row["sign_matches"] += signs
                row["shared_signs"] += eligible_signs
                bucket = groups[group_name(fidelity)]
                bucket["count"] += 1
                bucket["fidelity_sum"] += fidelity
                bucket["overlap_sum"] += overlap
                bucket["jaccard_sum"] += jac
        shard_counts[shard.name] = count
        print(f"Validated {shard.name}: {count:,} rows", flush=True)

    if all_keys != expected_keys:
        raise ValueError(f"Missing explanation pairs: {len(expected_keys-all_keys):,}")
    n = len(all_keys)
    course_results = []
    for name, row in courses.items():
        count = row["count"]
        course_results.append({
            "Course": name, "Recommendations": count,
            "Mean_Model_Score": round(row["score_sum"] / count, 6),
            "Mean_LIME_Fidelity": round(row["fidelity_sum"] / count, 6),
            "Mean_Top5_Overlap": round(row["overlap_sum"] / count, 6),
            "Mean_Jaccard": round(row["jaccard_sum"] / count, 6),
            "Zero_Overlap_Percent": percent(row["zero"], count),
            "Low_Fidelity_Percent": percent(row["low"], count),
            "Shared_Sign_Agreement": round(row["sign_matches"] / row["shared_signs"], 6) if row["shared_signs"] else None,
        })
    course_results.sort(key=lambda r: -r["Recommendations"])
    group_results = []
    for label in ["Low (<0.50)", "Moderate (0.50–0.75)", "Higher (≥0.75)"]:
        row = groups[label]
        if row["count"]:
            group_results.append({"Group": label, "Recommendations": row["count"],
                                  "Mean_LIME_Fidelity": round(row["fidelity_sum"] / row["count"], 6),
                                  "Mean_Top5_Overlap": round(row["overlap_sum"] / row["count"], 6),
                                  "Mean_Jaccard": round(row["jaccard_sum"] / row["count"], 6)})
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": "SHAP and LIME saved per recommendation; top-five per-method feature rankings exclude the synthetic SHAP aggregation row. LIME R² is in-sample local surrogate fit; agreement is descriptive only.",
        "total_recommendations": n,
        "unique_learners": int(rec.userid_DI.nunique()),
        "unique_courses": int(rec.course_id.nunique()),
        "shard_counts": shard_counts,
        "mean_lime_fidelity": round(total_fidelity/n, 6),
        "mean_top5_overlap": round(total_overlap/n, 6),
        "mean_jaccard": round(total_jaccard/n, 6),
        "zero_overlap_percent": percent(total_zero, n),
        "shared_sign_agreement": round(total_sign_matches/total_shared, 6) if total_shared else None,
        "shared_sign_denominator": total_shared,
        "shared_sign_numerator": total_sign_matches,
        "overlap_distribution": [{"Shared_Features": int(k), "Recommendations": v,
                                  "Percentage": percent(v,n)} for k,v in sorted(overlap_freq.items())],
        "fidelity_groups": group_results,
        "courses": course_results,
        "most_frequent_shap_top1": [{"Feature": k, "Recommendations": v} for k,v in shap_top_freq.most_common(12)],
        "most_frequent_lime_top1": [{"Feature": k, "Recommendations": v} for k,v in lime_top_freq.most_common(12)],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--expected", type=int, default=177247)
    args = parser.parse_args()
    result = aggregate(args.data_dir, expected_total=args.expected)
    output = args.data_dir / "public_analytics.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"PASSED: {result['total_recommendations']:,} unique recommendations, public metrics: {output}")


if __name__ == "__main__":
    main()
