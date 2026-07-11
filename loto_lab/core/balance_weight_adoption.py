from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import difflib
import json
import math

import pandas as pd

from balance_weight_research import (
    BALANCE_WEIGHT_RESEARCH_VERSION,
    CURRENT_BALANCE_WEIGHTS,
    MIN_WEIGHT_RECOMMENDATION_SAMPLES,
    REVIEW_HISTORY_COLUMNS,
    SUBSCORE_LABELS,
    _percent,
    _round,
    _safe_float,
    _safe_int,
    _scenario_weights,
    _stable_json,
    build_per_draw_candidate_ranking,
    build_per_draw_ranking_summary,
    build_ranking_stability_summary,
    build_rolling_weight_evaluation,
    candidate_weights_hash,
    read_review_history,
    simulate_candidate_weights,
)


ADOPTION_DRY_RUN_VERSION = "balance_weight_adoption_dry_run_v1"
APPROVAL_HISTORY_COLUMNS = [
    "approval_id",
    "approved_at",
    "game",
    "review_id",
    "candidate_weights_hash",
    "approval_status",
    "approval_comment",
    "readiness_score",
    "dry_run_summary_json",
    "drift_summary_json",
    "safety_checks_json",
    "approver",
]
APPROVAL_STATUSES = {
    "unapproved",
    "needs_rework",
    "dry_run_approved",
    "approved_for_manual_preparation",
}
REVIEW_DECISION_LABELS = {
    "candidate_for_adoption": "採用候補",
    "hold": "保留",
    "rejected": "却下",
    "unreviewed": "未判定",
}
REQUIRED_WEIGHT_KEYS = tuple(CURRENT_BALANCE_WEIGHTS.keys())


def now_text():
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def parse_datetime(value):
    text = str(value or "").strip()
    if not text:
        return pd.NaT
    return pd.to_datetime(text, errors="coerce")


def parse_candidate_weights_json(value):
    if value is None:
        return None, "empty"
    try:
        if pd.isna(value):
            return None, "empty"
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return None, "empty"
    try:
        parsed = json.loads(text)
    except Exception:
        return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "not_dict"
    weights = OrderedDict()
    for key, raw in parsed.items():
        number = _safe_float(raw, None)
        if number is None or math.isnan(number):
            weights[str(key)] = raw
        else:
            weights[str(key)] = float(number)
    return weights, "ok"


def normalize_review_history(history):
    if history is None or history.empty:
        return pd.DataFrame(columns=[*REVIEW_HISTORY_COLUMNS, "_reviewed_at_dt", "_row_order"])
    df = history.copy()
    for column in REVIEW_HISTORY_COLUMNS:
        if column not in df:
            df[column] = ""
    df = df[REVIEW_HISTORY_COLUMNS].copy()
    df["_reviewed_at_dt"] = df["reviewed_at"].map(parse_datetime)
    df["_row_order"] = range(len(df))
    return df


def load_review_history(path):
    return normalize_review_history(read_review_history(path))


def latest_review_decisions(history, game=None):
    df = normalize_review_history(history)
    columns = [
        *REVIEW_HISTORY_COLUMNS,
        "decision_label",
        "is_latest",
        "decision_rank_key",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    if game:
        df = df[df["game"].astype(str) == str(game)].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)
    df["decision_label"] = df["decision"].map(REVIEW_DECISION_LABELS).fillna(df["decision"])
    df["decision_rank_key"] = (
        df["game"].astype(str)
        + "__"
        + df["candidate_weights_hash"].astype(str)
        + "__"
        + df["weights_version"].astype(str)
    )
    df = df.sort_values(["decision_rank_key", "_reviewed_at_dt", "_row_order"], na_position="first")
    latest_index = df.groupby("decision_rank_key", dropna=False).tail(1).index
    df["is_latest"] = df.index.isin(latest_index)
    return df[columns]


def selectable_reviews(history, game=None):
    latest = latest_review_decisions(history, game=game)
    if latest.empty:
        return latest
    latest_rows = latest[latest["is_latest"]].copy()
    priority = {"candidate_for_adoption": 0, "hold": 1, "unreviewed": 2, "rejected": 3}
    latest_rows["_priority"] = latest_rows["decision"].map(priority).fillna(9)
    latest_rows = latest_rows.sort_values(["_priority", "reviewed_at", "candidate_type"], ascending=[True, False, True])
    return latest_rows.drop(columns=["_priority"], errors="ignore")


def adoption_overview(review_history, approval_history=None, game=None):
    reviews = selectable_reviews(review_history, game=game)
    if isinstance(approval_history, pd.DataFrame):
        approvals = approval_history.copy()
        for column in APPROVAL_HISTORY_COLUMNS:
            if column not in approvals:
                approvals[column] = ""
        approvals = approvals[APPROVAL_HISTORY_COLUMNS]
    else:
        approvals = read_approval_history(approval_history) if approval_history is not None else pd.DataFrame(columns=APPROVAL_HISTORY_COLUMNS)
    if game and not approvals.empty:
        approvals = approvals[approvals["game"].astype(str) == str(game)].copy()
    rows = []
    games = [game] if game else sorted(set(reviews.get("game", pd.Series(dtype=str)).astype(str)) | set(approvals.get("game", pd.Series(dtype=str)).astype(str)))
    for target_game in games:
        r = reviews[reviews["game"].astype(str) == str(target_game)] if not reviews.empty else pd.DataFrame()
        a = approvals[approvals["game"].astype(str) == str(target_game)] if not approvals.empty else pd.DataFrame()
        latest_approval = a.sort_values("approved_at").tail(1) if not a.empty and "approved_at" in a else pd.DataFrame()
        rows.append(
            {
                "game": target_game,
                "adoption_candidates": int((r.get("decision", pd.Series(dtype=str)) == "candidate_for_adoption").sum()) if not r.empty else 0,
                "dry_run_unapproved": int((r.get("decision", pd.Series(dtype=str)) == "candidate_for_adoption").sum()) if a.empty and not r.empty else max(0, int((r.get("decision", pd.Series(dtype=str)) == "candidate_for_adoption").sum()) - int((a.get("approval_status", pd.Series(dtype=str)) == "dry_run_approved").sum()) - int((a.get("approval_status", pd.Series(dtype=str)) == "approved_for_manual_preparation").sum())),
                "dry_run_approved": int((a.get("approval_status", pd.Series(dtype=str)) == "dry_run_approved").sum()) if not a.empty else 0,
                "manual_preparation_approved": int((a.get("approval_status", pd.Series(dtype=str)) == "approved_for_manual_preparation").sum()) if not a.empty else 0,
                "latest_approval_at": latest_approval.iloc[0].get("approved_at", "") if not latest_approval.empty else "",
                "latest_candidate_type": r.iloc[0].get("candidate_type", "") if not r.empty else "",
                "readiness_state": "needs_review" if not r.empty else "not_ready",
            }
        )
    return pd.DataFrame(rows)


def candidate_from_review_row(review_row):
    if review_row is None:
        return {}
    data = dict(review_row)
    weights, status = parse_candidate_weights_json(data.get("candidate_weights_json"))
    data["candidate_weights"] = weights or OrderedDict()
    data["candidate_weights_status"] = status
    return data


def compare_candidate_to_current(candidate_weights):
    current = OrderedDict((key, float(value)) for key, value in CURRENT_BALANCE_WEIGHTS.items())
    candidate = OrderedDict((str(key), value) for key, value in (candidate_weights or {}).items())
    rows = []
    for key in sorted(set(current) | set(candidate)):
        current_value = _safe_float(current.get(key), None)
        candidate_value = _safe_float(candidate.get(key), None)
        delta = None if current_value is None or candidate_value is None else candidate_value - current_value
        rows.append(
            {
                "subscore_key": key,
                "display_name": SUBSCORE_LABELS.get(key, key),
                "current_weight": current_value,
                "candidate_weight": candidate_value,
                "weight_delta": _round(delta, 6),
                "delta_percent": _round((delta / current_value * 100) if current_value not in (None, 0) and delta is not None else None, 3),
            }
        )
    return pd.DataFrame(rows)


def weight_diff_summary(diff_df):
    if diff_df is None or diff_df.empty:
        return pd.DataFrame(
            [
                {
                    "metric": "status",
                    "value": "no_candidate",
                }
            ]
        )
    changed = diff_df[pd.to_numeric(diff_df["weight_delta"], errors="coerce").abs().fillna(0) > 0.000001]
    candidate_sum = pd.to_numeric(diff_df["candidate_weight"], errors="coerce").sum()
    current_sum = pd.to_numeric(diff_df["current_weight"], errors="coerce").sum()
    max_inc = diff_df.sort_values("weight_delta", ascending=False).head(1)
    max_dec = diff_df.sort_values("weight_delta", ascending=True).head(1)
    rows = [
        {"metric": "current_weight_sum", "value": _round(current_sum, 6)},
        {"metric": "candidate_weight_sum", "value": _round(candidate_sum, 6)},
        {"metric": "normalization_error", "value": _round(abs(1.0 - candidate_sum), 6)},
        {"metric": "changed_items", "value": int(len(changed))},
        {"metric": "unchanged_items", "value": int(len(diff_df) - len(changed))},
        {"metric": "max_abs_delta", "value": _round(pd.to_numeric(diff_df["weight_delta"], errors="coerce").abs().max(), 6)},
        {"metric": "max_increase_item", "value": max_inc.iloc[0]["subscore_key"] if not max_inc.empty else ""},
        {"metric": "max_decrease_item", "value": max_dec.iloc[0]["subscore_key"] if not max_dec.empty else ""},
    ]
    return pd.DataFrame(rows)


def validate_candidate(candidate, game=None):
    weights = candidate.get("candidate_weights") if isinstance(candidate, dict) else {}
    review_hash = str(candidate.get("candidate_weights_hash", "") if isinstance(candidate, dict) else "")
    review_game = str(candidate.get("game", "") if isinstance(candidate, dict) else "")
    review_version = str(candidate.get("weights_version", "") if isinstance(candidate, dict) else "")
    json_status = candidate.get("candidate_weights_status", "empty") if isinstance(candidate, dict) else "empty"
    keys = set(weights or {})
    required = set(REQUIRED_WEIGHT_KEYS)
    rows = []

    def add(check, status, detail):
        rows.append({"check": check, "status": status, "detail": detail})

    if json_status == "ok" and isinstance(weights, dict):
        add("candidate_weights_json", "ready", "candidate_weights_json is a dictionary")
    else:
        add("candidate_weights_json", "blocked", json_status)

    missing = sorted(required - keys)
    extra = sorted(keys - required)
    add("required_keys", "ready" if not missing else "blocked", ",".join(missing) if missing else "all required keys exist")
    add("extra_keys", "ready" if not extra else "warning", ",".join(extra) if extra else "no extra keys")

    numeric_values = []
    non_numeric = []
    negative = []
    high = []
    low = []
    for key, value in (weights or {}).items():
        number = _safe_float(value, None)
        if number is None or math.isnan(number):
            non_numeric.append(key)
            continue
        numeric_values.append(number)
        if number < 0:
            negative.append(key)
        if number > 0.35:
            high.append(key)
        if number < 0.0:
            low.append(key)
    total = sum(numeric_values)
    add("numeric_values", "ready" if not non_numeric else "blocked", ",".join(non_numeric) if non_numeric else "all weights are numeric")
    add("non_negative", "ready" if not negative else "blocked", ",".join(negative) if negative else "no negative weights")
    add("weight_sum", "ready" if 0.95 <= total <= 1.05 else "blocked", _round(total, 6))
    add("upper_bound", "ready" if not high else "warning", ",".join(high) if high else "no unusually high weights")
    add("lower_bound", "ready" if not low else "blocked", ",".join(low) if low else "no below-minimum weights")

    recalculated_hash = candidate_weights_hash(weights) if isinstance(weights, dict) and not non_numeric else ""
    add("candidate_hash", "ready" if review_hash and recalculated_hash == review_hash else "blocked", f"stored={review_hash} recalculated={recalculated_hash}")
    add("weights_version", "ready" if review_version == BALANCE_WEIGHT_RESEARCH_VERSION else "warning", review_version or "missing")
    add("game", "ready" if not game or review_game == game else "blocked", review_game or "missing")
    add("review_decision", "ready" if str(candidate.get("decision", "")) in REVIEW_DECISION_LABELS else "warning", str(candidate.get("decision", "")))
    add("sample_count", "ready" if _safe_int(candidate.get("sample_count"), 0) > 0 else "warning", _safe_int(candidate.get("sample_count"), 0))
    df = pd.DataFrame(rows)
    if (df["status"] == "blocked").any():
        overall = "blocked"
    elif (df["status"] == "warning").any():
        overall = "warning"
    else:
        overall = "ready"
    return overall, df


def _weights_frame_for_scenarios(candidate_weights):
    rows = []
    for scenario, weights in (("current", CURRENT_BALANCE_WEIGHTS), ("selected_candidate", candidate_weights or {})):
        for key in REQUIRED_WEIGHT_KEYS:
            rows.append(
                {
                    "scenario": scenario,
                    "subscore_key": key,
                    "candidate_weight": _safe_float(weights.get(key), 0.0),
                }
            )
    return pd.DataFrame(rows)


def run_adoption_dry_run(reports, candidate, game=None):
    overall, safety = validate_candidate(candidate, game=game)
    if overall == "blocked":
        return {
            "status": "blocked",
            "safety_checks": safety,
            "summary": pd.DataFrame(),
            "per_draw": pd.DataFrame(),
            "rolling": pd.DataFrame(),
            "drift": pd.DataFrame(),
            "ranking": pd.DataFrame(),
        }
    weights = candidate.get("candidate_weights", OrderedDict())
    weights_df = _weights_frame_for_scenarios(weights)
    simulation = simulate_candidate_weights(reports, weights_df)
    ranking = build_per_draw_candidate_ranking(reports, weights_df)
    per_draw_summary = build_per_draw_ranking_summary(ranking)
    stability = build_ranking_stability_summary(ranking, per_draw_summary)
    summary = summarize_dry_run(simulation, per_draw_summary, stability)
    rolling = compare_rolling_for_candidate(reports, candidate)
    drift = build_drift_summary(candidate, summary, rolling)
    per_draw = build_per_draw_comparison(per_draw_summary)
    return {
        "status": overall,
        "safety_checks": safety,
        "summary": summary,
        "per_draw": per_draw,
        "rolling": rolling,
        "drift": drift,
        "ranking": stability,
    }


def summarize_dry_run(simulation, per_draw_summary, stability):
    rows = []
    sim = simulation.set_index("scenario") if simulation is not None and not simulation.empty and "scenario" in simulation else pd.DataFrame()
    for metric in ("evaluation_count", "high_score_avg_match", "low_score_avg_match", "high_low_match_gap", "hit3_rate", "top_candidate_avg_match"):
        current_value = sim.loc["current", metric] if "current" in sim.index and metric in sim else None
        candidate_value = sim.loc["selected_candidate", metric] if "selected_candidate" in sim.index and metric in sim else None
        rows.append(
            {
                "metric": metric,
                "current_value": current_value,
                "candidate_value": candidate_value,
                "delta": _round(_safe_float(candidate_value, 0.0) - _safe_float(current_value, 0.0)) if metric != "hit3_rate" else "",
                "status": "reference",
                "note": "read-only dry-run",
            }
        )
    if per_draw_summary is not None and not per_draw_summary.empty:
        for metric, source in (
            ("top1_mean_match", "top_candidate_average_match"),
            ("top3_mean_match", "top3_candidate_average_match"),
            ("best_candidate_top1_rate", "best_actual_is_rank1"),
            ("best_candidate_top3_rate", "best_actual_within_top3"),
        ):
            current = per_draw_summary[per_draw_summary["candidate_type"] == "current"]
            candidate = per_draw_summary[per_draw_summary["candidate_type"] == "selected_candidate"]
            current_value = pd.to_numeric(current[source], errors="coerce").mean() if not current.empty else None
            candidate_value = pd.to_numeric(candidate[source], errors="coerce").mean() if not candidate.empty else None
            rows.append(
                {
                    "metric": metric,
                    "current_value": _round(current_value),
                    "candidate_value": _round(candidate_value),
                    "delta": _round(_safe_float(candidate_value, 0.0) - _safe_float(current_value, 0.0)),
                    "status": "reference",
                    "note": "same draw ranking comparison",
                }
            )
    if stability is not None and not stability.empty:
        stab = stability.set_index("candidate_type")
        for metric in ("ranking_stability_score", "tie_rate", "top_candidate_switch_rate"):
            current_value = stab.loc["current", metric] if "current" in stab.index and metric in stab else None
            candidate_value = stab.loc["selected_candidate", metric] if "selected_candidate" in stab.index and metric in stab else None
            rows.append(
                {
                    "metric": metric,
                    "current_value": current_value,
                    "candidate_value": candidate_value,
                    "delta": "",
                    "status": "reference",
                    "note": "ranking stability",
                }
            )
    return pd.DataFrame(rows)


def build_per_draw_comparison(per_draw_summary):
    columns = [
        "draw_no",
        "candidate_count",
        "current_top_candidate",
        "candidate_top_candidate",
        "current_top_match",
        "candidate_top_match",
        "best_actual_candidate",
        "current_best_rank",
        "candidate_best_rank",
        "rank_improved",
        "rank_worsened",
        "top_candidate_changed",
    ]
    if per_draw_summary is None or per_draw_summary.empty:
        return pd.DataFrame(columns=columns)
    current = per_draw_summary[per_draw_summary["candidate_type"] == "current"].set_index("draw_no")
    candidate = per_draw_summary[per_draw_summary["candidate_type"] == "selected_candidate"].set_index("draw_no")
    rows = []
    for draw_no in sorted(set(current.index) & set(candidate.index)):
        c = current.loc[draw_no]
        s = candidate.loc[draw_no]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[0]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[0]
        current_rank = _safe_int(c.get("best_actual_hit_candidate_rank"), 0)
        candidate_rank = _safe_int(s.get("best_actual_hit_candidate_rank"), 0)
        rows.append(
            {
                "draw_no": draw_no,
                "candidate_count": c.get("candidate_count", ""),
                "current_top_candidate": c.get("top_candidate_no", ""),
                "candidate_top_candidate": s.get("top_candidate_no", ""),
                "current_top_match": c.get("top_candidate_main_match_count", ""),
                "candidate_top_match": s.get("top_candidate_main_match_count", ""),
                "best_actual_candidate": c.get("best_actual_candidate_no", ""),
                "current_best_rank": current_rank,
                "candidate_best_rank": candidate_rank,
                "rank_improved": bool(candidate_rank and current_rank and candidate_rank < current_rank),
                "rank_worsened": bool(candidate_rank and current_rank and candidate_rank > current_rank),
                "top_candidate_changed": str(c.get("top_candidate_no", "")) != str(s.get("top_candidate_no", "")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compare_rolling_for_candidate(reports, candidate):
    rolling = build_rolling_weight_evaluation(reports)
    columns = [
        "window_size",
        "evaluation_count",
        "current_top1_mean_match",
        "candidate_top1_mean_match",
        "top1_delta",
        "current_top3_mean_match",
        "candidate_top3_mean_match",
        "top3_delta",
        "current_best_top1_rate",
        "candidate_best_top1_rate",
        "best_top1_rate_delta",
        "scope_note",
    ]
    if rolling is None or rolling.empty:
        return pd.DataFrame(columns=columns)
    selected_type = str(candidate.get("candidate_type", ""))
    rows = []
    for window, group in rolling.groupby("window_size"):
        current = group[group["candidate_type"] == "current"].tail(1)
        candidate_rows = group[group["candidate_type"] == selected_type].tail(1)
        if candidate_rows.empty:
            candidate_rows = group[group["candidate_type"] == "selected_candidate"].tail(1)
        if current.empty or candidate_rows.empty:
            continue
        c = current.iloc[0]
        s = candidate_rows.iloc[0]
        rows.append(
            {
                "window_size": int(window),
                "evaluation_count": _safe_int(s.get("evaluation_count"), 0),
                "current_top1_mean_match": c.get("top1_mean_match", ""),
                "candidate_top1_mean_match": s.get("top1_mean_match", ""),
                "top1_delta": _round(_safe_float(s.get("top1_mean_match"), 0.0) - _safe_float(c.get("top1_mean_match"), 0.0)),
                "current_top3_mean_match": c.get("top3_mean_match", ""),
                "candidate_top3_mean_match": s.get("top3_mean_match", ""),
                "top3_delta": _round(_safe_float(s.get("top3_mean_match"), 0.0) - _safe_float(c.get("top3_mean_match"), 0.0)),
                "current_best_top1_rate": c.get("best_candidate_top1_rate", ""),
                "candidate_best_top1_rate": s.get("best_candidate_top1_rate", ""),
                "best_top1_rate_delta": _round(_safe_float(s.get("best_candidate_top1_rate"), 0.0) - _safe_float(c.get("best_candidate_top1_rate"), 0.0)),
                "scope_note": "current available data; no production write",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_drift_summary(candidate, dry_run_summary, rolling_summary):
    saved = {
        "top1_mean_match": _safe_float(candidate.get("top1_mean_match"), None),
        "top3_mean_match": _safe_float(candidate.get("top3_mean_match"), None),
        "best_candidate_top1_rate": _safe_float(candidate.get("best_candidate_top1_rate"), None),
        "rolling_evaluation_count": _safe_float(candidate.get("rolling_evaluation_count"), 0),
        "sample_count": _safe_float(candidate.get("sample_count"), 0),
    }
    current = {}
    if dry_run_summary is not None and not dry_run_summary.empty:
        by_metric = dry_run_summary.set_index("metric")
        for metric in ("top1_mean_match", "top3_mean_match", "best_candidate_top1_rate"):
            if metric in by_metric.index:
                current[metric] = _safe_float(by_metric.loc[metric].get("candidate_value"), None)
    current["rolling_evaluation_count"] = pd.to_numeric(rolling_summary.get("evaluation_count", pd.Series(dtype=float)), errors="coerce").max() if rolling_summary is not None and not rolling_summary.empty else 0
    current["sample_count"] = saved["sample_count"]
    rows = []
    for metric in ("top1_mean_match", "top3_mean_match", "best_candidate_top1_rate", "rolling_evaluation_count", "sample_count"):
        saved_value = saved.get(metric)
        current_value = current.get(metric)
        drift = None if saved_value is None or current_value is None else _safe_float(current_value, 0.0) - _safe_float(saved_value, 0.0)
        if drift is None:
            status = "insufficient_data"
        elif abs(drift) < 0.05:
            status = "stable"
        elif abs(drift) < 0.25:
            status = "slightly_changed"
        else:
            status = "materially_changed"
        rows.append(
            {
                "metric": f"{metric}_drift",
                "review_saved_value": saved_value,
                "current_value": _round(current_value),
                "drift": _round(drift),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def readiness_score(candidate, safety_checks, dry_run_summary, rolling_summary, drift_summary, approval_history=None):
    score = 0.0
    reasons = []
    decision = str(candidate.get("decision", ""))
    if decision == "candidate_for_adoption":
        score += 20
    elif decision == "hold":
        score += 8
        reasons.append("review decision is hold")
    elif decision == "rejected":
        reasons.append("review decision is rejected")
    else:
        score += 4
    sample_count = _safe_int(candidate.get("sample_count"), 0)
    score += min(20, sample_count / max(MIN_WEIGHT_RECOMMENDATION_SAMPLES, 1) * 20)
    rolling_count = _safe_int(candidate.get("rolling_evaluation_count"), 0)
    score += min(10, rolling_count * 2)
    if safety_checks is not None and not safety_checks.empty:
        if (safety_checks["status"] == "blocked").any():
            reasons.append("safety check is blocked")
            score = min(score, 30)
        elif (safety_checks["status"] == "warning").any():
            score += 8
        else:
            score += 15
    if dry_run_summary is not None and not dry_run_summary.empty:
        deltas = pd.to_numeric(dry_run_summary.get("delta", pd.Series(dtype=float)), errors="coerce").dropna()
        if not deltas.empty and deltas.mean() > 0:
            score += 10
        else:
            score += 4
    if rolling_summary is not None and not rolling_summary.empty:
        top1_delta = pd.to_numeric(rolling_summary.get("top1_delta", pd.Series(dtype=float)), errors="coerce").mean()
        score += 8 if not pd.isna(top1_delta) and top1_delta >= 0 else 3
    if drift_summary is not None and not drift_summary.empty:
        statuses = set(drift_summary["status"].astype(str))
        if "materially_changed" in statuses:
            reasons.append("material drift detected")
            score -= 15
        elif "slightly_changed" in statuses:
            score += 4
        elif "stable" in statuses:
            score += 8
    if "experimental" in str(candidate.get("candidate_type", "")).lower():
        reasons.append("experimental candidate requires special care")
        score = min(score, 74)
    approval_statuses = set()
    if approval_history is not None and not approval_history.empty:
        approval_statuses = set(approval_history.get("approval_status", pd.Series(dtype=str)).astype(str))
    if "dry_run_approved" in approval_statuses:
        score += 5
    if "approved_for_manual_preparation" in approval_statuses:
        score += 8
    score = max(0.0, min(100.0, score))
    if score >= 85:
        state = "ready_for_manual_change"
    elif score >= 65:
        state = "provisionally_ready"
    elif score >= 35:
        state = "needs_review"
    else:
        state = "not_ready"
    return pd.DataFrame(
        [
            {
                "adoption_readiness_score": _round(score),
                "readiness_state": state,
                "decision": decision,
                "sample_count": sample_count,
                "rolling_evaluation_count": rolling_count,
                "note": "; ".join(reasons) if reasons else "manual review required before any change",
            }
        ]
    )


def configuration_preview(candidate, target_file="loto_lab/core/balance_hypothesis_engine.py"):
    weights = candidate.get("candidate_weights", OrderedDict())
    diff = compare_candidate_to_current(weights)
    rows = []
    for _, row in diff.iterrows():
        if _safe_float(row.get("weight_delta"), 0.0) == 0:
            continue
        rows.append(
            {
                "target_file_candidate": target_file,
                "target_key": row["subscore_key"],
                "current_value": row["current_weight"],
                "candidate_value": row["candidate_weight"],
                "delta": row["weight_delta"],
                "weights_version_candidate": f"{BALANCE_WEIGHT_RESEARCH_VERSION}_manual_candidate",
                "reason": "manual dry-run preview only; no file write",
            }
        )
    return pd.DataFrame(rows)


def patch_preview_text(candidate):
    weights = candidate.get("candidate_weights", OrderedDict())
    current_lines = [_stable_json(dict(CURRENT_BALANCE_WEIGHTS))]
    candidate_lines = [_stable_json(dict((key, _safe_float(weights.get(key), 0.0)) for key in REQUIRED_WEIGHT_KEYS))]
    return "\n".join(difflib.unified_diff(current_lines, candidate_lines, fromfile="current_weights", tofile="candidate_weights", lineterm=""))


def rollback_plan(candidate, current_commit=""):
    return {
        "current_weights_version": BALANCE_WEIGHT_RESEARCH_VERSION,
        "current_weights": dict(CURRENT_BALANCE_WEIGHTS),
        "candidate_weights_version": candidate.get("weights_version", ""),
        "candidate_weights": dict(candidate.get("candidate_weights", {})),
        "target_files": ["loto_lab/core/balance_hypothesis_engine.py"],
        "restore_values": dict(CURRENT_BALANCE_WEIGHTS),
        "verification_tests": [
            ".venv\\Scripts\\python.exe -m compileall loto_lab",
            "git diff --check",
            ".venv\\Scripts\\python.exe -m loto_lab.core.streamlit_smoke_check",
        ],
        "git_rollback_candidates": [
            "record pre-change commit id before manual change",
            "use git revert <manual-change-commit> if a committed manual change must be undone",
            "do not roll back research data CSV/JSONL automatically",
        ],
        "current_commit": current_commit,
        "note": "review package only; no automatic rollback is executed",
    }


def read_approval_history(path):
    if path is None:
        return pd.DataFrame(columns=APPROVAL_HISTORY_COLUMNS)
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=APPROVAL_HISTORY_COLUMNS)
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            df = pd.read_csv(target, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(target)
    for column in APPROVAL_HISTORY_COLUMNS:
        if column not in df:
            df[column] = ""
    return df[APPROVAL_HISTORY_COLUMNS]


def build_approval_row(game, candidate, approval_status, approval_comment, readiness, dry_run_summary, drift_summary, safety_checks, approver=""):
    if approval_status not in APPROVAL_STATUSES:
        approval_status = "unapproved"
    readiness_score_value = ""
    if readiness is not None and not readiness.empty:
        readiness_score_value = readiness.iloc[0].get("adoption_readiness_score", "")
    approval_id = f"approval-{game}-{candidate.get('candidate_weights_hash', '')[:12]}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return {
        "approval_id": approval_id,
        "approved_at": now_text(),
        "game": game,
        "review_id": candidate.get("review_id", ""),
        "candidate_weights_hash": candidate.get("candidate_weights_hash", ""),
        "approval_status": approval_status,
        "approval_comment": str(approval_comment or ""),
        "readiness_score": readiness_score_value,
        "dry_run_summary_json": dataframe_records_json(dry_run_summary),
        "drift_summary_json": dataframe_records_json(drift_summary),
        "safety_checks_json": dataframe_records_json(safety_checks),
        "approver": str(approver or ""),
    }


def append_approval_history(path, approval_row):
    if not approval_row:
        return False, "approval row is empty"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    history = read_approval_history(target)
    row_df = pd.DataFrame([{column: approval_row.get(column, "") for column in APPROVAL_HISTORY_COLUMNS}])
    updated = pd.concat([history, row_df], ignore_index=True)
    updated.to_csv(target, index=False, encoding="utf-8-sig")
    return True, f"saved approval history: {approval_row.get('approval_id', '')}"


def dataframe_records_json(df):
    if df is None or df.empty:
        return "[]"
    return json.dumps(df.to_dict(orient="records"), ensure_ascii=False, default=str)


def build_adoption_package(game, candidate, dry_run, diff_df, diff_summary, readiness, config_preview, rollback, approvals=None):
    package = {
        "package_version": ADOPTION_DRY_RUN_VERSION,
        "generated_at": now_text(),
        "game": game,
        "review_id": candidate.get("review_id", ""),
        "reviewed_at": candidate.get("reviewed_at", ""),
        "candidate_type": candidate.get("candidate_type", ""),
        "decision": candidate.get("decision", ""),
        "review_comment": candidate.get("review_comment", ""),
        "candidate_weights_hash": candidate.get("candidate_weights_hash", ""),
        "current_weights_version": BALANCE_WEIGHT_RESEARCH_VERSION,
        "candidate_weights_version": candidate.get("weights_version", ""),
        "current_weights": dict(CURRENT_BALANCE_WEIGHTS),
        "candidate_weights": dict(candidate.get("candidate_weights", {})),
        "weight_diff": diff_df.to_dict(orient="records") if diff_df is not None else [],
        "weight_diff_summary": diff_summary.to_dict(orient="records") if diff_summary is not None else [],
        "dry_run_summary": dry_run.get("summary", pd.DataFrame()).to_dict(orient="records") if dry_run else [],
        "rolling_summary": dry_run.get("rolling", pd.DataFrame()).to_dict(orient="records") if dry_run else [],
        "ranking_summary": dry_run.get("per_draw", pd.DataFrame()).to_dict(orient="records") if dry_run else [],
        "drift_summary": dry_run.get("drift", pd.DataFrame()).to_dict(orient="records") if dry_run else [],
        "readiness_summary": readiness.to_dict(orient="records") if readiness is not None else [],
        "safety_checks": dry_run.get("safety_checks", pd.DataFrame()).to_dict(orient="records") if dry_run else [],
        "configuration_preview": config_preview.to_dict(orient="records") if config_preview is not None else [],
        "rollback_plan": rollback,
        "approval_status": approvals.to_dict(orient="records") if approvals is not None and not approvals.empty else [],
        "production_note": "manual preparation package only; no production weight is changed",
    }
    return package


def adoption_package_json_bytes(package):
    return json.dumps(package, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def dry_run_report_csv_bytes(package):
    rows = []
    for section in ("weight_diff", "dry_run_summary", "rolling_summary", "ranking_summary", "drift_summary", "safety_checks", "configuration_preview", "approval_status"):
        values = package.get(section, [])
        for item in values:
            if isinstance(item, dict):
                rows.append(
                    {
                        "section": section,
                        "metric": item.get("metric", item.get("check", item.get("target_key", item.get("approval_status", "")))),
                        "current_value": item.get("current_value", item.get("current_weight", "")),
                        "candidate_value": item.get("candidate_value", item.get("candidate_weight", "")),
                        "delta": item.get("delta", item.get("weight_delta", "")),
                        "status": item.get("status", ""),
                        "note": item.get("note", item.get("detail", "")),
                    }
                )
    rows.append({"section": "rollback", "metric": "rollback_plan", "current_value": "", "candidate_value": "", "delta": "", "status": "reference", "note": _stable_json(package.get("rollback_plan", {}))})
    df = pd.DataFrame(rows, columns=["section", "metric", "current_value", "candidate_value", "delta", "status", "note"])
    return ("\ufeff" + df.to_csv(index=False)).encode("utf-8")
