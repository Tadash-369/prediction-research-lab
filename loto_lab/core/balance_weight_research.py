from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


BALANCE_WEIGHT_RESEARCH_VERSION = "balance_weight_research_v1"
BALANCE_WEIGHT_REVIEW_VERSION = "balance_weight_review_v1"
MIN_WEIGHT_RESEARCH_SAMPLES = 10
MIN_WEIGHT_RECOMMENDATION_SAMPLES = 20
MAX_CONSERVATIVE_WEIGHT_DELTA = 0.02
MAX_BALANCED_WEIGHT_DELTA = 0.05
MAX_EXPERIMENTAL_WEIGHT_DELTA = 0.10
HIGH_SCORE_THRESHOLD = 75.0

SUBSCORE_LABELS = OrderedDict(
    [
        ("odd_even", "奇数・偶数"),
        ("high_low", "高低バランス"),
        ("sum", "合計値"),
        ("consecutive", "連番"),
        ("last_digit", "下一桁"),
        ("tens_group", "十の位"),
        ("hot_cold", "ホット・コールド"),
        ("gap", "出現間隔"),
        ("bonus_neighbor", "ボーナス数字周辺"),
        ("ball_set", "セット球"),
    ]
)

CURRENT_BALANCE_WEIGHTS = OrderedDict(
    [
        ("odd_even", 0.11),
        ("high_low", 0.11),
        ("sum", 0.13),
        ("consecutive", 0.09),
        ("last_digit", 0.08),
        ("tens_group", 0.08),
        ("hot_cold", 0.13),
        ("gap", 0.11),
        ("bonus_neighbor", 0.08),
        ("ball_set", 0.08),
    ]
)


def _safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(float(str(value).replace(",", "").strip()))
    except Exception:
        return default


def _round(value, digits=3):
    number = _safe_float(value)
    if number is None or math.isnan(number):
        return None
    return round(number, digits)


def _clamp(value, low=0.0, high=100.0):
    return max(low, min(high, float(value)))


def _percent(value):
    return f"{round(float(value) * 100, 1)}%"


def _find_column(df, candidates=None, fallback_index=None, contains=None):
    if df is None or df.empty:
        return None
    candidates = candidates or []
    for column in candidates:
        if column in df:
            return column
    if contains:
        lowered = [str(token).lower() for token in contains]
        for column in df.columns:
            text = str(column).lower()
            if any(token in text for token in lowered):
                return column
    if fallback_index is not None and 0 <= fallback_index < len(df.columns):
        return df.columns[fallback_index]
    return None


def _read_json(value):
    if value is None or str(value).strip() == "":
        return {}
    try:
        if pd.isna(value):
            return {}
    except Exception:
        pass
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {"_invalid_json": True}
    return parsed if isinstance(parsed, dict) else {}


def _subscore_from_details(details, key):
    if not isinstance(details, dict):
        return None
    candidates = [key, f"{key}_score", f"balance_{key}", f"{key}_balance_score"]
    for candidate in candidates:
        if candidate in details:
            return _safe_float(details.get(candidate))
    for nested_key in ("subscores", "scores", "balance_subscores", "details"):
        nested = details.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for candidate in candidates:
            if candidate in nested:
                return _safe_float(nested.get(candidate))
    return None


def _is_chamini_sp_model(value):
    text = str(value or "").lower()
    return any(token in text for token in ("chamini_sp", "chaminisp", "chamini6", "god mode"))


def _report_columns(df):
    key_col = _find_column(df, ["検証キー"], fallback_index=0, contains=["検証"])
    key_first = bool(df is not None and len(df.columns) and key_col == df.columns[0])
    return {
        "key": key_col,
        "id": _find_column(df, ["予想ID"], fallback_index=1 if key_first else 0, contains=["予想id", "prediction_id"]),
        "draw": _find_column(df, ["開催回"], fallback_index=2 if key_first else 1, contains=["開催", "draw"]),
        "candidate": _find_column(df, ["候補番号"], fallback_index=5 if key_first else 3, contains=["候補", "candidate"]),
        "model": _find_column(df, ["使用モデル"], fallback_index=6 if key_first else 3, contains=["使用モデル", "model"]),
        "main_match": _find_column(df, ["本数字一致数", "main_match"], fallback_index=11 if key_first else 7, contains=["本数字一致", "main_match"]),
        "bonus_match": _find_column(df, ["ボーナス一致数", "bonus_match"], fallback_index=14 if key_first else 9, contains=["ボーナス一致", "bonus_match"]),
        "total_match": _find_column(df, ["総一致数", "total_match"], contains=["総一致", "total_match"]),
        "balance_score": _find_column(df, ["balance_score"]),
        "balance_grade": _find_column(df, ["balance_grade"]),
        "details": _find_column(df, ["balance_details_json"]),
    }


def _now_text():
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def _stable_json(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _csv_text(df):
    if df is None or df.empty:
        return ""
    return df.to_csv(index=False, encoding="utf-8-sig")


def get_current_balance_weights():
    rows = []
    for key, weight in CURRENT_BALANCE_WEIGHTS.items():
        rows.append(
            {
                "subscore_key": key,
                "display_name": SUBSCORE_LABELS.get(key, key),
                "current_weight": round(weight, 4),
                "weights_version": BALANCE_WEIGHT_RESEARCH_VERSION,
            }
        )
    return pd.DataFrame(rows)


def _prepare_research_rows(reports):
    columns = [
        "verification_key",
        "prediction_id",
        "draw_no",
        "candidate_no",
        "model_name",
        "main_match",
        "bonus_match",
        "total_match",
        "balance_score",
        "balance_grade",
        "details_status",
        *SUBSCORE_LABELS.keys(),
    ]
    if reports is None or reports.empty:
        return pd.DataFrame(columns=columns)
    df = reports.copy()
    cols = _report_columns(df)
    model_col = cols.get("model")
    if model_col in df:
        mask = df[model_col].map(_is_chamini_sp_model)
        if mask.any():
            df = df[mask].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for _, row in df.iterrows():
        main_match = _safe_float(row.get(cols.get("main_match")), 0.0) if cols.get("main_match") in df else 0.0
        bonus_match = _safe_float(row.get(cols.get("bonus_match")), 0.0) if cols.get("bonus_match") in df else 0.0
        total_match = _safe_float(row.get(cols.get("total_match")), None) if cols.get("total_match") in df else None
        if total_match is None:
            total_match = main_match + bonus_match
        details = _read_json(row.get(cols.get("details"))) if cols.get("details") in df else {}
        if details.get("_invalid_json"):
            details_status = "invalid_json"
        elif details.get("balance_not_evaluated") in (True, "1", 1, "true", "True"):
            details_status = "not_evaluated"
        elif details:
            details_status = "ok"
        else:
            details_status = "missing"
        prepared = {
            "verification_key": row.get(cols.get("key"), "") if cols.get("key") in df else "",
            "prediction_id": row.get(cols.get("id"), "") if cols.get("id") in df else "",
            "draw_no": _safe_int(row.get(cols.get("draw"))) if cols.get("draw") in df else 0,
            "candidate_no": _safe_int(row.get(cols.get("candidate"))) if cols.get("candidate") in df else 0,
            "model_name": row.get(cols.get("model"), "") if cols.get("model") in df else "",
            "main_match": main_match,
            "bonus_match": bonus_match,
            "total_match": total_match,
            "balance_score": _safe_float(row.get(cols.get("balance_score"))) if cols.get("balance_score") in df else None,
            "balance_grade": row.get(cols.get("balance_grade"), "") if cols.get("balance_grade") in df else "",
            "details_status": details_status,
        }
        for key in SUBSCORE_LABELS:
            subscore = _subscore_from_details(details, key)
            if subscore is None and key in df:
                subscore = _safe_float(row.get(key))
            prepared[key] = subscore
        rows.append(prepared)
    return pd.DataFrame(rows, columns=columns)


def _confidence_level(count):
    if count < 5:
        return "insufficient"
    if count < MIN_WEIGHT_RESEARCH_SAMPLES:
        return "reference_only"
    if count < MIN_WEIGHT_RECOMMENDATION_SAMPLES:
        return "provisional"
    return "research_ready"


def _reliability_score(count):
    if count <= 0:
        return 0.0
    if count < MIN_WEIGHT_RESEARCH_SAMPLES:
        return round((count / MIN_WEIGHT_RESEARCH_SAMPLES) * 40, 3)
    if count < MIN_WEIGHT_RECOMMENDATION_SAMPLES:
        return round(40 + ((count - MIN_WEIGHT_RESEARCH_SAMPLES) / MIN_WEIGHT_RECOMMENDATION_SAMPLES) * 35, 3)
    return round(min(100.0, 75 + math.log1p(count - MIN_WEIGHT_RECOMMENDATION_SAMPLES) * 8), 3)


def evaluate_subscore_research(reports):
    prepared = _prepare_research_rows(reports)
    total_rows = len(prepared)
    rows = []
    for key, label in SUBSCORE_LABELS.items():
        current_weight = CURRENT_BALANCE_WEIGHTS[key]
        if total_rows == 0:
            rows.append(
                {
                    "subscore_key": key,
                    "display_name": label,
                    "current_weight": current_weight,
                    "evaluation_count": 0,
                    "confidence_level": "insufficient",
                    "average_subscore": None,
                    "high_score_avg_match": None,
                    "low_score_avg_match": None,
                    "performance_gap_score": 0.0,
                    "sample_reliability_score": 0.0,
                    "stability_score": 0.0,
                    "recent_performance_score": 0.0,
                    "long_term_performance_score": 0.0,
                    "availability_score": 0.0,
                    "correlation_reference_score": 0.0,
                    "research_contribution_score": 0.0,
                    "not_evaluated_rate": "0.0%",
                    "rationale": "検証データが不足しています",
                }
            )
            continue

        series = pd.to_numeric(prepared[key], errors="coerce")
        valid = prepared[series.notna()].copy()
        count = len(valid)
        availability = count / total_rows if total_rows else 0.0
        reliability = _reliability_score(count)
        confidence = _confidence_level(count)
        if count == 0:
            rows.append(
                {
                    "subscore_key": key,
                    "display_name": label,
                    "current_weight": current_weight,
                    "evaluation_count": 0,
                    "confidence_level": confidence,
                    "average_subscore": None,
                    "high_score_avg_match": None,
                    "low_score_avg_match": None,
                    "performance_gap_score": 0.0,
                    "sample_reliability_score": reliability,
                    "stability_score": 0.0,
                    "recent_performance_score": 0.0,
                    "long_term_performance_score": 0.0,
                    "availability_score": round(availability * 100, 3),
                    "correlation_reference_score": 0.0,
                    "research_contribution_score": 0.0,
                    "not_evaluated_rate": _percent(1.0),
                    "rationale": "サブスコア詳細が保存されていないため評価不能です",
                }
            )
            continue

        valid["_subscore"] = pd.to_numeric(valid[key], errors="coerce")
        high = valid[valid["_subscore"] >= HIGH_SCORE_THRESHOLD]
        low = valid[valid["_subscore"] < HIGH_SCORE_THRESHOLD]
        high_avg = high["main_match"].mean() if not high.empty else None
        low_avg = low["main_match"].mean() if not low.empty else None
        gap = (high_avg - low_avg) if high_avg is not None and low_avg is not None else 0.0
        performance_gap_score = _clamp(50 + gap * 22)

        ordered = valid.sort_values("draw_no")
        recent = ordered.tail(min(5, len(ordered)))
        recent_avg = recent["main_match"].mean() if not recent.empty else 0.0
        long_avg = ordered["main_match"].mean() if not ordered.empty else 0.0
        recent_performance_score = _clamp(50 + (recent_avg - long_avg) * 24)
        long_term_performance_score = _clamp(50 + (long_avg - 1.5) * 18)
        stability_score = _clamp(100 - abs(recent_avg - long_avg) * 35) if count >= 2 else 0.0

        correlation_score = 0.0
        if count >= MIN_WEIGHT_RESEARCH_SAMPLES and valid["_subscore"].nunique() > 1 and valid["main_match"].nunique() > 1:
            correlation = valid["_subscore"].corr(valid["main_match"])
            if not pd.isna(correlation):
                correlation_score = _clamp(abs(float(correlation)) * 100)

        availability_score = availability * 100
        contribution = (
            performance_gap_score * 0.30
            + reliability * 0.20
            + stability_score * 0.18
            + recent_performance_score * 0.12
            + long_term_performance_score * 0.10
            + availability_score * 0.06
            + correlation_score * 0.04
        )
        if count < MIN_WEIGHT_RESEARCH_SAMPLES:
            contribution = min(contribution, 55.0)
        if count < 5:
            contribution = min(contribution, 35.0)
        not_evaluated_rate = 1 - availability
        rows.append(
            {
                "subscore_key": key,
                "display_name": label,
                "current_weight": round(current_weight, 4),
                "evaluation_count": int(count),
                "confidence_level": confidence,
                "average_subscore": _round(valid["_subscore"].mean()),
                "high_score_avg_match": _round(high_avg),
                "low_score_avg_match": _round(low_avg),
                "performance_gap_score": _round(performance_gap_score),
                "sample_reliability_score": _round(reliability),
                "stability_score": _round(stability_score),
                "recent_performance_score": _round(recent_performance_score),
                "long_term_performance_score": _round(long_term_performance_score),
                "availability_score": _round(availability_score),
                "correlation_reference_score": _round(correlation_score),
                "research_contribution_score": _round(contribution),
                "not_evaluated_rate": _percent(not_evaluated_rate),
                "rationale": _rationale(label, confidence, gap, recent_avg, long_avg, not_evaluated_rate),
            }
        )
    return pd.DataFrame(rows)


def _rationale(label, confidence, gap, recent_avg, long_avg, not_evaluated_rate):
    if confidence in ("insufficient", "reference_only"):
        return f"{label}: サンプル不足のため参考値です。"
    parts = [f"{label}: "]
    if gap > 0:
        parts.append("高スコア群が低スコア群を上回っています。")
    elif gap < 0:
        parts.append("低スコア群が高スコア群を上回っています。")
    else:
        parts.append("高低スコア群の差は小さいです。")
    if recent_avg > long_avg:
        parts.append("直近成績は長期より強めです。")
    elif recent_avg < long_avg:
        parts.append("直近成績は長期より弱めです。")
    if not_evaluated_rate >= 0.5:
        parts.append("評価不能率が高いため変更は保守的に扱います。")
    return "".join(parts)


def _scenario_delta(row, max_delta):
    score = _safe_float(row.get("research_contribution_score"), 0.0)
    count = _safe_int(row.get("evaluation_count"), 0)
    not_evaluated_text = str(row.get("not_evaluated_rate", "0")).replace("%", "")
    not_evaluated_rate = _safe_float(not_evaluated_text, 0.0) / 100.0
    if count < MIN_WEIGHT_RESEARCH_SAMPLES:
        return 0.0
    reliability_factor = 0.5 if count < MIN_WEIGHT_RECOMMENDATION_SAMPLES else 1.0
    missing_factor = max(0.2, 1.0 - not_evaluated_rate)
    signal = max(-1.0, min(1.0, (score - 50.0) / 50.0))
    return signal * max_delta * reliability_factor * missing_factor


def _normalize_weights(weights):
    clipped = OrderedDict((key, max(0.02, min(0.25, float(value)))) for key, value in weights.items())
    total = sum(clipped.values())
    if total <= 0:
        return CURRENT_BALANCE_WEIGHTS.copy()
    return OrderedDict((key, round(value / total, 6)) for key, value in clipped.items())


def generate_candidate_weights(research_df):
    scenarios = [
        ("current", 0.0, "現行研究用ウェイト"),
        ("conservative_candidate", MAX_CONSERVATIVE_WEIGHT_DELTA, "小幅な研究候補"),
        ("balanced_candidate", MAX_BALANCED_WEIGHT_DELTA, "成績と安定性を反映する候補"),
        ("experimental_candidate", MAX_EXPERIMENTAL_WEIGHT_DELTA, "実験用の大きめ候補"),
    ]
    rows = []
    summary_rows = []
    research_by_key = research_df.set_index("subscore_key") if research_df is not None and not research_df.empty else pd.DataFrame()
    for scenario, max_delta, description in scenarios:
        proposed = OrderedDict()
        reasons = {}
        for key, current in CURRENT_BALANCE_WEIGHTS.items():
            if scenario == "current" or key not in research_by_key.index:
                delta = 0.0
                reason = "現行維持"
            else:
                row = research_by_key.loc[key]
                delta = _scenario_delta(row, max_delta)
                reason = str(row.get("rationale", "研究評価に基づく候補"))
            proposed[key] = current + delta
            reasons[key] = reason
        normalized = _normalize_weights(proposed)
        max_abs_delta = 0.0
        changed = 0
        for key, current in CURRENT_BALANCE_WEIGHTS.items():
            candidate = normalized[key]
            delta = candidate - current
            max_abs_delta = max(max_abs_delta, abs(delta))
            if abs(delta) > 0.0005:
                changed += 1
            rows.append(
                {
                    "scenario": scenario,
                    "subscore_key": key,
                    "display_name": SUBSCORE_LABELS[key],
                    "current_weight": round(current, 6),
                    "candidate_weight": round(candidate, 6),
                    "delta": round(delta, 6),
                    "reason": reasons[key],
                    "weights_version": BALANCE_WEIGHT_RESEARCH_VERSION,
                }
            )
        summary_rows.append(
            {
                "scenario": scenario,
                "description": description,
                "changed_items": changed,
                "max_abs_delta": round(max_abs_delta, 6),
                "total_weight": round(sum(normalized.values()), 6),
                "note": "本番重みには自動反映しません",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def _weighted_score(row, weights):
    weighted = 0.0
    available = 0.0
    for key, weight in weights.items():
        score = _safe_float(row.get(key))
        if score is None:
            continue
        weighted += score * weight
        available += weight
    if available <= 0:
        return None
    return weighted / available


def simulate_candidate_weights(reports, candidate_weights_df):
    prepared = _prepare_research_rows(reports)
    columns = [
        "scenario",
        "evaluation_count",
        "average_calculated_score",
        "high_score_avg_match",
        "low_score_avg_match",
        "high_low_match_gap",
        "hit3_rate",
        "top_candidate_avg_match",
        "top_candidate_count",
        "note",
    ]
    if prepared.empty or candidate_weights_df is None or candidate_weights_df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for scenario, weights_group in candidate_weights_df.groupby("scenario", sort=False):
        weights = OrderedDict((row["subscore_key"], _safe_float(row["candidate_weight"], 0.0)) for _, row in weights_group.iterrows())
        scored = prepared.copy()
        scored["calculated_score"] = scored.apply(lambda row: _weighted_score(row, weights), axis=1)
        scored = scored[scored["calculated_score"].notna()].copy()
        if scored.empty:
            rows.append(
                {
                    "scenario": scenario,
                    "evaluation_count": 0,
                    "average_calculated_score": None,
                    "high_score_avg_match": None,
                    "low_score_avg_match": None,
                    "high_low_match_gap": None,
                    "hit3_rate": "0.0%",
                    "top_candidate_avg_match": None,
                    "top_candidate_count": 0,
                    "note": "サブスコア詳細不足",
                }
            )
            continue
        high = scored[scored["calculated_score"] >= HIGH_SCORE_THRESHOLD]
        low = scored[scored["calculated_score"] < HIGH_SCORE_THRESHOLD]
        high_avg = high["main_match"].mean() if not high.empty else None
        low_avg = low["main_match"].mean() if not low.empty else None
        top_rows = []
        if "draw_no" in scored:
            for _, group in scored.groupby("draw_no"):
                if not group.empty:
                    top_rows.append(group.sort_values(["calculated_score", "candidate_no"], ascending=[False, True]).iloc[0])
        top_df = pd.DataFrame(top_rows)
        rows.append(
            {
                "scenario": scenario,
                "evaluation_count": int(len(scored)),
                "average_calculated_score": _round(scored["calculated_score"].mean()),
                "high_score_avg_match": _round(high_avg),
                "low_score_avg_match": _round(low_avg),
                "high_low_match_gap": _round((high_avg - low_avg) if high_avg is not None and low_avg is not None else None),
                "hit3_rate": _percent((scored["main_match"] >= 3).mean()),
                "top_candidate_avg_match": _round(top_df["main_match"].mean()) if not top_df.empty else None,
                "top_candidate_count": int(len(top_df)),
                "note": "過去保存済みサブスコアによる研究シミュレーション",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_balance_failure_research(research_df, simulation_df):
    rows = []
    if research_df is None or research_df.empty:
        return pd.DataFrame(
            [
                {
                    "failure_type": "insufficient_samples",
                    "affected_subscores": "-",
                    "severity": "reference",
                    "evidence": "研究評価に使えるサブスコアが不足しています",
                    "recommended_action": "本番重みは維持し、詳細JSONが保存された検証を蓄積する",
                }
            ]
        )
    missing = research_df[pd.to_numeric(research_df["evaluation_count"], errors="coerce").fillna(0) < MIN_WEIGHT_RESEARCH_SAMPLES]
    if not missing.empty:
        rows.append(
            {
                "failure_type": "insufficient_samples",
                "affected_subscores": ", ".join(missing["display_name"].head(5)),
                "severity": "reference",
                "evidence": f"{len(missing)}項目で研究最低件数に未達",
                "recommended_action": "候補重みは参考値に留め、検証件数を増やす",
            }
        )
    low_gap = research_df[pd.to_numeric(research_df["performance_gap_score"], errors="coerce").fillna(50) < 45]
    if not low_gap.empty:
        rows.append(
            {
                "failure_type": "high_score_low_hit",
                "affected_subscores": ", ".join(low_gap["display_name"].head(5)),
                "severity": "medium",
                "evidence": "高スコア群が低スコア群を十分に上回っていません",
                "recommended_action": "該当項目は保守的に減点または維持で比較する",
            }
        )
    unstable = research_df[pd.to_numeric(research_df["stability_score"], errors="coerce").fillna(100) < 45]
    if not unstable.empty:
        rows.append(
            {
                "failure_type": "long_term_mismatch",
                "affected_subscores": ", ".join(unstable["display_name"].head(5)),
                "severity": "medium",
                "evidence": "直近と長期の成績差が大きい項目があります",
                "recommended_action": "直近偏重と長期安定の候補を分けて比較する",
            }
        )
    high_missing = research_df[research_df["not_evaluated_rate"].astype(str).str.replace("%", "", regex=False).map(lambda value: _safe_float(value, 0.0) >= 50)]
    if not high_missing.empty:
        rows.append(
            {
                "failure_type": "high_not_evaluated_rate",
                "affected_subscores": ", ".join(high_missing["display_name"].head(5)),
                "severity": "reference",
                "evidence": "評価不能率が高い項目があります",
                "recommended_action": "詳細JSON不足の間は本番反映せず、表示上の研究候補に留める",
            }
        )
    if simulation_df is not None and not simulation_df.empty:
        current = simulation_df[simulation_df["scenario"] == "current"]
        if not current.empty:
            row = current.iloc[0]
            high = _safe_float(row.get("high_score_avg_match"))
            low = _safe_float(row.get("low_score_avg_match"))
            if high is not None and low is not None and high <= low:
                rows.append(
                    {
                        "failure_type": "score_concentration",
                        "affected_subscores": "overall",
                        "severity": "medium",
                        "evidence": "現行計算スコアの高スコア群が低スコア群を上回っていません",
                        "recommended_action": "上位候補の差別化とサブスコア重みの分散を検証する",
                    }
                )
    if not rows:
        rows.append(
            {
                "failure_type": "no_major_failure",
                "affected_subscores": "-",
                "severity": "low",
                "evidence": "大きな失敗分類は検出されていません",
                "recommended_action": "本番重みは維持し、候補重みは研究画面で比較を続ける",
            }
        )
    return pd.DataFrame(rows)


def _report_columns(df):
    """Column detector override that supports both current Japanese CSVs and legacy mojibake labels."""
    key_col = _find_column(df, ["検証キー", "verification_key"], fallback_index=0, contains=["検証", "verification"])
    key_first = bool(df is not None and len(df.columns) and key_col == df.columns[0])
    return {
        "key": key_col,
        "id": _find_column(df, ["予想ID", "prediction_id"], fallback_index=1 if key_first else 0, contains=["予想id", "prediction_id"]),
        "draw": _find_column(df, ["開催回", "draw_no", "draw"], fallback_index=2 if key_first else 1, contains=["開催", "draw"]),
        "candidate": _find_column(df, ["候補番号", "candidate_no", "candidate"], fallback_index=5 if key_first else 3, contains=["候補", "candidate"]),
        "model": _find_column(df, ["使用モデル", "model_name", "model"], fallback_index=6 if key_first else 3, contains=["使用モデル", "model"]),
        "main_match": _find_column(df, ["本数字一致数", "main_match"], fallback_index=11 if key_first else 7, contains=["本数字一致", "main_match"]),
        "bonus_match": _find_column(df, ["ボーナス一致数", "bonus_match"], fallback_index=14 if key_first else 9, contains=["ボーナス一致", "bonus_match"]),
        "total_match": _find_column(df, ["総一致数", "total_match"], contains=["総一致", "total_match"]),
        "balance_score": _find_column(df, ["balance_score"]),
        "balance_grade": _find_column(df, ["balance_grade"]),
        "details": _find_column(df, ["balance_details_json"]),
    }


def _scenario_weights(candidate_weights_df, scenario):
    if candidate_weights_df is None or candidate_weights_df.empty:
        return CURRENT_BALANCE_WEIGHTS.copy()
    group = candidate_weights_df[candidate_weights_df["scenario"] == scenario]
    if group.empty:
        return CURRENT_BALANCE_WEIGHTS.copy()
    weights = OrderedDict()
    for _, row in group.iterrows():
        key = str(row.get("subscore_key", "")).strip()
        if key in CURRENT_BALANCE_WEIGHTS:
            weights[key] = _safe_float(row.get("candidate_weight"), CURRENT_BALANCE_WEIGHTS[key])
    for key, value in CURRENT_BALANCE_WEIGHTS.items():
        weights.setdefault(key, value)
    return _normalize_weights(weights)


def candidate_weights_hash(weights):
    if isinstance(weights, pd.DataFrame):
        if "subscore_key" in weights and "candidate_weight" in weights:
            data = OrderedDict(
                (str(row["subscore_key"]), _round(row["candidate_weight"], 6))
                for _, row in weights.sort_values("subscore_key").iterrows()
            )
        else:
            data = weights.to_dict(orient="records")
    else:
        data = OrderedDict((str(key), _round(value, 6)) for key, value in dict(weights).items())
    return hashlib.sha256(_stable_json(data).encode("utf-8")).hexdigest()


def _candidate_weights_json(weights):
    return _stable_json(OrderedDict((key, _round(value, 6)) for key, value in dict(weights).items()))


def _ranking_columns():
    return [
        "draw_no",
        "candidate_type",
        "prediction_id",
        "candidate_no",
        "model_name",
        "calculated_score",
        "rank",
        "tie_count",
        "is_tied_top",
        "candidate_count",
        "main_match",
        "bonus_match",
        "total_match",
        "balance_score",
    ]


def _ranking_summary_columns():
    return [
        "draw_no",
        "candidate_type",
        "candidate_count",
        "top_candidate_no",
        "top_candidate_prediction_id",
        "top_tie_count",
        "is_tied_top",
        "top_candidate_main_match_count",
        "top_candidate_average_match",
        "top3_candidate_average_match",
        "best_actual_hit_candidate_rank",
        "best_actual_candidate_no",
        "best_actual_match_count",
        "best_actual_is_rank1",
        "best_actual_within_top3",
    ]


def _prepare_rankable_rows(reports):
    prepared = _prepare_research_rows(reports)
    if prepared.empty:
        return prepared
    prepared = prepared.copy()
    prepared["draw_no"] = pd.to_numeric(prepared["draw_no"], errors="coerce")
    prepared = prepared[prepared["draw_no"].notna() & (prepared["draw_no"] > 0)].copy()
    if prepared.empty:
        return prepared
    for column in ("candidate_no", "main_match", "bonus_match", "total_match", "balance_score"):
        if column in prepared:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    dedupe_columns = [column for column in ("draw_no", "prediction_id", "candidate_no", "model_name") if column in prepared]
    if dedupe_columns:
        prepared = prepared.drop_duplicates(subset=dedupe_columns, keep="first")
    return prepared


def build_candidate_ranking_rows(reports, candidate_weights_df):
    prepared = _prepare_rankable_rows(reports)
    if prepared.empty or candidate_weights_df is None or candidate_weights_df.empty:
        return pd.DataFrame(columns=_ranking_columns())

    rows = []
    scenarios = list(dict.fromkeys(candidate_weights_df["scenario"].astype(str)))
    for scenario in scenarios:
        weights = _scenario_weights(candidate_weights_df, scenario)
        scored = prepared.copy()
        scored["calculated_score"] = scored.apply(lambda row: _weighted_score(row, weights), axis=1)
        scored = scored[scored["calculated_score"].notna()].copy()
        if scored.empty:
            continue
        scored["candidate_count"] = scored.groupby("draw_no")["candidate_no"].transform("count")
        scored = scored[scored["candidate_count"] > 1].copy()
        if scored.empty:
            continue
        scored["score_key"] = pd.to_numeric(scored["calculated_score"], errors="coerce").round(6)
        scored["rank"] = scored.groupby("draw_no")["score_key"].rank(method="dense", ascending=False).astype(int)
        scored["tie_count"] = scored.groupby(["draw_no", "score_key"])["candidate_no"].transform("count").astype(int)
        scored["is_tied_top"] = (scored["rank"] == 1) & (scored["tie_count"] > 1)
        for _, row in scored.iterrows():
            rows.append(
                {
                    "draw_no": _safe_int(row.get("draw_no")),
                    "candidate_type": scenario,
                    "prediction_id": str(row.get("prediction_id", "") or ""),
                    "candidate_no": _safe_int(row.get("candidate_no")),
                    "model_name": str(row.get("model_name", "") or ""),
                    "calculated_score": _round(row.get("calculated_score")),
                    "rank": _safe_int(row.get("rank")),
                    "tie_count": _safe_int(row.get("tie_count")),
                    "is_tied_top": bool(row.get("is_tied_top")),
                    "candidate_count": _safe_int(row.get("candidate_count")),
                    "main_match": _round(row.get("main_match")),
                    "bonus_match": _round(row.get("bonus_match")),
                    "total_match": _round(row.get("total_match")),
                    "balance_score": _round(row.get("balance_score")),
                }
            )
    return pd.DataFrame(rows, columns=_ranking_columns())


def build_per_draw_candidate_ranking(reports, candidate_weights_df):
    return build_candidate_ranking_rows(reports, candidate_weights_df)


def _candidate_display(series):
    values = []
    for value in series:
        number = _safe_int(value, None)
        values.append(str(number) if number is not None else str(value))
    return "|".join(values)


def _prediction_display(series):
    values = [str(value) for value in series if str(value).strip()]
    return "|".join(values)


def build_per_draw_ranking_summary(ranking_df):
    if ranking_df is None or ranking_df.empty:
        return pd.DataFrame(columns=_ranking_summary_columns())
    rows = []
    for (draw_no, scenario), group in ranking_df.groupby(["draw_no", "candidate_type"], sort=True):
        group = group.copy()
        top = group[group["rank"] == 1].copy()
        top3 = group[group["rank"] <= 3].copy()
        best_match = pd.to_numeric(group["main_match"], errors="coerce").max()
        best_rows = group[pd.to_numeric(group["main_match"], errors="coerce") == best_match].copy()
        best_rank = int(pd.to_numeric(best_rows["rank"], errors="coerce").min()) if not best_rows.empty else None
        rows.append(
            {
                "draw_no": _safe_int(draw_no),
                "candidate_type": scenario,
                "candidate_count": int(pd.to_numeric(group["candidate_no"], errors="coerce").count()),
                "top_candidate_no": _candidate_display(top["candidate_no"]) if not top.empty else "",
                "top_candidate_prediction_id": _prediction_display(top["prediction_id"]) if not top.empty else "",
                "top_tie_count": int(len(top)),
                "is_tied_top": bool(len(top) > 1 or top.get("is_tied_top", pd.Series(dtype=bool)).any()),
                "top_candidate_main_match_count": _round(pd.to_numeric(top["main_match"], errors="coerce").mean()) if not top.empty else None,
                "top_candidate_average_match": _round(pd.to_numeric(top["main_match"], errors="coerce").mean()) if not top.empty else None,
                "top3_candidate_average_match": _round(pd.to_numeric(top3["main_match"], errors="coerce").mean()) if not top3.empty else None,
                "best_actual_hit_candidate_rank": best_rank,
                "best_actual_candidate_no": _candidate_display(best_rows["candidate_no"]) if not best_rows.empty else "",
                "best_actual_match_count": _round(best_match),
                "best_actual_is_rank1": bool(best_rank == 1) if best_rank is not None else False,
                "best_actual_within_top3": bool(best_rank is not None and best_rank <= 3),
            }
        )
    return pd.DataFrame(rows, columns=_ranking_summary_columns())


def _split_candidate_set(value):
    text = str(value or "").strip()
    if not text:
        return set()
    return {token for token in text.replace(",", "|").split("|") if token.strip()}


def build_ranking_stability_summary(ranking_df, summary_df=None):
    columns = [
        "candidate_type",
        "evaluation_count",
        "ranking_stability_score",
        "rank_change_mean",
        "rank_change_max",
        "top_candidate_switch_rate",
        "top_candidate_fixed_rate",
        "tie_rate",
    ]
    if ranking_df is None or ranking_df.empty:
        return pd.DataFrame(columns=columns)
    if summary_df is None or summary_df.empty:
        summary_df = build_per_draw_ranking_summary(ranking_df)
    current = ranking_df[ranking_df["candidate_type"] == "current"]
    current_ranks = {}
    for _, row in current.iterrows():
        current_ranks[(row["draw_no"], row["candidate_no"])] = _safe_float(row["rank"], 0.0)
    current_top = {}
    if summary_df is not None and not summary_df.empty:
        for _, row in summary_df[summary_df["candidate_type"] == "current"].iterrows():
            current_top[row["draw_no"]] = _split_candidate_set(row.get("top_candidate_no"))

    rows = []
    for scenario, group in ranking_df.groupby("candidate_type", sort=False):
        changes = []
        for _, row in group.iterrows():
            base_rank = current_ranks.get((row["draw_no"], row["candidate_no"]))
            if base_rank is not None:
                changes.append(abs(_safe_float(row["rank"], 0.0) - base_rank))
        scenario_summary = summary_df[summary_df["candidate_type"] == scenario] if summary_df is not None and not summary_df.empty else pd.DataFrame()
        switch_values = []
        tie_values = []
        for _, row in scenario_summary.iterrows():
            draw_no = row.get("draw_no")
            if draw_no in current_top:
                switch_values.append(_split_candidate_set(row.get("top_candidate_no")) != current_top[draw_no])
            tie_values.append(bool(row.get("is_tied_top")))
        switch_rate = sum(switch_values) / len(switch_values) if switch_values else 0.0
        tie_rate = sum(tie_values) / len(tie_values) if tie_values else 0.0
        rank_change_mean = sum(changes) / len(changes) if changes else 0.0
        rank_change_max = max(changes) if changes else 0.0
        stability_score = _clamp(100 - rank_change_mean * 12 - switch_rate * 35 - tie_rate * 10)
        rows.append(
            {
                "candidate_type": scenario,
                "evaluation_count": int(len(scenario_summary)),
                "ranking_stability_score": _round(stability_score),
                "rank_change_mean": _round(rank_change_mean),
                "rank_change_max": _round(rank_change_max),
                "top_candidate_switch_rate": _percent(switch_rate),
                "top_candidate_fixed_rate": _percent(1 - switch_rate),
                "tie_rate": _percent(tie_rate),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _aggregate_ranking_summary(summary_df):
    columns = [
        "evaluation_count",
        "top1_mean_match",
        "top3_mean_match",
        "best_candidate_top1_rate",
        "best_candidate_top3_rate",
        "tie_rate",
    ]
    if summary_df is None or summary_df.empty:
        return {column: 0 if column == "evaluation_count" else None for column in columns}
    return {
        "evaluation_count": int(len(summary_df)),
        "top1_mean_match": _round(pd.to_numeric(summary_df["top_candidate_average_match"], errors="coerce").mean()),
        "top3_mean_match": _round(pd.to_numeric(summary_df["top3_candidate_average_match"], errors="coerce").mean()),
        "best_candidate_top1_rate": _round(summary_df["best_actual_is_rank1"].astype(bool).mean()),
        "best_candidate_top3_rate": _round(summary_df["best_actual_within_top3"].astype(bool).mean()),
        "tie_rate": _round(summary_df["is_tied_top"].astype(bool).mean()),
    }


def build_rolling_weight_evaluation(reports, windows=(5, 10, 20)):
    columns = [
        "window_size",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "candidate_type",
        "evaluation_count",
        "top1_mean_match",
        "top3_mean_match",
        "best_candidate_top1_rate",
        "best_candidate_top3_rate",
        "ranking_stability_score",
        "ranking_stability_variance",
        "top1_mean_match_delta",
        "top3_mean_match_delta",
        "best_candidate_top1_rate_delta",
        "best_candidate_top3_rate_delta",
        "ranking_stability_delta",
        "note",
    ]
    prepared = _prepare_rankable_rows(reports)
    if prepared.empty:
        return pd.DataFrame(columns=columns)
    draws = sorted(int(value) for value in pd.to_numeric(prepared["draw_no"], errors="coerce").dropna().unique())
    if len(draws) < 3:
        return pd.DataFrame(columns=columns)

    all_rows = []
    for requested_window in windows:
        window = min(int(requested_window), max(1, len(draws) - 1))
        if window < 2:
            continue
        test_summaries = []
        train_ranges = []
        for test_index in range(window, len(draws)):
            train_draws = draws[test_index - window : test_index]
            test_draw = draws[test_index]
            train_df = prepared[prepared["draw_no"].isin(train_draws)].copy()
            test_df = prepared[prepared["draw_no"] == test_draw].copy()
            if train_df.empty or test_df.empty:
                continue
            research = evaluate_subscore_research(train_df)
            candidate_weights, _ = generate_candidate_weights(research)
            ranking = build_per_draw_candidate_ranking(test_df, candidate_weights)
            if ranking.empty:
                continue
            summary = build_per_draw_ranking_summary(ranking)
            if summary.empty:
                continue
            summary["train_start"] = min(train_draws)
            summary["train_end"] = max(train_draws)
            summary["test_draw"] = test_draw
            test_summaries.append(summary)
            train_ranges.append((min(train_draws), max(train_draws), test_draw))
        if not test_summaries:
            continue
        merged = pd.concat(test_summaries, ignore_index=True)
        stability = build_ranking_stability_summary(
            pd.concat(
                [
                    build_per_draw_candidate_ranking(
                        prepared[prepared["draw_no"] == row["test_draw"]],
                        generate_candidate_weights(evaluate_subscore_research(prepared[prepared["draw_no"].isin(range(row["train_start"], row["train_end"] + 1))]))[0],
                    )
                    for _, row in merged[["train_start", "train_end", "test_draw"]].drop_duplicates().iterrows()
                ],
                ignore_index=True,
            )
        )
        stability_map = {}
        if not stability.empty:
            stability_map = dict(zip(stability["candidate_type"], pd.to_numeric(stability["ranking_stability_score"], errors="coerce")))
        current_metrics = _aggregate_ranking_summary(merged[merged["candidate_type"] == "current"])
        current_stability = _safe_float(stability_map.get("current"), 100.0)
        for scenario, group in merged.groupby("candidate_type", sort=False):
            metrics = _aggregate_ranking_summary(group)
            stability_score = _safe_float(stability_map.get(scenario), current_stability)
            all_rows.append(
                {
                    "window_size": int(window),
                    "train_start": min(item[0] for item in train_ranges),
                    "train_end": max(item[1] for item in train_ranges),
                    "test_start": min(item[2] for item in train_ranges),
                    "test_end": max(item[2] for item in train_ranges),
                    "candidate_type": scenario,
                    "evaluation_count": metrics["evaluation_count"],
                    "top1_mean_match": metrics["top1_mean_match"],
                    "top3_mean_match": metrics["top3_mean_match"],
                    "best_candidate_top1_rate": metrics["best_candidate_top1_rate"],
                    "best_candidate_top3_rate": metrics["best_candidate_top3_rate"],
                    "ranking_stability_score": _round(stability_score),
                    "ranking_stability_variance": None,
                    "top1_mean_match_delta": _round((_safe_float(metrics["top1_mean_match"], 0.0) - _safe_float(current_metrics["top1_mean_match"], 0.0))),
                    "top3_mean_match_delta": _round((_safe_float(metrics["top3_mean_match"], 0.0) - _safe_float(current_metrics["top3_mean_match"], 0.0))),
                    "best_candidate_top1_rate_delta": _round((_safe_float(metrics["best_candidate_top1_rate"], 0.0) - _safe_float(current_metrics["best_candidate_top1_rate"], 0.0))),
                    "best_candidate_top3_rate_delta": _round((_safe_float(metrics["best_candidate_top3_rate"], 0.0) - _safe_float(current_metrics["best_candidate_top3_rate"], 0.0))),
                    "ranking_stability_delta": _round(stability_score - current_stability),
                    "note": "research_only_no_production_apply",
                }
            )
    result = pd.DataFrame(all_rows, columns=columns)
    if not result.empty:
        variance = result.groupby("candidate_type")["ranking_stability_score"].transform(lambda values: _round(pd.to_numeric(values, errors="coerce").var(ddof=0), 6))
        result["ranking_stability_variance"] = variance
    return result


def build_production_readiness_checklist(weight_research):
    research = weight_research.get("subscore_research", pd.DataFrame()) if isinstance(weight_research, dict) else pd.DataFrame()
    rolling = weight_research.get("rolling_evaluation", pd.DataFrame()) if isinstance(weight_research, dict) else pd.DataFrame()
    ranking_summary = weight_research.get("per_draw_ranking_summary", pd.DataFrame()) if isinstance(weight_research, dict) else pd.DataFrame()
    sample_count = int(pd.to_numeric(research.get("evaluation_count", pd.Series(dtype=float)), errors="coerce").max()) if research is not None and not research.empty and "evaluation_count" in research else 0
    rolling_count = int(pd.to_numeric(rolling.get("evaluation_count", pd.Series(dtype=float)), errors="coerce").max()) if rolling is not None and not rolling.empty and "evaluation_count" in rolling else 0
    tie_rate = _safe_float(ranking_summary["is_tied_top"].astype(bool).mean(), 0.0) if ranking_summary is not None and not ranking_summary.empty and "is_tied_top" in ranking_summary else 0.0
    rows = [
        {
            "check_item": "sample_count",
            "status": "pass" if sample_count >= MIN_WEIGHT_RECOMMENDATION_SAMPLES else "hold",
            "value": sample_count,
            "required": MIN_WEIGHT_RECOMMENDATION_SAMPLES,
            "note": "Manual review only. Production weights are not changed by Ver1.10.",
        },
        {
            "check_item": "rolling_evaluation_count",
            "status": "pass" if rolling_count > 0 else "hold",
            "value": rolling_count,
            "required": 1,
            "note": "Rolling checks must use past draws only.",
        },
        {
            "check_item": "tie_rate",
            "status": "review" if tie_rate > 0 else "pass",
            "value": _percent(tie_rate),
            "required": "review tied top ranks",
            "note": "Tied top ranks are not treated as a single clear winner.",
        },
        {
            "check_item": "manual_decision",
            "status": "required",
            "value": "unreviewed",
            "required": "candidate_for_adoption / hold / rejected",
            "note": "A manual decision can be saved to review history without changing predictions.",
        },
        {
            "check_item": "production_apply",
            "status": "blocked",
            "value": "not_available",
            "required": "separate future version",
            "note": "Ver1.10 intentionally has no production apply button or auto-apply path.",
        },
    ]
    return pd.DataFrame(rows)


def build_review_candidates(weight_research, game=""):
    columns = [
        "review_id",
        "game",
        "candidate_type",
        "weights_version",
        "candidate_weights_hash",
        "created_at",
        "sample_count",
        "rolling_evaluation_count",
        "top1_mean_match",
        "top3_mean_match",
        "best_candidate_top1_rate",
        "best_candidate_top3_rate",
        "current_comparison_json",
        "decision",
    ]
    if not isinstance(weight_research, dict):
        return pd.DataFrame(columns=columns)
    candidate_weights = weight_research.get("candidate_weights", pd.DataFrame())
    if candidate_weights is None or candidate_weights.empty:
        return pd.DataFrame(columns=columns)
    rolling = weight_research.get("rolling_evaluation", pd.DataFrame())
    simulation = weight_research.get("simulation", pd.DataFrame())
    research = weight_research.get("subscore_research", pd.DataFrame())
    sample_count = int(pd.to_numeric(research.get("evaluation_count", pd.Series(dtype=float)), errors="coerce").max()) if research is not None and not research.empty and "evaluation_count" in research else 0
    rows = []
    for scenario in dict.fromkeys(candidate_weights["scenario"].astype(str)):
        if scenario == "current":
            continue
        weights = _scenario_weights(candidate_weights, scenario)
        weight_hash = candidate_weights_hash(weights)
        rolling_row = pd.DataFrame()
        if rolling is not None and not rolling.empty:
            rolling_row = rolling[rolling["candidate_type"] == scenario].tail(1)
        sim_row = pd.DataFrame()
        if simulation is not None and not simulation.empty:
            sim_row = simulation[simulation["scenario"] == scenario].tail(1)
        top1 = _safe_float(rolling_row.iloc[0].get("top1_mean_match")) if not rolling_row.empty else (_safe_float(sim_row.iloc[0].get("top_candidate_avg_match")) if not sim_row.empty else None)
        top3 = _safe_float(rolling_row.iloc[0].get("top3_mean_match")) if not rolling_row.empty else None
        comparison = {}
        if not rolling_row.empty:
            for key in ("top1_mean_match_delta", "top3_mean_match_delta", "best_candidate_top1_rate_delta", "best_candidate_top3_rate_delta", "ranking_stability_delta"):
                comparison[key] = _safe_float(rolling_row.iloc[0].get(key))
        rows.append(
            {
                "review_id": f"balance-review-{game or 'unknown'}-{scenario}-{weight_hash[:12]}",
                "game": game,
                "candidate_type": scenario,
                "weights_version": BALANCE_WEIGHT_RESEARCH_VERSION,
                "candidate_weights_hash": weight_hash,
                "created_at": _now_text(),
                "sample_count": sample_count,
                "rolling_evaluation_count": _safe_int(rolling_row.iloc[0].get("evaluation_count")) if not rolling_row.empty else 0,
                "top1_mean_match": _round(top1),
                "top3_mean_match": _round(top3),
                "best_candidate_top1_rate": _round(rolling_row.iloc[0].get("best_candidate_top1_rate")) if not rolling_row.empty else None,
                "best_candidate_top3_rate": _round(rolling_row.iloc[0].get("best_candidate_top3_rate")) if not rolling_row.empty else None,
                "current_comparison_json": _stable_json(comparison),
                "decision": "unreviewed",
            }
        )
    return pd.DataFrame(rows, columns=columns)


REVIEW_HISTORY_COLUMNS = [
    "review_id",
    "reviewed_at",
    "game",
    "candidate_type",
    "weights_version",
    "candidate_weights_json",
    "candidate_weights_hash",
    "decision",
    "review_comment",
    "sample_count",
    "rolling_evaluation_count",
    "top1_mean_match",
    "top3_mean_match",
    "best_candidate_top1_rate",
    "best_candidate_top3_rate",
    "current_comparison_json",
    "reviewer",
]


def build_manual_review_row(weight_research, game, candidate_type, decision, review_comment="", reviewer=""):
    if decision not in {"candidate_for_adoption", "hold", "rejected", "unreviewed"}:
        decision = "unreviewed"
    candidates = build_review_candidates(weight_research, game=game)
    matched = candidates[candidates["candidate_type"] == candidate_type] if not candidates.empty else pd.DataFrame()
    if matched.empty:
        return None
    row = matched.iloc[0].to_dict()
    weights = _scenario_weights(weight_research.get("candidate_weights", pd.DataFrame()), candidate_type)
    return {
        "review_id": row.get("review_id", ""),
        "reviewed_at": _now_text(),
        "game": game,
        "candidate_type": candidate_type,
        "weights_version": BALANCE_WEIGHT_RESEARCH_VERSION,
        "candidate_weights_json": _candidate_weights_json(weights),
        "candidate_weights_hash": row.get("candidate_weights_hash", candidate_weights_hash(weights)),
        "decision": decision,
        "review_comment": str(review_comment or ""),
        "sample_count": row.get("sample_count", 0),
        "rolling_evaluation_count": row.get("rolling_evaluation_count", 0),
        "top1_mean_match": row.get("top1_mean_match"),
        "top3_mean_match": row.get("top3_mean_match"),
        "best_candidate_top1_rate": row.get("best_candidate_top1_rate"),
        "best_candidate_top3_rate": row.get("best_candidate_top3_rate"),
        "current_comparison_json": row.get("current_comparison_json", "{}"),
        "reviewer": str(reviewer or ""),
    }


def read_review_history(path):
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=REVIEW_HISTORY_COLUMNS)
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            df = pd.read_csv(target, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(target)
    for column in REVIEW_HISTORY_COLUMNS:
        if column not in df:
            df[column] = ""
    return df[REVIEW_HISTORY_COLUMNS]


def append_review_history(path, review_row):
    if not review_row:
        return False, "review row is empty"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    history = read_review_history(target)
    row_df = pd.DataFrame([{column: review_row.get(column, "") for column in REVIEW_HISTORY_COLUMNS}])
    updated = pd.concat([history, row_df], ignore_index=True)
    updated.to_csv(target, index=False, encoding="utf-8-sig")
    return True, f"saved review history: {review_row.get('review_id', '')}"


def build_review_export_package(weight_research, game, candidate_type):
    if not isinstance(weight_research, dict):
        weight_research = {}
    candidate_weights_df = weight_research.get("candidate_weights", pd.DataFrame())
    current_weights = _scenario_weights(candidate_weights_df, "current")
    candidate_weights = _scenario_weights(candidate_weights_df, candidate_type)
    differences = OrderedDict((key, _round(candidate_weights.get(key, 0.0) - current_weights.get(key, 0.0), 6)) for key in CURRENT_BALANCE_WEIGHTS)
    review_candidates = build_review_candidates(weight_research, game=game)
    review_row = review_candidates[review_candidates["candidate_type"] == candidate_type]
    metadata = review_row.iloc[0].to_dict() if not review_row.empty else {}
    per_draw = weight_research.get("per_draw_ranking_summary", pd.DataFrame())
    per_draw = per_draw[per_draw["candidate_type"] == candidate_type] if per_draw is not None and not per_draw.empty and "candidate_type" in per_draw else pd.DataFrame()
    rolling = weight_research.get("rolling_evaluation", pd.DataFrame())
    rolling = rolling[rolling["candidate_type"] == candidate_type] if rolling is not None and not rolling.empty and "candidate_type" in rolling else pd.DataFrame()
    return {
        "game": game,
        "candidate_type": candidate_type,
        "generated_at": _now_text(),
        "weights_version": BALANCE_WEIGHT_RESEARCH_VERSION,
        "review_version": BALANCE_WEIGHT_REVIEW_VERSION,
        "candidate_weights_hash": metadata.get("candidate_weights_hash", candidate_weights_hash(candidate_weights)),
        "confidence": {
            "sample_count": metadata.get("sample_count", 0),
            "rolling_evaluation_count": metadata.get("rolling_evaluation_count", 0),
            "decision": metadata.get("decision", "unreviewed"),
        },
        "current_weights": dict(current_weights),
        "candidate_weights": dict(candidate_weights),
        "differences": dict(differences),
        "subscore_research": weight_research.get("subscore_research", pd.DataFrame()).to_dict(orient="records") if isinstance(weight_research.get("subscore_research"), pd.DataFrame) else [],
        "per_draw_ranking_evaluation": per_draw.to_dict(orient="records"),
        "rolling_evaluation": rolling.to_dict(orient="records"),
        "current_comparison": json.loads(metadata.get("current_comparison_json", "{}")) if str(metadata.get("current_comparison_json", "")).strip() else {},
        "ai_improvement_failure_analysis": weight_research.get("failure_research", pd.DataFrame()).to_dict(orient="records") if isinstance(weight_research.get("failure_research"), pd.DataFrame) else [],
        "production_note": "research_export_only_no_production_apply",
    }


def review_export_json_bytes(package):
    return json.dumps(package, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def review_export_csv_bytes(package):
    rows = []
    for key in ("game", "candidate_type", "generated_at", "weights_version", "review_version", "candidate_weights_hash", "production_note"):
        rows.append({"section": "metadata", "field": key, "value": package.get(key, "")})
    for section in ("confidence", "current_weights", "candidate_weights", "differences", "current_comparison"):
        values = package.get(section, {})
        if isinstance(values, dict):
            for key, value in values.items():
                rows.append({"section": section, "field": key, "value": value})
    for section in ("subscore_research", "per_draw_ranking_evaluation", "rolling_evaluation", "ai_improvement_failure_analysis"):
        values = package.get(section, [])
        for index, item in enumerate(values, start=1):
            rows.append({"section": section, "field": f"row_{index}", "value": _stable_json(item)})
    df = pd.DataFrame(rows, columns=["section", "field", "value"])
    return ("\ufeff" + df.to_csv(index=False)).encode("utf-8")


def build_balance_weight_research(reports, draw_size=6):
    current_weights = get_current_balance_weights()
    research = evaluate_subscore_research(reports)
    candidate_weights, candidate_summary = generate_candidate_weights(research)
    simulation = simulate_candidate_weights(reports, candidate_weights)
    failure_research = build_balance_failure_research(research, simulation)
    per_draw_ranking = build_per_draw_candidate_ranking(reports, candidate_weights)
    per_draw_ranking_summary = build_per_draw_ranking_summary(per_draw_ranking)
    ranking_stability = build_ranking_stability_summary(per_draw_ranking, per_draw_ranking_summary)
    rolling_evaluation = build_rolling_weight_evaluation(reports)
    result = {
        "current_weights": current_weights,
        "subscore_research": research,
        "candidate_weights": candidate_weights,
        "candidate_summary": candidate_summary,
        "simulation": simulation,
        "failure_research": failure_research,
        "per_draw_ranking": per_draw_ranking,
        "per_draw_ranking_summary": per_draw_ranking_summary,
        "ranking_stability": ranking_stability,
        "rolling_evaluation": rolling_evaluation,
        "metadata": pd.DataFrame(
            [
                {
                    "weights_version": BALANCE_WEIGHT_RESEARCH_VERSION,
                    "review_version": BALANCE_WEIGHT_REVIEW_VERSION,
                    "draw_size": draw_size,
                    "min_research_samples": MIN_WEIGHT_RESEARCH_SAMPLES,
                    "min_recommendation_samples": MIN_WEIGHT_RECOMMENDATION_SAMPLES,
                    "production_change": "none",
                }
            ]
        ),
    }
    result["review_candidates"] = build_review_candidates(result)
    result["production_checklist"] = build_production_readiness_checklist(result)
    return {
        **result,
    }
