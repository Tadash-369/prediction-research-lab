from __future__ import annotations

from collections import OrderedDict
import json
import math

import pandas as pd


BALANCE_WEIGHT_RESEARCH_VERSION = "balance_weight_research_v1"
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
            prepared[key] = _subscore_from_details(details, key)
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


def build_balance_weight_research(reports, draw_size=6):
    current_weights = get_current_balance_weights()
    research = evaluate_subscore_research(reports)
    candidate_weights, candidate_summary = generate_candidate_weights(research)
    simulation = simulate_candidate_weights(reports, candidate_weights)
    failure_research = build_balance_failure_research(research, simulation)
    return {
        "current_weights": current_weights,
        "subscore_research": research,
        "candidate_weights": candidate_weights,
        "candidate_summary": candidate_summary,
        "simulation": simulation,
        "failure_research": failure_research,
        "metadata": pd.DataFrame(
            [
                {
                    "weights_version": BALANCE_WEIGHT_RESEARCH_VERSION,
                    "draw_size": draw_size,
                    "min_research_samples": MIN_WEIGHT_RESEARCH_SAMPLES,
                    "min_recommendation_samples": MIN_WEIGHT_RECOMMENDATION_SAMPLES,
                    "production_change": "none",
                }
            ]
        ),
    }
