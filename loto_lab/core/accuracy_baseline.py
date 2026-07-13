from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPORT_FILES = {
    "loto6_model_accuracy": "loto6_model_accuracy_baseline.csv",
    "loto7_model_accuracy": "loto7_model_accuracy_baseline.csv",
    "loto6_draw_metrics": "loto6_model_draw_metrics.csv",
    "loto7_draw_metrics": "loto7_model_draw_metrics.csv",
    "loto6_overlap": "loto6_model_overlap_matrix.csv",
    "loto7_overlap": "loto7_model_overlap_matrix.csv",
    "loto6_marginal": "loto6_marginal_contribution.csv",
    "loto7_marginal": "loto7_marginal_contribution.csv",
    "loto6_quality": "loto6_data_quality_issues.csv",
    "loto7_quality": "loto7_data_quality_issues.csv",
}

SUMMARY_FILE = "accuracy_baseline_summary.json"
MIN_RANKING_DRAWS = 3
REGENERATED_MARKERS = ("regenerated", "research regeneration", "研究用再生成", "再生成")
DEMO_MARKERS = ("test", "demo", "sample", "fixture", "テスト", "デモ", "サンプル")


@dataclass(frozen=True)
class GameConfig:
    game: str
    draw_size: int
    number_max: int
    bonus_size: int


GAME_CONFIGS = {
    "loto6": GameConfig("loto6", 6, 43, 1),
    "loto7": GameConfig("loto7", 7, 37, 2),
}


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    raise UnicodeDecodeError("csv", b"", 0, 1, f"encoding could not be detected: {path}")


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _int(value: Any) -> int | None:
    try:
        return int(float(_text(value).replace(",", "")))
    except (TypeError, ValueError):
        return None


def parse_number_set(value: Any) -> list[int]:
    text = _text(value)
    for separator in (",", " ", "/", "|", "・"):
        text = text.replace(separator, "-")
    values = []
    for part in text.split("-"):
        number = _int(part)
        if number is not None:
            values.append(number)
    return values


def _date(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(_text(value), errors="coerce")
    return None if pd.isna(parsed) else parsed


def _issue(game: str, prediction_id: str, draw_no: Any, code: str, detail: str, excluded: bool = True) -> dict[str, Any]:
    return {
        "game": game,
        "prediction_id": prediction_id,
        "draw_no": draw_no if draw_no is not None else "",
        "issue_code": code,
        "detail": detail,
        "excluded_from_official_ranking": bool(excluded),
    }


def classify_predictions(predictions: pd.DataFrame, results: pd.DataFrame, game: str, verification: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    config = GAME_CONFIGS[game]
    required = ["予想ID", "開催回", "予想日", "候補番号", "予想番号", "使用モデル", "保存日時"]
    result_required = ["開催回", "抽せん日", "本数字", "ボーナス数字"]
    classified_columns = list(dict.fromkeys([*(predictions.columns if predictions is not None else []), "分類", "除外理由"]))
    if predictions is None or predictions.empty:
        return {
            "classified": _empty(classified_columns),
            "eligible": _empty(classified_columns),
            "unverified": _empty(classified_columns),
            "excluded": _empty(classified_columns),
            "quality": _empty(["game", "prediction_id", "draw_no", "issue_code", "detail", "excluded_from_official_ranking"]),
        }

    missing = [column for column in required if column not in predictions.columns]
    if missing:
        quality = [_issue(game, "", "", "missing_prediction_columns", ", ".join(missing))]
        broken = predictions.copy()
        broken["分類"] = "データ不足または破損データ"
        broken["除外理由"] = "missing_prediction_columns"
        return {"classified": broken, "eligible": broken.iloc[0:0], "unverified": broken.iloc[0:0], "excluded": broken, "quality": pd.DataFrame(quality)}

    result_map: dict[int, pd.Series] = {}
    quality: list[dict[str, Any]] = []
    if results is not None and not results.empty:
        missing_results = [column for column in result_required if column not in results.columns]
        if missing_results:
            quality.append(_issue(game, "", "", "missing_result_columns", ", ".join(missing_results)))
        else:
            for _, row in results.iterrows():
                draw_no = _int(row.get("開催回"))
                if draw_no is None:
                    quality.append(_issue(game, "", "", "missing_draw_no", "result row has no valid draw number"))
                    continue
                if draw_no in result_map:
                    quality.append(_issue(game, "", draw_no, "duplicate_result_draw", "multiple official result rows"))
                result_map[draw_no] = row

    duplicate_ids = predictions["予想ID"].astype(str).duplicated(keep=False)
    duplicate_rows = predictions.duplicated(subset=["開催回", "予想番号", "使用モデル"], keep=False)
    verified_ids = set()
    if verification is not None and not verification.empty and "予想ID" in verification.columns:
        verified_ids = {_text(value) for value in verification["予想ID"] if _text(value)}
    canonical_names: dict[str, set[str]] = {}
    rows = []
    for position, (_, source) in enumerate(predictions.iterrows()):
        row = source.to_dict()
        prediction_id = _text(source.get("予想ID"))
        draw_no = _int(source.get("開催回"))
        model_name = _text(source.get("使用モデル"))
        numbers = parse_number_set(source.get("予想番号"))
        saved_at = _date(source.get("保存日時"))
        prediction_date = _date(source.get("予想日"))
        combined = " ".join(_text(source.get(column)).lower() for column in ("予想ID", "使用モデル", "予想理由"))
        reasons: list[str] = []
        classification = "正式保存予測"

        if not prediction_id:
            reasons.append("missing_prediction_id")
            quality.append(_issue(game, prediction_id, draw_no, "missing_prediction_id", "prediction id is empty"))
        if draw_no is None:
            reasons.append("missing_draw_no")
            quality.append(_issue(game, prediction_id, draw_no, "missing_draw_no", "draw number is invalid"))
        if not model_name:
            reasons.append("missing_model_name")
            quality.append(_issue(game, prediction_id, draw_no, "missing_model_name", "model name is empty"))
        if saved_at is None and prediction_date is None:
            reasons.append("missing_prediction_datetime")
            quality.append(_issue(game, prediction_id, draw_no, "missing_prediction_datetime", "saved and prediction dates are unavailable"))
        if len(numbers) != config.draw_size:
            reasons.append("invalid_number_count")
            quality.append(_issue(game, prediction_id, draw_no, "invalid_number_count", f"expected {config.draw_size}, got {len(numbers)}"))
        if len(set(numbers)) != len(numbers):
            reasons.append("duplicate_numbers")
            quality.append(_issue(game, prediction_id, draw_no, "duplicate_numbers", _text(source.get("予想番号"))))
        if any(number < 1 or number > config.number_max for number in numbers):
            reasons.append("number_out_of_range")
            quality.append(_issue(game, prediction_id, draw_no, "number_out_of_range", _text(source.get("予想番号"))))
        if bool(duplicate_ids.iloc[position]):
            reasons.append("duplicate_prediction_id")
            quality.append(_issue(game, prediction_id, draw_no, "duplicate_prediction_id", "prediction id is duplicated"))
        if bool(duplicate_rows.iloc[position]):
            reasons.append("duplicate_prediction")
            quality.append(_issue(game, prediction_id, draw_no, "duplicate_prediction", "same draw, model, and numbers are duplicated"))
        if (game == "loto6" and prediction_id.upper().startswith("L7-")) or (game == "loto7" and prediction_id and not prediction_id.upper().startswith("L7-")):
            reasons.append("mixed_game_prediction")
            quality.append(_issue(game, prediction_id, draw_no, "mixed_game_prediction", "prediction id prefix does not match game"))

        if any(marker in combined for marker in REGENERATED_MARKERS):
            classification = "現在のコードによる再生成予測"
            reasons.append("regenerated_prediction")
            quality.append(_issue(game, prediction_id, draw_no, "suspected_regenerated_prediction", "regeneration marker detected"))
        elif any(marker in combined for marker in DEMO_MARKERS):
            classification = "サンプル・テスト・デモ予測"
            reasons.append("test_or_demo_prediction")
            quality.append(_issue(game, prediction_id, draw_no, "test_or_demo_prediction", "test/demo marker detected"))

        result = result_map.get(draw_no) if draw_no is not None else None
        if result is None:
            if not reasons:
                classification = "未検証の正式保存予測"
            quality.append(_issue(game, prediction_id, draw_no, "result_not_registered", "official result is unavailable", False))
        else:
            draw_date = _date(result.get("抽せん日"))
            created = saved_at or prediction_date
            if draw_date is None:
                reasons.append("missing_draw_date")
                quality.append(_issue(game, prediction_id, draw_no, "missing_draw_date", "official draw date is unavailable"))
            elif created is None:
                reasons.append("official_status_indeterminate")
                quality.append(_issue(game, prediction_id, draw_no, "official_status_indeterminate", "creation time cannot be compared"))
            elif created.normalize() > draw_date.normalize():
                reasons.append("created_after_draw")
                quality.append(_issue(game, prediction_id, draw_no, "future_data_suspected", f"created {created} after draw date {draw_date.date()}"))
            elif created.normalize() == draw_date.normalize():
                reasons.append("same_day_timing_indeterminate")
                quality.append(_issue(game, prediction_id, draw_no, "official_status_indeterminate", "draw time is unavailable for same-day prediction"))

        if reasons and classification in ("正式保存予測", "未検証の正式保存予測"):
            classification = "データ不足または破損データ"
        elif classification == "正式保存予測" and prediction_id in verified_ids:
            classification = "正式検証済み予測"
        row["分類"] = classification
        row["除外理由"] = " / ".join(dict.fromkeys(reasons))
        row["_numbers"] = tuple(numbers)
        row["_draw_no"] = draw_no
        row["_model"] = model_name
        rows.append(row)

        normalized = "".join(model_name.lower().split())
        if normalized:
            canonical_names.setdefault(normalized, set()).add(model_name)

    for spellings in canonical_names.values():
        if len(spellings) > 1:
            quality.append(_issue(game, "", "", "model_name_variation", " / ".join(sorted(spellings)), False))

    classified = pd.DataFrame(rows)
    for (draw_no, model_name), group in classified.groupby(["_draw_no", "_model"], dropna=False):
        candidates = sorted(value for value in (_int(item) for item in group["候補番号"]) if value is not None and value > 0)
        if candidates and (len(candidates) < 3 or candidates != list(range(1, max(candidates) + 1))):
            quality.append(_issue(game, "", draw_no, "insufficient_ticket_count", f"{model_name}: candidates={candidates}", False))
    eligible = classified[classified["分類"].isin(["正式保存予測", "正式検証済み予測"]) & (classified["除外理由"] == "")].copy()
    unverified = classified[classified["分類"] == "未検証の正式保存予測"].copy()
    excluded = classified.drop(index=eligible.index.union(unverified.index)).copy()
    return {
        "classified": classified,
        "eligible": eligible,
        "unverified": unverified,
        "excluded": excluded,
        "quality": pd.DataFrame(quality),
    }


def _result_map(results: pd.DataFrame) -> dict[int, tuple[set[int], set[int], str]]:
    mapped = {}
    if results is None or results.empty:
        return mapped
    for _, row in results.iterrows():
        draw_no = _int(row.get("開催回"))
        if draw_no is not None:
            mapped[draw_no] = (set(parse_number_set(row.get("本数字"))), set(parse_number_set(row.get("ボーナス数字"))), _text(row.get("抽せん日")))
    return mapped


def build_ticket_metrics(eligible: pd.DataFrame, results: pd.DataFrame, game: str) -> pd.DataFrame:
    columns = ["game", "prediction_id", "draw_no", "draw_date", "model", "candidate_no", "numbers", "main_matches", "bonus_matches", "matches_with_bonus"]
    if eligible is None or eligible.empty:
        return _empty(columns)
    mapped = _result_map(results)
    rows = []
    for _, row in eligible.iterrows():
        draw_no = _int(row.get("開催回"))
        if draw_no not in mapped:
            continue
        actual, bonus, draw_date = mapped[draw_no]
        numbers = set(row.get("_numbers", parse_number_set(row.get("予想番号"))))
        main_matches = len(numbers & actual)
        bonus_matches = len(numbers & bonus)
        rows.append({
            "game": game,
            "prediction_id": _text(row.get("予想ID")),
            "draw_no": draw_no,
            "draw_date": draw_date,
            "model": _text(row.get("使用モデル")),
            "candidate_no": _int(row.get("候補番号")) or 0,
            "numbers": "-".join(f"{number:02d}" for number in sorted(numbers)),
            "main_matches": main_matches,
            "bonus_matches": bonus_matches,
            "matches_with_bonus": main_matches + bonus_matches,
        })
    return pd.DataFrame(rows, columns=columns)


def build_draw_metrics(ticket_metrics: pd.DataFrame, results: pd.DataFrame, game: str) -> pd.DataFrame:
    columns = ["game", "model", "draw_no", "draw_date", "ticket_count", "average_main_matches", "best_main_matches", "main_number_coverage", "coverage_rate", "matches_per_ticket", "coverage_per_ticket", "hit3_draw", "hit4_draw", "rolling_10_average", "rolling_30_average", "performance_trend"]
    if ticket_metrics is None or ticket_metrics.empty:
        return _empty(columns)
    actual_map = _result_map(results)
    rows = []
    for (model, draw_no), group in ticket_metrics.groupby(["model", "draw_no"], sort=True):
        actual, _, draw_date = actual_map[int(draw_no)]
        predicted_union = set()
        for value in group["numbers"]:
            predicted_union.update(parse_number_set(value))
        coverage = len(predicted_union & actual)
        ticket_count = len(group)
        best = int(group["main_matches"].max())
        rows.append({
            "game": game, "model": model, "draw_no": int(draw_no), "draw_date": draw_date,
            "ticket_count": ticket_count, "average_main_matches": float(group["main_matches"].mean()),
            "best_main_matches": best, "main_number_coverage": coverage,
            "coverage_rate": coverage / max(len(actual), 1), "matches_per_ticket": float(group["main_matches"].sum()) / ticket_count,
            "coverage_per_ticket": coverage / ticket_count, "hit3_draw": int(best >= 3), "hit4_draw": int(best >= 4),
        })
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["model", "draw_no"]).reset_index(drop=True)
    frame["rolling_10_average"] = frame.groupby("model")["average_main_matches"].transform(lambda values: values.rolling(10, min_periods=1).mean())
    frame["rolling_30_average"] = frame.groupby("model")["average_main_matches"].transform(lambda values: values.rolling(30, min_periods=1).mean())
    frame["performance_trend"] = frame.groupby("model")["rolling_10_average"].diff().fillna(0.0)
    return frame.reindex(columns=columns)


def _streak(values: list[int], predicate) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if predicate(value) else 0
        longest = max(longest, current)
    return longest


def _bootstrap_interval(values: list[float], seed: int, repetitions: int = 1000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(repetitions):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    means.sort()
    return means[int(repetitions * 0.025)], means[min(int(repetitions * 0.975), repetitions - 1)]


def _stable_seed(game: str, draw_no: int, model: str = "") -> int:
    digest = hashlib.sha256(f"accuracy-baseline-v1|{game}|{draw_no}|{model}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def build_random_draw_metrics(draw_metrics: pd.DataFrame, results: pd.DataFrame, game: str) -> pd.DataFrame:
    columns = ["game", "model", "draw_no", "ticket_count", "random_average_matches", "random_best_matches", "random_coverage", "random_baseline_source"]
    if draw_metrics is None or draw_metrics.empty:
        return _empty(columns)
    config = GAME_CONFIGS[game]
    actual_map = _result_map(results)
    random_names = draw_metrics["model"].astype(str).str.lower().str.contains("random|ランダム", regex=True)
    formal_random = draw_metrics[random_names]
    rows = []
    for _, source in draw_metrics.iterrows():
        draw_no = int(source["draw_no"])
        actual = actual_map[draw_no][0]
        ticket_count = int(source["ticket_count"])
        saved_baseline = formal_random[(formal_random["draw_no"] == draw_no) & (formal_random["ticket_count"] == ticket_count)]
        if not saved_baseline.empty:
            baseline = saved_baseline.iloc[0]
            average = float(baseline["average_main_matches"])
            best = int(baseline["best_main_matches"])
            coverage = int(baseline["main_number_coverage"])
            source_label = "正式保存random予測"
        else:
            rng = random.Random(_stable_seed(game, draw_no, _text(source["model"])))
            tickets = [set(rng.sample(range(1, config.number_max + 1), config.draw_size)) for _ in range(ticket_count)]
            matches = [len(ticket & actual) for ticket in tickets]
            union = set().union(*tickets) if tickets else set()
            average = sum(matches) / len(matches)
            best = max(matches)
            coverage = len(union & actual)
            source_label = "固定seed研究用random"
        rows.append({
            "game": game, "model": source["model"], "draw_no": draw_no, "ticket_count": ticket_count,
            "random_average_matches": average, "random_best_matches": best,
            "random_coverage": coverage, "random_baseline_source": source_label,
        })
    return pd.DataFrame(rows, columns=columns)


def _period_mean(draws: pd.DataFrame, count: int | None, column: str) -> float | None:
    if draws.empty:
        return None
    source = draws.sort_values("draw_no").tail(count) if count else draws
    return float(source[column].mean())


def build_model_accuracy(ticket_metrics: pd.DataFrame, draw_metrics: pd.DataFrame, random_metrics: pd.DataFrame, game: str) -> pd.DataFrame:
    config = GAME_CONFIGS[game]
    columns = ["game", "model", "ranking_eligible", "data_status", "evaluated_draws", "evaluated_tickets", "average_main_matches", "median_main_matches", "std_main_matches", *[f"match_{i}_count" for i in range(config.draw_size + 1)], "max_main_matches", "max_match_draws", *[f"hit_{i}_plus_rate" for i in range(1, config.draw_size + 1)], "bonus_match_count", "average_matches_with_bonus", "recent_10_average", "recent_30_average", "all_period_average", "first_half_average", "second_half_average", "longest_poor_streak", "longest_good_streak", "average_draw_best", "average_coverage", "average_ticket_efficiency", "random_average_delta", "random_hit3_rate_delta", "random_best_delta", "random_coverage_delta", "random_wins", "random_ties", "random_losses", "improvement_rate", "random_delta_ci_low", "random_delta_ci_high", "statistical_advantage"]
    if ticket_metrics is None or ticket_metrics.empty:
        return _empty(columns)
    rows = []
    for model, tickets in ticket_metrics.groupby("model", sort=True):
        tickets = tickets.sort_values(["draw_no", "candidate_no"])
        draws = draw_metrics[draw_metrics["model"] == model].sort_values("draw_no")
        random_draws = random_metrics[random_metrics["model"] == model]
        paired = draws.merge(random_draws, on=["game", "model", "draw_no", "ticket_count"], how="left")
        values = tickets["main_matches"].astype(float)
        midpoint = max(len(draws) // 2, 1)
        first = draws.iloc[:midpoint]
        second = draws.iloc[midpoint:]
        diffs = (paired["average_main_matches"] - paired["random_average_matches"]).dropna().tolist()
        ci_low, ci_high = _bootstrap_interval(diffs, _stable_seed(game, 0, str(model)))
        comparison = paired["average_main_matches"] - paired["random_average_matches"]
        row = {
            "game": game, "model": model, "ranking_eligible": len(draws) >= MIN_RANKING_DRAWS,
            "data_status": "評価可能" if len(draws) >= MIN_RANKING_DRAWS else f"データ不足（{len(draws)}/{MIN_RANKING_DRAWS}開催回）",
            "evaluated_draws": len(draws), "evaluated_tickets": len(tickets),
            "average_main_matches": float(values.mean()), "median_main_matches": float(values.median()),
            "std_main_matches": float(values.std(ddof=0)), "max_main_matches": int(values.max()),
            "max_match_draws": ",".join(str(int(value)) for value in sorted(tickets.loc[values == values.max(), "draw_no"].unique())),
            "bonus_match_count": int(tickets["bonus_matches"].sum()), "average_matches_with_bonus": float(tickets["matches_with_bonus"].mean()),
            "recent_10_average": _period_mean(draws, 10, "average_main_matches"), "recent_30_average": _period_mean(draws, 30, "average_main_matches"),
            "all_period_average": _period_mean(draws, None, "average_main_matches"),
            "first_half_average": float(first["average_main_matches"].mean()) if not first.empty else None,
            "second_half_average": float(second["average_main_matches"].mean()) if not second.empty else None,
            "longest_poor_streak": _streak(draws["best_main_matches"].astype(int).tolist(), lambda value: value < 2),
            "longest_good_streak": _streak(draws["best_main_matches"].astype(int).tolist(), lambda value: value >= 3),
            "average_draw_best": float(draws["best_main_matches"].mean()), "average_coverage": float(draws["main_number_coverage"].mean()),
            "average_ticket_efficiency": float(draws["coverage_per_ticket"].mean()),
            "random_average_delta": float(diffs and sum(diffs) / len(diffs) or 0.0),
            "random_hit3_rate_delta": float((draws["hit3_draw"].mean() - (paired["random_best_matches"] >= 3).mean()) if not paired.empty else 0.0),
            "random_best_delta": float((paired["best_main_matches"] - paired["random_best_matches"]).mean()) if not paired.empty else 0.0,
            "random_coverage_delta": float((paired["main_number_coverage"] - paired["random_coverage"]).mean()) if not paired.empty else 0.0,
            "random_wins": int((comparison > 0).sum()), "random_ties": int((comparison == 0).sum()), "random_losses": int((comparison < 0).sum()),
            "improvement_rate": float((comparison > 0).mean()) if len(comparison) else 0.0,
            "random_delta_ci_low": ci_low, "random_delta_ci_high": ci_high,
            "statistical_advantage": "優位差あり" if ci_low is not None and ci_low > 0 else "統計的な優位性を確認できず",
        }
        for match_count in range(config.draw_size + 1):
            row[f"match_{match_count}_count"] = int((values == match_count).sum())
        for threshold in range(1, config.draw_size + 1):
            row[f"hit_{threshold}_plus_rate"] = float((values >= threshold).mean())
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=columns)


def build_overlap_matrix(ticket_metrics: pd.DataFrame, game: str) -> pd.DataFrame:
    columns = ["game", "model_a", "model_b", "common_draws", "average_jaccard", "average_common_numbers", "candidate_union_overlap_rate", "model_a_unique_rate", "model_b_unique_rate", "exact_match_tickets"]
    if ticket_metrics is None or ticket_metrics.empty:
        return _empty(columns)
    rows = []
    models = sorted(ticket_metrics["model"].unique())
    for index, model_a in enumerate(models):
        for model_b in models[index:]:
            a = ticket_metrics[ticket_metrics["model"] == model_a]
            b = ticket_metrics[ticket_metrics["model"] == model_b]
            common_draws = sorted(set(a["draw_no"]) & set(b["draw_no"]))
            stats = []
            exact = 0
            for draw_no in common_draws:
                a_tickets = [frozenset(parse_number_set(value)) for value in a.loc[a["draw_no"] == draw_no, "numbers"]]
                b_tickets = [frozenset(parse_number_set(value)) for value in b.loc[b["draw_no"] == draw_no, "numbers"]]
                a_union = set().union(*a_tickets) if a_tickets else set()
                b_union = set().union(*b_tickets) if b_tickets else set()
                union = a_union | b_union
                common = a_union & b_union
                exact += len(set(a_tickets) & set(b_tickets))
                stats.append((len(common) / len(union) if union else 0.0, len(common), len(common) / max(min(len(a_union), len(b_union)), 1), len(a_union - b_union) / max(len(a_union), 1), len(b_union - a_union) / max(len(b_union), 1)))
            if stats:
                rows.append({"game": game, "model_a": model_a, "model_b": model_b, "common_draws": len(stats), "average_jaccard": sum(x[0] for x in stats) / len(stats), "average_common_numbers": sum(x[1] for x in stats) / len(stats), "candidate_union_overlap_rate": sum(x[2] for x in stats) / len(stats), "model_a_unique_rate": sum(x[3] for x in stats) / len(stats), "model_b_unique_rate": sum(x[4] for x in stats) / len(stats), "exact_match_tickets": exact})
    return pd.DataFrame(rows, columns=columns)


def build_marginal_contribution(ticket_metrics: pd.DataFrame, results: pd.DataFrame, game: str) -> pd.DataFrame:
    columns = ["game", "model", "method", "evaluated_draws", "all_models_average_match", "without_model_average_delta", "only_model_average_match", "random_replacement_average_delta", "best_match_delta", "coverage_delta", "hit3_draw_delta", "hit4_draw_delta", "overlap_reduction", "unique_hit_numbers_provided", "positive_contribution_draws", "negative_contribution_draws", "no_contribution_draws"]
    if ticket_metrics is None or ticket_metrics.empty:
        return _empty(columns)
    actual_map = _result_map(results)
    rows = []
    for model in sorted(ticket_metrics["model"].unique()):
        per_draw = []
        for draw_no, all_group in ticket_metrics.groupby("draw_no"):
            target = all_group[all_group["model"] == model]
            if target.empty:
                continue
            other = all_group[all_group["model"] != model]
            actual = actual_map[int(draw_no)][0]
            all_avg = float(all_group["main_matches"].mean())
            without_avg = float(other["main_matches"].mean()) if not other.empty else 0.0
            only_avg = float(target["main_matches"].mean())
            all_union = set().union(*(set(parse_number_set(value)) for value in all_group["numbers"]))
            other_union = set().union(*(set(parse_number_set(value)) for value in other["numbers"])) if not other.empty else set()
            target_union = set().union(*(set(parse_number_set(value)) for value in target["numbers"]))
            coverage_delta = len(all_union & actual) - len(other_union & actual)
            best_delta = int(all_group["main_matches"].max()) - (int(other["main_matches"].max()) if not other.empty else 0)
            rng = random.Random(_stable_seed(game, int(draw_no), str(model)))
            config = GAME_CONFIGS[game]
            random_matches = [len(set(rng.sample(range(1, config.number_max + 1), config.draw_size)) & actual) for _ in range(len(target))]
            replaced_avg = (float(other["main_matches"].sum()) + sum(random_matches)) / (len(other) + len(random_matches))
            per_draw.append({"all_avg": all_avg, "without_delta": all_avg - without_avg, "only_avg": only_avg, "replace_delta": all_avg - replaced_avg, "best_delta": best_delta, "coverage_delta": coverage_delta, "hit3_delta": int(all_group["main_matches"].max() >= 3) - int((not other.empty) and other["main_matches"].max() >= 3), "hit4_delta": int(all_group["main_matches"].max() >= 4) - int((not other.empty) and other["main_matches"].max() >= 4), "overlap_reduction": len(target_union & other_union) / max(len(target_union), 1), "unique_hits": len((target_union - other_union) & actual), "sign": 1 if coverage_delta > 0 or best_delta > 0 else (-1 if all_avg < without_avg else 0)})
        frame = pd.DataFrame(per_draw)
        if frame.empty:
            continue
        rows.append({"game": game, "model": model, "method": "保存済み予測集合による研究用近似（正式アンサンブル再現ではない）", "evaluated_draws": len(frame), "all_models_average_match": frame["all_avg"].mean(), "without_model_average_delta": frame["without_delta"].mean(), "only_model_average_match": frame["only_avg"].mean(), "random_replacement_average_delta": frame["replace_delta"].mean(), "best_match_delta": frame["best_delta"].mean(), "coverage_delta": frame["coverage_delta"].mean(), "hit3_draw_delta": int(frame["hit3_delta"].sum()), "hit4_draw_delta": int(frame["hit4_delta"].sum()), "overlap_reduction": frame["overlap_reduction"].mean(), "unique_hit_numbers_provided": int(frame["unique_hits"].sum()), "positive_contribution_draws": int((frame["sign"] > 0).sum()), "negative_contribution_draws": int((frame["sign"] < 0).sum()), "no_contribution_draws": int((frame["sign"] == 0).sum())})
    return pd.DataFrame(rows, columns=columns)


def build_rankings(model_accuracy: pd.DataFrame, overlap: pd.DataFrame, marginal: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if model_accuracy is None or model_accuracy.empty:
        return {}
    eligible = model_accuracy[model_accuracy["ranking_eligible"] == True].copy()  # noqa: E712
    if eligible.empty:
        return {}
    uniqueness = {}
    if overlap is not None and not overlap.empty:
        for model in eligible["model"]:
            pairs = overlap[(overlap["model_a"] == model) | (overlap["model_b"] == model)]
            pairs = pairs[pairs["model_a"] != pairs["model_b"]]
            uniqueness[model] = 1.0 - float(pairs["average_jaccard"].mean()) if not pairs.empty else 1.0
    contribution = marginal.set_index("model")["coverage_delta"].to_dict() if marginal is not None and not marginal.empty else {}
    eligible["uniqueness_score"] = eligible["model"].map(uniqueness).fillna(1.0)
    eligible["marginal_score"] = eligible["model"].map(contribution).fillna(0.0)
    specs = {
        "平均一致数": ("average_main_matches", False), "開催回最高一致数": ("average_draw_best", False),
        "3個以上一致率": ("hit_3_plus_rate", False), "4個以上一致率": ("hit_4_plus_rate", False),
        "安定性": ("std_main_matches", True), "直近10回": ("recent_10_average", False),
        "直近30回": ("recent_30_average", False), "長期成績": ("all_period_average", False),
        "ランダム基準超過": ("random_average_delta", False), "独自性": ("uniqueness_score", False),
        "限界貢献度": ("marginal_score", False), "口数効率": ("average_ticket_efficiency", False),
    }
    return {name: eligible.sort_values(column, ascending=ascending)[["model", column, "evaluated_draws", "data_status"]].reset_index(drop=True) for name, (column, ascending) in specs.items() if column in eligible}


def analyze_game(predictions: pd.DataFrame, results: pd.DataFrame, game: str, verification: pd.DataFrame | None = None) -> dict[str, Any]:
    classified = classify_predictions(predictions, results, game, verification)
    tickets = build_ticket_metrics(classified["eligible"], results, game)
    draws = build_draw_metrics(tickets, results, game)
    random_draws = build_random_draw_metrics(draws, results, game)
    accuracy = build_model_accuracy(tickets, draws, random_draws, game)
    overlap = build_overlap_matrix(tickets, game)
    marginal = build_marginal_contribution(tickets, results, game)
    rankings = build_rankings(accuracy, overlap, marginal)
    return {**classified, "ticket_metrics": tickets, "draw_metrics": draws, "random_draw_metrics": random_draws, "model_accuracy": accuracy, "overlap": overlap, "marginal": marginal, "rankings": rankings}


def build_accuracy_baseline(loto6_predictions: pd.DataFrame, loto6_results: pd.DataFrame, loto7_predictions: pd.DataFrame, loto7_results: pd.DataFrame, generated_at: datetime | None = None, loto6_verification: pd.DataFrame | None = None, loto7_verification: pd.DataFrame | None = None) -> dict[str, Any]:
    generated_at = generated_at or datetime.now()
    games = {
        "loto6": analyze_game(loto6_predictions, loto6_results, "loto6", loto6_verification),
        "loto7": analyze_game(loto7_predictions, loto7_results, "loto7", loto7_verification),
    }
    summary = {
        "schema_version": "1.0", "report_type": "Prediction Accuracy Research Baseline",
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "official_definition": "保存済み正式予測で、抽せん日より前の保存を確認でき、結果・番号・モデル・開催回が有効なもの",
        "same_day_policy": "抽せん時刻が保存されていないため、同日予測は判定不能として正式ランキングから除外",
        "regenerated_policy": "再生成・test・demo・sample識別子を含む行は別分類し正式ランキングから除外",
        "random_policy": "正式random行がない比較ではgame・開催回・モデル別固定seedの研究用randomを使用し、保存しない",
        "marginal_policy": "保存済みモデル予測集合による研究用近似であり正式アンサンブル条件の再現ではない",
        "minimum_ranking_draws": MIN_RANKING_DRAWS,
        "games": {},
    }
    for game, analysis in games.items():
        eligible = analysis["eligible"]
        classified = analysis["classified"]
        summary["games"][game] = {
            "saved_predictions": len(classified), "eligible_predictions": len(eligible),
            "formally_verified_predictions": int((classified["分類"] == "正式検証済み予測").sum()) if not classified.empty else 0,
            "eligible_draws": int(eligible["開催回"].nunique()) if not eligible.empty else 0,
            "unverified_predictions": len(analysis["unverified"]), "excluded_predictions": len(analysis["excluded"]),
            "quality_issues": len(analysis["quality"]),
            "evaluated_models": sorted(analysis["model_accuracy"]["model"].tolist()) if not analysis["model_accuracy"].empty else [],
            "insufficient_models": sorted(analysis["model_accuracy"].loc[analysis["model_accuracy"]["ranking_eligible"] == False, "model"].tolist()) if not analysis["model_accuracy"].empty else [],  # noqa: E712
        }
    return {"summary": summary, "games": games}


def load_accuracy_baseline(data_dir: Path) -> dict[str, Any]:
    return build_accuracy_baseline(
        read_csv_safely(data_dir / "predictions.csv"), read_csv_safely(data_dir / "results.csv"),
        read_csv_safely(data_dir / "loto7_predictions.csv"), read_csv_safely(data_dir / "loto7_results.csv"),
        loto6_verification=read_csv_safely(data_dir / "verification" / "verification_reports.csv"),
        loto7_verification=read_csv_safely(data_dir / "verification" / "loto7_verification_reports.csv"),
    )


def planned_report_files(report_root: Path, generated_at: str | None = None) -> list[Path]:
    stamp = generated_at or datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = report_root / "snapshots" / stamp
    return [snapshot / name for name in REPORT_FILES.values()] + [snapshot / SUMMARY_FILE, report_root / "latest.json"]


def save_accuracy_baseline_report(report: dict[str, Any], report_root: Path, generated_at: datetime | None = None) -> dict[str, Any]:
    generated_at = generated_at or datetime.now()
    stamp = generated_at.strftime("%Y%m%d_%H%M%S")
    snapshot = report_root / "snapshots" / stamp
    if snapshot.exists():
        raise FileExistsError(f"snapshot already exists: {snapshot}")
    snapshot.mkdir(parents=True, exist_ok=False)
    frames = {
        "loto6_model_accuracy": report["games"]["loto6"]["model_accuracy"],
        "loto7_model_accuracy": report["games"]["loto7"]["model_accuracy"],
        "loto6_draw_metrics": report["games"]["loto6"]["draw_metrics"],
        "loto7_draw_metrics": report["games"]["loto7"]["draw_metrics"],
        "loto6_overlap": report["games"]["loto6"]["overlap"],
        "loto7_overlap": report["games"]["loto7"]["overlap"],
        "loto6_marginal": report["games"]["loto6"]["marginal"],
        "loto7_marginal": report["games"]["loto7"]["marginal"],
        "loto6_quality": report["games"]["loto6"]["quality"],
        "loto7_quality": report["games"]["loto7"]["quality"],
    }
    written = []
    for key, filename in REPORT_FILES.items():
        path = snapshot / filename
        frames[key].to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    summary_path = snapshot / SUMMARY_FILE
    summary_path.write_text(json.dumps(report["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(summary_path)
    latest_path = report_root / "latest.json"
    latest_payload = {"snapshot": snapshot.relative_to(report_root).as_posix(), "generated_at": generated_at.isoformat(timespec="seconds"), "files": [path.name for path in written]}
    temporary = latest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(latest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(latest_path)
    return {"snapshot": snapshot, "latest": latest_path, "files": written, "row_counts": {key: len(frame) for key, frame in frames.items()}}
