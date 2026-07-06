from collections import Counter, OrderedDict
from datetime import datetime
from itertools import combinations
import json
import random
from pathlib import Path

import pandas as pd


PROJECT_JAPANESE_NAME = "分析研究所"
PROJECT_ENGLISH_NAME = "Prediction Research Lab"
PROJECT_SHORT_NAME = "PRL"

ANTI_POPULAR_EXPECTED_VALUE_KEY = "anti_popular_expected_value"
ANTI_POPULAR_EXPECTED_VALUE_LABEL = "人と被りにくい期待値最大化モデル"
CHAMINI6_GOD_MODE_KEY = "chamini6_god_mode"
CHAMINI6_GOD_MODE_LABEL = "Chamini6 God Mode"

LOTO_MODEL_LABELS = OrderedDict(
    [
        ("frequency_analysis", "出現頻度分析"),
        ("cold_analysis", "コールドナンバー分析"),
        ("hot_analysis", "ホットナンバー分析"),
        ("odd_even_analysis", "奇数偶数分析"),
        ("sum_value_analysis", "合計値分析"),
        ("high_low_analysis", "高低分析"),
        ("consecutive_analysis", "連番分析"),
        ("last_digit_analysis", "下一桁分析"),
        ("pair_analysis", "ペア分析"),
        ("triple_analysis", "トリプル分析"),
        ("bonus_analysis", "ボーナス数字分析"),
        ("markov_chain", "マルコフ連鎖"),
        ("bayesian_estimation", "ベイズ推定"),
        ("monte_carlo", "モンテカルロシミュレーション"),
        ("machine_learning", "機械学習モデル"),
        ("random_baseline", "ランダム予測モデル"),
        (ANTI_POPULAR_EXPECTED_VALUE_KEY, ANTI_POPULAR_EXPECTED_VALUE_LABEL),
        (CHAMINI6_GOD_MODE_KEY, CHAMINI6_GOD_MODE_LABEL),
    ]
)

FUTURE_LOTO_MODEL_LABELS = OrderedDict(
    [
        ("cycle_analysis", "周期分析"),
        ("clustering_analysis", "クラスタリング分析"),
        ("outlier_analysis", "異常値分析"),
        ("network_analysis", "ネットワーク分析"),
        ("ensemble_learning", "アンサンブル学習"),
        ("non_overlap_jackpot", "人と被りにくい高額当選狙いモデル"),
    ]
)

ARL_MODEL_LABELS = LOTO_MODEL_LABELS

MODEL_ALIASES = {
    "frequency_balance": "frequency_analysis",
    "recent_trend": "hot_analysis",
    "overdue_interval": "cold_analysis",
    "ai_improved": "machine_learning",
    "non_overlap": "random_baseline",
    "fixed_numbers": "frequency_analysis",
    "variable_numbers": "cold_analysis",
    "hybrid": "machine_learning",
    "hot10": "hot_analysis",
    "hot20": "hot_analysis",
    "hot50": "hot_analysis",
    "cold_revival": "cold_analysis",
    "bonus_promotion": "bonus_analysis",
    "anti_popular": ANTI_POPULAR_EXPECTED_VALUE_KEY,
    "anti_popular_ev": ANTI_POPULAR_EXPECTED_VALUE_KEY,
    "non_overlap_jackpot": ANTI_POPULAR_EXPECTED_VALUE_KEY,
    "chamini6": CHAMINI6_GOD_MODE_KEY,
    "god_mode": CHAMINI6_GOD_MODE_KEY,
}

AI_MODEL_SCORE_COLUMNS = {
    "frequency_analysis": ["出現頻度スコア"],
    "hot_analysis": ["直近傾向スコア"],
    "cold_analysis": ["未出現期間スコア"],
    "bonus_analysis": ["ボーナス傾向スコア"],
    "machine_learning": ["総合スコア", "出現頻度スコア", "直近傾向スコア", "未出現期間スコア", "ボーナス傾向スコア"],
    "bayesian_estimation": ["総合スコア", "出現頻度スコア", "直近傾向スコア"],
    "markov_chain": ["総合スコア", "直近傾向スコア"],
    "pair_analysis": ["総合スコア"],
    "triple_analysis": ["総合スコア"],
    "odd_even_analysis": ["総合スコア"],
    "high_low_analysis": ["総合スコア"],
    "sum_value_analysis": ["総合スコア"],
    "consecutive_analysis": ["総合スコア"],
    "last_digit_analysis": ["総合スコア"],
    "monte_carlo": ["総合スコア"],
    "random_baseline": ["総合スコア"],
    ANTI_POPULAR_EXPECTED_VALUE_KEY: ["総合スコア"],
    CHAMINI6_GOD_MODE_KEY: ["総合スコア"],
}

CONTRIBUTION_COLUMNS = [
    "研究所",
    "予想ID",
    "開催回",
    "候補番号",
    "予想数字",
    "選出モデル",
    "複数モデル一致",
    "的中",
    "寄与スコア",
    "保存日時",
]

RESEARCH_CYCLE_COLUMNS = [
    "研究所",
    "予想ID",
    "開催回",
    "仮説",
    "分析",
    "予想",
    "結果",
    "検証",
    "改善",
    "再予想",
    "保存日時",
]

VIDEO_HYPOTHESIS_COLUMNS = ["動画名", "仮説", "法則抽出", "成績", "採用可否", "保存日時"]

WINNING_CONDITION_HISTORY_COLUMNS = [
    "lottery_type",
    "draw_no",
    "prediction_id",
    "prediction_date",
    "predicted_numbers",
    "actual_numbers",
    "matched_numbers",
    "missed_numbers",
    "should_have_included_numbers",
    "should_have_excluded_numbers",
    "matched_count",
    "failure_reason",
    "winning_condition_analysis",
    "useful_models",
    "weak_models",
    "weight_up_models",
    "weight_down_models",
    "model_improvement_analysis",
    "ensemble_analysis",
    "next_hypothesis",
    "created_at",
]

MODEL_IMPROVEMENT_HISTORY_COLUMNS = [
    "lottery_type",
    "draw_no",
    "prediction_id",
    "model_name",
    "predicted_numbers",
    "actual_numbers",
    "matched_numbers",
    "missed_numbers",
    "should_have_included_numbers",
    "should_have_excluded_numbers",
    "matched_count",
    "needed_conditions",
    "next_hypothesis",
    "created_at",
]

PURCHASE_COLUMNS = [
    "lottery_type",
    "draw_no",
    "purchase_date",
    "numbers",
    "prediction_method",
    "model_name",
    "ticket_count",
    "cost",
    "result_numbers",
    "bonus_numbers",
    "matched_count",
    "bonus_matched_count",
    "prize_rank",
    "payout",
    "profit_loss",
    "status",
    "notes",
    "created_at",
]

PURCHASE_DISPLAY_COLUMNS = [
    "開催回",
    "購入日",
    "購入番号",
    "予測方式",
    "モデル名",
    "口数",
    "購入金額",
    "一致数",
    "BONUS一致数",
    "当選等級",
    "払戻金",
    "収支",
    "状態",
    "メモ",
]
HIGH_PRIZE_TICKET_ROLES = [
    "本命・安定型",
    "直近トレンド型",
    "高額当選狙い・低人気型",
]

PREDICTION_PATTERN_ROLES = [
    {
        "pattern_key": "A",
        "pattern_label": "Pattern A",
        "role_label": "本命型",
        "emphasized_factors": "既存分析モデルの上位スコア、奇数偶数、高低、合計値の安定性",
        "selection_reason": "過去成績と候補スコアを優先し、検証しやすい本命軸として採用。",
    },
    {
        "pattern_key": "B",
        "pattern_label": "Pattern B",
        "role_label": "バランス型",
        "emphasized_factors": "本命コア1〜2個、モデルスコア、数字帯の分散、他Patternとの重複抑制",
        "selection_reason": "本命型と完全には重ねず、共通コアを残してバランスを調整。",
    },
    {
        "pattern_key": "C",
        "pattern_label": "Pattern C",
        "role_label": "チャレンジ型",
        "emphasized_factors": "31超え数字、前回当選番号との適度な重複、3連続除外、人と被りにくい構成",
        "selection_reason": "当選時の分配リスクを下げる可能性を研究する補助モデルとして採用。",
    },
]

CHAMINI6_GOD_MODE_ENGINE = {
    "engine_key": CHAMINI6_GOD_MODE_KEY,
    "engine_name": CHAMINI6_GOD_MODE_LABEL,
    "status": "active",
    "lottery_types": ["loto6", "loto7"],
    "contract": "independent_engine",
    "notes": "既存分析エンジンを削除せず、独立した統合候補として接続する正式エンジン。",
}

PREDICTION_ENGINE_REGISTRY = OrderedDict(
    [
        ("standard_loto_models", {"engine_name": "既存ロト分析モデル群", "status": "active"}),
        (ANTI_POPULAR_EXPECTED_VALUE_KEY, {"engine_name": ANTI_POPULAR_EXPECTED_VALUE_LABEL, "status": "active_auxiliary"}),
        (CHAMINI6_GOD_MODE_ENGINE["engine_key"], CHAMINI6_GOD_MODE_ENGINE),
    ]
)

CHAMINI6_COMPONENT_MODEL_KEYS = [
    "frequency_analysis",
    "cold_analysis",
    "hot_analysis",
    "odd_even_analysis",
    "sum_value_analysis",
    "high_low_analysis",
    "consecutive_analysis",
    "last_digit_analysis",
    "pair_analysis",
    "triple_analysis",
    "bonus_analysis",
    "markov_chain",
    "bayesian_estimation",
    "monte_carlo",
    "machine_learning",
    ANTI_POPULAR_EXPECTED_VALUE_KEY,
]

ENSEMBLE_PREDICTION_COLUMNS = [
    "lottery_type",
    "target_draw",
    "prediction_date",
    "ticket_no",
    "ticket_role",
    "numbers",
    "adopted_models",
    "selection_reason",
    "prediction_score",
    "expected_score",
    "low_popularity_score",
    "past_similarity",
    "risk_level",
    "created_at",
]

TICKET_STRATEGY_COLUMNS = [
    "lottery_type",
    "target_draw",
    "prediction_date",
    "ticket_no",
    "ticket_role",
    "numbers",
    "adopted_models",
    "selection_reason",
    "expected_score",
    "low_popularity_score",
    "past_similarity",
    "risk_level",
    "created_at",
]

POPULARITY_SCORE_COLUMNS = [
    "lottery_type",
    "target_draw",
    "prediction_date",
    "ticket_no",
    "numbers",
    "low_popularity_score",
    "birthday_bias_score",
    "consecutive_score",
    "regularity_score",
    "last_digit_score",
    "sum_balance_score",
    "past_similarity_score",
    "visual_neatness_score",
    "risk_level",
    "created_at",
]

CONTINUOUS_WIN_RESEARCH_COLUMNS = [
    "prediction_date",
    "target_draw",
    "lottery_type",
    "numbers",
    "ticket_role",
    "adopted_models",
    "prediction_score",
    "low_popularity_score",
    "actual_numbers",
    "bonus_numbers",
    "matched_count",
    "bonus_matched_count",
    "missed_numbers",
    "hit_numbers",
    "expected_value",
    "failure_reason",
    "improvement_plan",
    "next_hypothesis",
    "created_at",
]

BACKTEST_SUMMARY_COLUMNS = [
    "lottery_type",
    "period",
    "model_name",
    "model_type",
    "evaluated_draws",
    "average_match",
    "max_match",
    "match3_rate",
    "match4_rate",
    "bonus_match_rate",
    "stability",
    "recent_score",
    "long_score",
    "random_delta",
    "expected_value",
    "created_at",
]

MODEL_WEIGHT_HISTORY_COLUMNS = [
    "lottery_type",
    "created_at",
    "model_key",
    "model_name",
    "evaluation_count",
    "average_match",
    "recent5_score",
    "recent10_score",
    "long_score",
    "stability",
    "expected_value",
    "bonus_match_rate",
    "raw_score",
    "applied_weight",
    "status",
]


def canonical_model_key(model_key):
    return MODEL_ALIASES.get(str(model_key), str(model_key))


def model_label(model_key):
    return ARL_MODEL_LABELS.get(canonical_model_key(model_key), str(model_key))


def safe_int(value, default=0):
    try:
        return int(str(value).replace(",", "").strip())
    except Exception:
        return default


def numbers_to_text(numbers):
    return "-".join(f"{int(number):02d}" for number in sorted(numbers))


def parse_numbers(value, number_max=None):
    numbers = []
    for part in str(value).replace(",", "-").replace(" ", "-").split("-"):
        number = safe_int(part, None)
        if number is None:
            continue
        if number_max is None or 1 <= number <= number_max:
            numbers.append(number)
    return sorted(numbers)


def build_fixed_prediction_overview(predictions, official_results=None, draw_size=6, number_max=None):
    columns = [
        "開催回",
        "予想日",
        "候補番号",
        "使用モデル",
        "予想番号",
        "予測スコア",
        "高額当選・連続当選モード",
        "検証状態",
        "保存日時",
    ]
    if predictions is None or predictions.empty or "開催回" not in predictions:
        return pd.DataFrame(columns=columns), None, "予想履歴がありません。"

    df = predictions.copy()
    df["_draw_no"] = pd.to_numeric(df["開催回"], errors="coerce")
    df = df[df["_draw_no"].notna()]
    if df.empty:
        return pd.DataFrame(columns=columns), None, "開催回を読み取れる予想履歴がありません。"
    df["_draw_no"] = df["_draw_no"].astype(int)

    official_rounds = set()
    if official_results is not None and not official_results.empty and "開催回" in official_results:
        official_values = pd.to_numeric(official_results["開催回"], errors="coerce").dropna()
        official_rounds = {int(value) for value in official_values}

    pending = df[~df["_draw_no"].isin(official_rounds)] if official_rounds else df
    source = pending if not pending.empty else df
    target_round = int(source["_draw_no"].max())
    target = source[source["_draw_no"] == target_round].copy()
    status = "結果待ち" if target_round not in official_rounds else "公式結果あり・検証可能"

    if "予測スコア" not in target:
        target["予測スコア"] = "-"
    if "保存日時" not in target:
        target["保存日時"] = "-"
    for column in ("予想日", "候補番号", "使用モデル", "予想番号"):
        if column not in target:
            target[column] = ""

    def normalized_numbers(value):
        numbers = parse_numbers(value, number_max)
        if len(numbers) == draw_size and len(set(numbers)) == draw_size:
            return numbers_to_text(numbers)
        return str(value or "")

    def research_mode(model_name):
        text = str(model_name or "")
        mode_words = ("高額", "連続", "アンサンブル", "低人気")
        return "対象" if any(word in text for word in mode_words) else "通常予測"

    target["開催回"] = target["_draw_no"].astype(int)
    target["予想番号"] = target["予想番号"].map(normalized_numbers)
    target["高額当選・連続当選モード"] = target["使用モデル"].map(research_mode)
    target["検証状態"] = status
    target = target.sort_values(["開催回", "候補番号", "使用モデル"], ascending=[False, True, True])
    return target.reindex(columns=columns), target_round, status


def validate_purchase_numbers(value, draw_size, number_max):
    raw_text = str(value or "").strip()
    if not raw_text:
        return [], ["購入番号を入力してください。"]
    tokens = [part for part in raw_text.replace(",", "-").replace(" ", "-").replace("　", "-").split("-") if part != ""]
    numbers = []
    errors = []
    for token in tokens:
        number = safe_int(token, None)
        if number is None:
            errors.append(f"{token} は数字として読み取れません。")
            continue
        if not 1 <= number <= number_max:
            errors.append(f"{number:02d} は範囲外です。1〜{number_max}で入力してください。")
            continue
        numbers.append(number)
    duplicates = sorted(number for number, count in Counter(numbers).items() if count >= 2)
    if duplicates:
        errors.append("重複している数字があります: " + numbers_to_text(duplicates))
    unique_numbers = sorted(set(numbers))
    if len(unique_numbers) != draw_size:
        errors.append(f"{draw_size}個の数字を入力してください。現在は{len(unique_numbers)}個です。")
    return unique_numbers, errors


def clean_number_rows(number_rows, number_max):
    cleaned = []
    for row in number_rows or []:
        numbers = sorted({safe_int(number) for number in row if 1 <= safe_int(number) <= number_max})
        if numbers:
            cleaned.append(numbers)
    return cleaned


def normalize_scores(scores, number_max):
    full_scores = {number: float(scores.get(number, 0.0)) for number in range(1, number_max + 1)}
    values = list(full_scores.values())
    if not values:
        return full_scores
    low = min(values)
    high = max(values)
    if high == low:
        return {number: 0.0 for number in full_scores}
    return {number: (value - low) / (high - low) for number, value in full_scores.items()}


def count_consecutive_pairs(numbers):
    ordered = sorted(numbers)
    return sum(1 for left, right in zip(ordered, ordered[1:]) if right - left == 1)


def has_three_consecutive(numbers):
    ordered = sorted(set(int(number) for number in numbers))
    run = 1
    for left, right in zip(ordered, ordered[1:]):
        if right - left == 1:
            run += 1
            if run >= 3:
                return True
        else:
            run = 1
    return False


def anti_popular_overlap_limit(draw_size):
    return 4 if int(draw_size) <= 6 else 5


def anti_popular_candidate_status(numbers, previous_numbers, number_max, draw_size):
    selected = sorted(set(safe_int(number) for number in numbers if 1 <= safe_int(number) <= number_max))
    previous = set(clean_number_rows([previous_numbers], number_max)[0]) if previous_numbers else set()
    previous_overlap = len(set(selected) & previous)
    overlap_limit = anti_popular_overlap_limit(draw_size)
    has_over_31 = any(number > 31 for number in selected)
    has_three_run = has_three_consecutive(selected)
    valid = (
        len(selected) == int(draw_size)
        and previous_overlap >= 1
        and previous_overlap < overlap_limit
        and has_over_31
        and not has_three_run
    )
    return {
        "valid": valid,
        "previous_overlap_count": previous_overlap,
        "previous_overlap_limit": overlap_limit - 1,
        "has_over_31": has_over_31,
        "three_consecutive_excluded": not has_three_run,
        "three_consecutive_found": has_three_run,
    }


def anti_popular_model_description():
    return (
        "当選確率そのものを上げる目的ではなく、当選時に他人と数字が被りにくく、"
        "賞金分配リスクを下げる可能性を研究する補助モデル。"
    )


def anti_popular_reason(numbers, status):
    return (
        f"{ANTI_POPULAR_EXPECTED_VALUE_LABEL}: {anti_popular_model_description()}"
        f"前回数字との重複{status['previous_overlap_count']}個、"
        f"31超え数字{'あり' if status['has_over_31'] else 'なし'}、"
        f"3連続チェック{'OK' if status['three_consecutive_excluded'] else '除外対象'}。"
    )


def _anti_popular_candidate_score(numbers, previous_numbers, rows, number_max, draw_size):
    status = anti_popular_candidate_status(numbers, previous_numbers, number_max, draw_size)
    if not status["valid"]:
        return None
    target_sum, sum_std = _history_sum_stats(rows, draw_size, number_max, precleaned=True) if rows else (sum(numbers), max(draw_size * 4, 1))
    target_sum = _safe_float(target_sum, sum(numbers))
    sum_std = max(_safe_float(sum_std, draw_size * 4), 1.0)
    over_31_count = sum(number > 31 for number in numbers)
    birthday_count = sum(number <= 31 for number in numbers)
    tail_penalty = sum(max(0, count - 1) for count in Counter(number % 10 for number in numbers).values()) * 4
    consecutive_penalty = count_consecutive_pairs(numbers) * 5
    overlap = status["previous_overlap_count"]
    preferred_overlap = 1 if draw_size <= 6 else 2
    overlap_penalty = abs(overlap - preferred_overlap) * 9
    sum_penalty = abs(sum(numbers) - target_sum) / sum_std * 6
    return over_31_count * 18 + max(0, draw_size - birthday_count) * 3 - tail_penalty - consecutive_penalty - overlap_penalty - sum_penalty


def generate_anti_popular_expected_value_picks(
    number_rows,
    number_max,
    draw_size,
    target_round=0,
    pick_count=1,
    anchor_numbers=None,
    max_anchor_overlap=None,
    attempts=5000,
):
    rows = clean_number_rows(number_rows, number_max)
    if not rows:
        return []
    previous_numbers = rows[-1]
    previous_set = set(previous_numbers)
    seed = int(target_round or 0) * 1009 + int(number_max) * 37 + int(draw_size) * 101
    rng = random.Random(seed)
    all_numbers = list(range(1, number_max + 1))
    anchor_set = set(anchor_numbers or [])

    candidates = {}
    for _ in range(attempts):
        forced = rng.sample(sorted(previous_set), min(rng.choice([1, 2]), len(previous_set)))
        remaining_pool = [number for number in all_numbers if number not in forced]
        sampled = forced + rng.sample(remaining_pool, draw_size - len(forced))
        numbers = tuple(sorted(sampled))
        if max_anchor_overlap is not None and anchor_set and len(set(numbers) & anchor_set) > int(max_anchor_overlap):
            continue
        score = _anti_popular_candidate_score(numbers, previous_numbers, rows, number_max, draw_size)
        if score is None:
            continue
        candidates[numbers] = max(score, candidates.get(numbers, float("-inf")))

    if not candidates:
        high_numbers = [number for number in range(32, number_max + 1)]
        base = sorted(rng.sample(sorted(previous_set), 1))
        fill_pool = [number for number in high_numbers + all_numbers if number not in base]
        for number in fill_pool:
            if len(base) >= draw_size:
                break
            base.append(number)
        numbers = tuple(sorted(base[:draw_size]))
        status = anti_popular_candidate_status(numbers, previous_numbers, number_max, draw_size)
        if status["valid"]:
            candidates[numbers] = _anti_popular_candidate_score(numbers, previous_numbers, rows, number_max, draw_size) or 0.0

    picks = []
    for numbers, score in sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:pick_count]:
        status = anti_popular_candidate_status(numbers, previous_numbers, number_max, draw_size)
        picks.append(
            {
                "numbers": tuple(numbers),
                "reason": anti_popular_reason(numbers, status),
                "model_key": ANTI_POPULAR_EXPECTED_VALUE_KEY,
                "model_name": ANTI_POPULAR_EXPECTED_VALUE_LABEL,
                "used_models": [ANTI_POPULAR_EXPECTED_VALUE_LABEL],
                "emphasized_factors": "前回数字との1個以上の重複、31超え数字、3連続除外、過度な前回重複の除外",
                "selection_reason": anti_popular_reason(numbers, status),
                "anti_popular_diagnostics": status,
                "model_description": anti_popular_model_description(),
                "expected_value_note": "分配リスク低減を研究する補助指標であり、当選確率の上昇を保証しません。",
                "prediction_score": round(float(score), 3),
            }
        )
    return picks


SET_BALL_COLUMN_CANDIDATES = (
    "球セット",
    "セット球",
    "抽せん球",
    "抽せん球セット",
    "使用球セット",
    "逅・そ繝・ヨ",
)


def detect_set_ball_column(frame):
    if frame is None or frame.empty:
        return None
    for column in SET_BALL_COLUMN_CANDIDATES:
        if column in frame.columns:
            return column
    for column in frame.columns:
        text = str(column)
        if "セット" in text or "set" in text.lower():
            return column
    return None


def build_set_ball_analysis(results, number_columns, number_max, draw_size, set_ball_frame=None, round_column="開催回"):
    empty_number_frame = pd.DataFrame(columns=["数字", "同一セット出現数", "全体出現数", "セット球スコア"])
    empty_summary = pd.DataFrame(columns=["項目", "値"])
    if results is None or results.empty:
        return {
            "available": False,
            "message": "抽せん履歴がないため、セット球分析をスキップしました。",
            "set_ball_column": "",
            "latest_set_ball": "",
            "score_map": {number: 0.0 for number in range(1, number_max + 1)},
            "summary_frame": empty_summary,
            "number_frame": empty_number_frame,
        }

    frame = results.copy()
    if round_column not in frame.columns:
        round_column = frame.columns[0] if len(frame.columns) else "開催回"
    for column in number_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[round_column] = pd.to_numeric(frame[round_column], errors="coerce")

    set_column = detect_set_ball_column(frame)
    if set_column is None and set_ball_frame is not None and not set_ball_frame.empty:
        set_frame = set_ball_frame.copy()
        set_round_column = round_column if round_column in set_frame.columns else set_frame.columns[0]
        set_column = detect_set_ball_column(set_frame)
        if set_column and set_round_column in set_frame.columns:
            set_frame[set_round_column] = pd.to_numeric(set_frame[set_round_column], errors="coerce")
            frame = frame.merge(
                set_frame[[set_round_column, set_column]].dropna(subset=[set_round_column]).drop_duplicates(set_round_column, keep="last"),
                left_on=round_column,
                right_on=set_round_column,
                how="left",
            )
            if set_round_column != round_column and set_round_column in frame.columns:
                frame = frame.drop(columns=[set_round_column])

    if set_column is None or set_column not in frame.columns:
        return {
            "available": False,
            "message": "セット球データなし。列が追加された場合のみChamini6の補助スコアに反映します。",
            "set_ball_column": "",
            "latest_set_ball": "",
            "score_map": {number: 0.0 for number in range(1, number_max + 1)},
            "summary_frame": empty_summary,
            "number_frame": empty_number_frame,
        }

    available_number_columns = [column for column in number_columns if column in frame.columns]
    if len(available_number_columns) < int(draw_size):
        return {
            "available": False,
            "message": "セット球分析に必要な数字列が不足しているため、補助スコアをスキップしました。",
            "set_ball_column": str(set_column),
            "latest_set_ball": "",
            "score_map": {number: 0.0 for number in range(1, number_max + 1)},
            "summary_frame": empty_summary,
            "number_frame": empty_number_frame,
        }

    valid = frame.dropna(subset=[round_column]).copy()
    valid[set_column] = valid[set_column].fillna("").astype(str).str.strip()
    valid = valid[valid[set_column] != ""]
    if valid.empty:
        return {
            "available": False,
            "message": "セット球列はありますが、有効なセット球値がありません。",
            "set_ball_column": str(set_column),
            "latest_set_ball": "",
            "score_map": {number: 0.0 for number in range(1, number_max + 1)},
            "summary_frame": empty_summary,
            "number_frame": empty_number_frame,
        }

    latest = valid.sort_values(round_column).tail(1).iloc[0]
    latest_set = str(latest.get(set_column, "")).strip()
    same_set = valid[valid[set_column] == latest_set].copy()
    all_rows = clean_number_rows(valid[available_number_columns].values.tolist(), number_max)
    same_rows = clean_number_rows(same_set[available_number_columns].values.tolist(), number_max)
    all_counter = Counter(number for row in all_rows for number in row)
    same_counter = Counter(number for row in same_rows for number in row)
    raw_scores = {}
    for number in range(1, number_max + 1):
        same_freq = same_counter.get(number, 0)
        all_freq = all_counter.get(number, 0)
        raw_scores[number] = same_freq * 1.5 + all_freq * 0.1
    score_map = normalize_scores(raw_scores, number_max)
    target_rows = same_rows or all_rows
    sums = [sum(row) for row in target_rows]
    odd_counts = [sum(number % 2 for number in row) for row in target_rows]
    high_counts = [sum(number > game_low_limit(number_max) for number in row) for row in target_rows]
    summary_rows = [
        {"項目": "最新セット球", "値": latest_set},
        {"項目": "同一セット履歴数", "値": int(len(same_rows))},
        {"項目": "平均合計値", "値": round(sum(sums) / max(len(sums), 1), 2) if sums else "-"},
        {"項目": "平均奇数数", "値": round(sum(odd_counts) / max(len(odd_counts), 1), 2) if odd_counts else "-"},
        {"項目": "平均高数字数", "値": round(sum(high_counts) / max(len(high_counts), 1), 2) if high_counts else "-"},
    ]
    number_rows = [
        {
            "数字": number,
            "同一セット出現数": int(same_counter.get(number, 0)),
            "全体出現数": int(all_counter.get(number, 0)),
            "セット球スコア": round(float(score_map.get(number, 0.0)) * 100, 2),
        }
        for number in range(1, number_max + 1)
    ]
    number_frame = pd.DataFrame(number_rows).sort_values(["セット球スコア", "同一セット出現数", "数字"], ascending=[False, False, True])
    return {
        "available": True,
        "message": f"セット球 {latest_set} の履歴をChamini6補助スコアに利用できます。",
        "set_ball_column": str(set_column),
        "latest_set_ball": latest_set,
        "score_map": score_map,
        "summary_frame": pd.DataFrame(summary_rows),
        "number_frame": number_frame,
    }


def _fixed_chamini6_component_weights():
    return {
        "frequency_analysis": 1.0,
        "cold_analysis": 0.9,
        "hot_analysis": 1.1,
        "odd_even_analysis": 0.55,
        "sum_value_analysis": 0.65,
        "high_low_analysis": 0.6,
        "consecutive_analysis": 0.5,
        "last_digit_analysis": 0.45,
        "pair_analysis": 0.95,
        "triple_analysis": 0.7,
        "bonus_analysis": 0.5,
        "markov_chain": 0.95,
        "bayesian_estimation": 0.9,
        "monte_carlo": 0.75,
        "machine_learning": 1.25,
        ANTI_POPULAR_EXPECTED_VALUE_KEY: 0.45,
    }


def _ai_weight_for_model(component_key, ai_weight_summary=None):
    summary = ai_weight_summary or {}
    weights = summary.get("model_weights", {}) or {}
    canonical = canonical_model_key(component_key)
    label = model_label(canonical)
    value = 0.0
    for key in (canonical, label, str(component_key)):
        if key in weights:
            value = _safe_float(weights.get(key), 0.0)
            break
    return _clamp(1.0 + value, 0.72, 1.32)


def build_chamini6_score_map(
    number_rows,
    bonus_rows,
    number_max,
    draw_size,
    target_round=0,
    reports=None,
    model_history=None,
    lottery_type="",
    ai_weight_summary=None,
    set_ball_analysis=None,
):
    rows = clean_number_rows(number_rows, number_max)
    bonus_rows = clean_number_rows(bonus_rows or [], number_max)
    combined = {number: 0.0 for number in range(1, number_max + 1)}
    fixed_weights = _fixed_chamini6_component_weights()
    performance_df = build_model_performance_weights(reports, draw_size, lottery_type, model_history)
    performance_weights = {
        str(row.get("model_key")): _safe_float(row.get("applied_weight"), 1.0)
        for _, row in performance_df.iterrows()
    }
    detail_rows = []
    for component_key in CHAMINI6_COMPONENT_MODEL_KEYS:
        raw_scores = build_model_scores(rows, component_key, number_max, draw_size, target_round, bonus_rows)
        normalized = normalize_scores(raw_scores, number_max)
        fixed = fixed_weights.get(component_key, 1.0)
        performance = performance_weights.get(component_key, 1.0)
        ai_weight = _ai_weight_for_model(component_key, ai_weight_summary)
        applied = _clamp(fixed * performance * ai_weight, 0.25, 2.4)
        for number, score in normalized.items():
            combined[number] += score * applied
        detail_rows.append(
            {
                "モデルキー": component_key,
                "モデル名": model_label(component_key),
                "固定重み": round(fixed, 3),
                "成績重み": round(performance, 3),
                "AI改善重み": round(ai_weight, 3),
                "適用重み": round(applied, 3),
                "状態": "反映",
            }
        )
    if set_ball_analysis and set_ball_analysis.get("available"):
        set_scores = normalize_scores(set_ball_analysis.get("score_map", {}), number_max)
        set_weight = 0.55
        for number, score in set_scores.items():
            combined[number] += score * set_weight
        detail_rows.append(
            {
                "モデルキー": "set_ball_analysis",
                "モデル名": "セット球分析",
                "固定重み": set_weight,
                "成績重み": 1.0,
                "AI改善重み": 1.0,
                "適用重み": set_weight,
                "状態": set_ball_analysis.get("message", "反映"),
            }
        )
    else:
        detail_rows.append(
            {
                "モデルキー": "set_ball_analysis",
                "モデル名": "セット球分析",
                "固定重み": 0.0,
                "成績重み": 1.0,
                "AI改善重み": 1.0,
                "適用重み": 0.0,
                "状態": (set_ball_analysis or {}).get("message", "セット球データなし"),
            }
        )
    return combined, pd.DataFrame(detail_rows)


def _chamini6_candidate_score(numbers, score_map, rows, number_max, draw_size, sum_stats=None):
    sum_stats = sum_stats or _history_sum_stats(rows, draw_size, number_max, precleaned=True)
    balance = _balance_score(numbers, rows, number_max, draw_size, sum_stats)
    raw = sum(_safe_float(score_map.get(number), 0.0) for number in numbers) / max(draw_size, 1) * 100
    over_31_bonus = 4 if any(number > 31 for number in numbers) else -8
    consecutive_penalty = 20 if has_three_consecutive(numbers) else count_consecutive_pairs(numbers) * 2
    return raw + balance * 0.2 + over_31_bonus - consecutive_penalty


def generate_chamini6_god_mode_picks(
    number_rows,
    bonus_rows,
    number_max,
    draw_size,
    target_round=0,
    reports=None,
    model_history=None,
    lottery_type="",
    ai_weight_summary=None,
    set_ball_analysis=None,
    pick_count=1,
):
    rows = clean_number_rows(number_rows, number_max)
    if not rows:
        return []
    score_map, detail_df = build_chamini6_score_map(
        rows,
        bonus_rows,
        number_max,
        draw_size,
        target_round,
        reports,
        model_history,
        lottery_type,
        ai_weight_summary,
        set_ball_analysis,
    )
    candidate_limit = min(number_max, max(draw_size + 9, 15 if draw_size <= 6 else 16))
    candidate_numbers = top_score_numbers(score_map, candidate_limit)
    if len(candidate_numbers) < draw_size:
        candidate_numbers = [number for number, _ in sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:candidate_limit]]
    low_limit = game_low_limit(number_max)
    sum_stats = _history_sum_stats(rows, draw_size, number_max, precleaned=True)
    candidates = []
    for combo in combinations(candidate_numbers, draw_size):
        numbers = tuple(sorted(combo))
        odd = sum(number % 2 for number in numbers)
        low = sum(number <= low_limit for number in numbers)
        if draw_size == 6 and (odd not in (2, 3, 4) or low not in (2, 3, 4)):
            continue
        if draw_size == 7 and (odd not in (3, 4) or low not in (3, 4)):
            continue
        if has_three_consecutive(numbers):
            continue
        score = _chamini6_candidate_score(numbers, score_map, rows, number_max, draw_size, sum_stats)
        candidates.append((score, numbers))
    if not candidates:
        fallback = tuple(sorted(candidate_numbers[:draw_size]))
        candidates.append((_chamini6_candidate_score(fallback, score_map, rows, number_max, draw_size, sum_stats), fallback))
    candidates.sort(reverse=True, key=lambda item: item[0])
    picks = []
    used = set()
    for score, numbers in candidates:
        if picks and len(set(numbers) & used) > max(2, draw_size // 2):
            continue
        set_text = "セット球分析を補助反映" if set_ball_analysis and set_ball_analysis.get("available") else "セット球データなし"
        ai_text = "AI改善重みを反映" if (ai_weight_summary or {}).get("available") else "AI改善重みは通常値"
        anti_status = anti_popular_candidate_status(numbers, rows[-1], number_max, draw_size)
        reason = (
            f"{CHAMINI6_GOD_MODE_LABEL}: 16分析エンジン、AI改善重み、"
            f"人と被りにくい期待値最大化モデルを統合。{ai_text} / {set_text}。"
        )
        picks.append(
            {
                "numbers": tuple(numbers),
                "reason": f"[{CHAMINI6_GOD_MODE_KEY}] {reason}",
                "model_key": CHAMINI6_GOD_MODE_KEY,
                "model_name": CHAMINI6_GOD_MODE_LABEL,
                "display_model_name": CHAMINI6_GOD_MODE_LABEL,
                "used_models": [model_label(key) for key in CHAMINI6_COMPONENT_MODEL_KEYS],
                "emphasized_factors": "総合統合、AI改善重み、セット球補助、低人気期待値、奇数偶数・高低・合計値バランス",
                "selection_reason": reason,
                "prediction_score": round(float(score), 3),
                "chamini6_detail": detail_df,
                "anti_popular_diagnostics": anti_status,
                "model_description": "複数モデルを削除・置換せず、研究用の統合候補として重み付けする独立エンジンです。",
            }
        )
        used.update(numbers)
        if len(picks) >= pick_count:
            break
    return picks


def _ranked_replacement_pool(number_max, score_map=None):
    score_map = score_map or {}
    return [
        number
        for number, _ in sorted(
            ((number, _safe_float(score_map.get(number), 0.0)) for number in range(1, number_max + 1)),
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )
    ]


def _limit_anchor_overlap(numbers, anchor_numbers, limit, number_max, score_map=None, core_count=2):
    selected = set(int(number) for number in numbers)
    anchor = set(int(number) for number in anchor_numbers or [])
    if not anchor:
        return tuple(sorted(selected))
    score_map = score_map or {}
    overlap = selected & anchor
    protected_count = min(max(core_count, 0), len(overlap), max(limit, 0))
    protected = set(sorted(overlap, key=lambda number: _safe_float(score_map.get(number), 0.0), reverse=True)[:protected_count])
    pool = _ranked_replacement_pool(number_max, score_map)
    while len(selected & anchor) > limit:
        removable = sorted(
            (number for number in selected & anchor if number not in protected),
            key=lambda number: _safe_float(score_map.get(number), 0.0),
        )
        if not removable:
            break
        remove_number = removable[0]
        replacement = None
        for candidate in pool:
            if candidate in selected or candidate in anchor:
                continue
            replacement = candidate
            break
        if replacement is None:
            break
        selected.remove(remove_number)
        selected.add(replacement)
    return tuple(sorted(selected))


def _pattern_overlap_summary(picks):
    labels = [pick.get("pattern_key", chr(65 + index)) for index, pick in enumerate(picks)]
    counts = {}
    for left_index, left in enumerate(picks):
        for right_index in range(left_index + 1, len(picks)):
            right = picks[right_index]
            key = f"{labels[left_index]}-{labels[right_index]}"
            counts[key] = len(set(left.get("numbers", [])) & set(right.get("numbers", [])))
    summary = " / ".join(f"{key}: {value}個" for key, value in counts.items()) if counts else "-"
    for pick in picks:
        pick["overlap_counts"] = counts
        pick["overlap_summary"] = summary
    return picks


def apply_prediction_pattern_roles(picks, draw_size, number_max, score_map=None, base_model_name="", anti_popular_pick=None):
    normalized = []
    for pick in picks or []:
        numbers = tuple(sorted(int(number) for number in pick.get("numbers", []) if 1 <= int(number) <= number_max))
        if len(numbers) != draw_size:
            continue
        item = dict(pick)
        item["numbers"] = numbers
        normalized.append(item)

    if anti_popular_pick:
        anti_item = dict(anti_popular_pick)
        anti_item["numbers"] = tuple(sorted(int(number) for number in anti_item.get("numbers", [])))
        if len(anti_item["numbers"]) == draw_size:
            if len(normalized) >= 3:
                normalized[2] = anti_item
            else:
                normalized.append(anti_item)

    normalized = normalized[:3]
    if not normalized:
        return []

    anchor = normalized[0]["numbers"]
    if len(normalized) >= 2:
        normalized[1]["numbers"] = _limit_anchor_overlap(normalized[1]["numbers"], anchor, 3, number_max, score_map, core_count=2)
    if len(normalized) >= 3:
        normalized[2]["numbers"] = _limit_anchor_overlap(normalized[2]["numbers"], anchor, 2, number_max, score_map, core_count=2)

    for index, pick in enumerate(normalized):
        role = PREDICTION_PATTERN_ROLES[min(index, len(PREDICTION_PATTERN_ROLES) - 1)]
        pick["pattern_key"] = role["pattern_key"]
        pick["pattern_label"] = role["pattern_label"]
        pick["role_label"] = role["role_label"]
        pick.setdefault("model_name", base_model_name or "既存分析モデル")
        pick.setdefault("used_models", [pick.get("model_name") or base_model_name or "既存分析モデル"])
        pick.setdefault("emphasized_factors", role["emphasized_factors"])
        pick.setdefault("selection_reason", role["selection_reason"])
        pick["reason"] = (
            f"{role['pattern_label']} {role['role_label']}。"
            f"使用モデル: {' / '.join(str(model) for model in pick.get('used_models', []))}。"
            f"重視した要素: {pick.get('emphasized_factors', role['emphasized_factors'])}。"
            f"{pick.get('reason', role['selection_reason'])}"
        )
    return _pattern_overlap_summary(normalized)


def anti_popular_verification_fields(predicted, previous_numbers, number_max, draw_size):
    status = anti_popular_candidate_status(predicted, previous_numbers, number_max, draw_size)
    return {
        "前回数字との重複数": status["previous_overlap_count"],
        "31超え数字の有無": "あり" if status["has_over_31"] else "なし",
        "3連続除外チェック": "OK" if status["three_consecutive_excluded"] else "3連続あり",
        "改善メモ": anti_popular_model_description(),
    }


def last_seen_gaps(number_rows, number_max):
    last_seen = {number: -1 for number in range(1, number_max + 1)}
    for index, row in enumerate(number_rows):
        for number in row:
            last_seen[number] = index
    draw_count = len(number_rows)
    return {
        number: draw_count if last_seen[number] < 0 else draw_count - last_seen[number]
        for number in range(1, number_max + 1)
    }


def frequency_scores(number_rows, number_max):
    counts = Counter(number for row in number_rows for number in row)
    return {number: float(counts[number]) for number in range(1, number_max + 1)}


def recent_scores(number_rows, number_max, window=30):
    rows = number_rows[-window:]
    scores = {number: 0.0 for number in range(1, number_max + 1)}
    for offset, row in enumerate(rows, start=1):
        weight = 0.5 + offset / max(len(rows), 1)
        for number in row:
            scores[number] += weight
    return scores


def cold_scores(number_rows, number_max):
    counts = frequency_scores(number_rows, number_max)
    gaps = last_seen_gaps(number_rows, number_max)
    return {number: gaps[number] + counts[number] * 0.15 for number in range(1, number_max + 1)}


def high_band_for_game(number_max):
    return range(32, number_max + 1) if number_max >= 43 else range(32, 38)


def co_occurrence_scores(number_rows, number_max, anchor_count=8):
    scores = {number: 0.0 for number in range(1, number_max + 1)}
    hot_numbers = [number for number, _ in Counter(number for row in number_rows[-50:] for number in row).most_common(anchor_count)]
    for row in number_rows:
        overlap = len(set(row) & set(hot_numbers))
        if overlap:
            for number in row:
                scores[number] += overlap
    return scores


def markov_scores(number_rows, number_max):
    scores = {number: 0.0 for number in range(1, number_max + 1)}
    if len(number_rows) < 2:
        return scores
    latest = set(number_rows[-1])
    for previous, current in zip(number_rows, number_rows[1:]):
        overlap = len(set(previous) & latest)
        if overlap:
            for number in current:
                scores[number] += overlap
    return scores


def monte_carlo_scores(number_rows, number_max, draw_size, target_round):
    base = normalize_scores(frequency_scores(number_rows, number_max), number_max)
    recent = normalize_scores(recent_scores(number_rows, number_max), number_max)
    cold = normalize_scores(cold_scores(number_rows, number_max), number_max)
    weights = {number: base[number] * 4 + recent[number] * 3 + cold[number] for number in range(1, number_max + 1)}
    scores = {number: 0.0 for number in range(1, number_max + 1)}
    rng = random.Random(int(target_round or 0) + number_max * 100 + draw_size)
    numbers = list(range(1, number_max + 1))
    for _ in range(240):
        pool = numbers[:]
        pick = []
        for _ in range(draw_size):
            total = sum(max(weights[number], 0.01) for number in pool)
            point = rng.random() * total
            upto = 0.0
            chosen = pool[-1]
            for number in pool:
                upto += max(weights[number], 0.01)
                if upto >= point:
                    chosen = number
                    break
            pick.append(chosen)
            pool.remove(chosen)
        for number in pick:
            scores[number] += 1.0
    return scores


def model_score_components(number_rows, number_max, draw_size, target_round, bonus_rows=None):
    bonus_rows = clean_number_rows(bonus_rows or [], number_max)
    freq = normalize_scores(frequency_scores(number_rows, number_max), number_max)
    recent = normalize_scores(recent_scores(number_rows, number_max), number_max)
    cold = normalize_scores(cold_scores(number_rows, number_max), number_max)
    pair = normalize_scores(co_occurrence_scores(number_rows, number_max, 8), number_max)
    markov = normalize_scores(markov_scores(number_rows, number_max), number_max)
    bonus = normalize_scores(frequency_scores(bonus_rows, number_max), number_max) if bonus_rows else freq
    return {
        "freq": freq,
        "recent": recent,
        "cold": cold,
        "pair": pair,
        "markov": markov,
        "bonus": bonus,
    }


def build_model_scores(number_rows, model_key, number_max, draw_size, target_round=0, bonus_rows=None):
    key = canonical_model_key(model_key)
    rows = clean_number_rows(number_rows, number_max)
    if key == "random_baseline":
        rng = random.Random(int(target_round or 0) + number_max)
        return {number: rng.random() for number in range(1, number_max + 1)}
    if not rows:
        return {number: 0.0 for number in range(1, number_max + 1)}
    if key == ANTI_POPULAR_EXPECTED_VALUE_KEY:
        latest = set(rows[-1])
        rng = random.Random(int(target_round or 0) * 1009 + number_max * 37 + draw_size)
        return {
            number: (1.2 if number in latest else 0.0)
            + (1.0 if number > 31 else 0.0)
            + (0.25 if number % 10 not in (0, 5, 7) else 0.0)
            + rng.random() * 0.35
            for number in range(1, number_max + 1)
        }
    if key == CHAMINI6_GOD_MODE_KEY:
        combined = {number: 0.0 for number in range(1, number_max + 1)}
        fixed_weights = {
            "frequency_analysis": 1.0,
            "cold_analysis": 0.9,
            "hot_analysis": 1.1,
            "odd_even_analysis": 0.55,
            "sum_value_analysis": 0.65,
            "high_low_analysis": 0.6,
            "consecutive_analysis": 0.5,
            "last_digit_analysis": 0.45,
            "pair_analysis": 0.95,
            "triple_analysis": 0.7,
            "bonus_analysis": 0.5,
            "markov_chain": 0.95,
            "bayesian_estimation": 0.9,
            "monte_carlo": 0.75,
            "machine_learning": 1.25,
            ANTI_POPULAR_EXPECTED_VALUE_KEY: 0.45,
        }
        for component_key in CHAMINI6_COMPONENT_MODEL_KEYS:
            raw_scores = build_model_scores(rows, component_key, number_max, draw_size, target_round, bonus_rows)
            normalized = normalize_scores(raw_scores, number_max)
            weight = fixed_weights.get(component_key, 1.0)
            for number, score in normalized.items():
                combined[number] += score * weight
        return combined

    components = model_score_components(rows, number_max, draw_size, target_round, bonus_rows)
    counts = Counter(number for row in rows for number in row)
    gaps = last_seen_gaps(rows, number_max)
    high_band = set(high_band_for_game(number_max))
    recent_rows = rows[-20:]
    recent_odd_avg = sum(sum(number % 2 for number in row) for row in recent_rows) / max(len(recent_rows), 1)
    recent_high_avg = sum(sum(number in high_band for number in row) for row in recent_rows) / max(len(recent_rows), 1)
    target_sum = sum(sum(row) for row in rows) / max(len(rows), 1)
    target_slot = target_sum / max(draw_size, 1)

    if key == "frequency_analysis":
        return frequency_scores(rows, number_max)
    if key == "hot_analysis":
        return recent_scores(rows, number_max)
    if key == "cold_analysis":
        return cold_scores(rows, number_max)
    if key == "odd_even_analysis":
        prefer_odd = recent_odd_avg >= draw_size / 2
        return {number: (1.0 if number % 2 == int(prefer_odd) else 0.45) + components["freq"][number] for number in range(1, number_max + 1)}
    if key == "sum_value_analysis":
        return {
            number: max(0.0, 1.0 - abs(number - target_slot) / max(number_max / 2, 1)) + components["freq"][number] * 0.5
            for number in range(1, number_max + 1)
        }
    if key == "high_low_analysis":
        prefer_high = recent_high_avg >= max(draw_size / 3, 1)
        return {number: (1.0 if (number in high_band) == prefer_high else 0.35) + components["recent"][number] for number in range(1, number_max + 1)}
    if key == "consecutive_analysis":
        hot = {number for number, _ in counts.most_common(10)}
        return {
            number: sum(1 for near in (number - 1, number + 1) if near in hot) + components["recent"][number]
            for number in range(1, number_max + 1)
        }
    if key == "last_digit_analysis":
        digit_counts = Counter(number % 10 for row in rows[-50:] for number in row)
        return {number: digit_counts[number % 10] + components["freq"][number] for number in range(1, number_max + 1)}
    if key == "pair_analysis":
        return co_occurrence_scores(rows, number_max, 8)
    if key == "triple_analysis":
        return co_occurrence_scores(rows, number_max, 12)
    if key == "bonus_analysis":
        return {number: components["bonus"][number] * 2 + components["freq"][number] for number in range(1, number_max + 1)}
    if key == "markov_chain":
        return markov_scores(rows, number_max)
    if key == "bayesian_estimation":
        total_draws = len(rows)
        return {number: (counts[number] + 1) / (total_draws + number_max) + components["recent"][number] for number in range(1, number_max + 1)}
    if key == "monte_carlo":
        return monte_carlo_scores(rows, number_max, draw_size, target_round)
    if key == "machine_learning":
        return {
            number: components["freq"][number] * 35
            + components["recent"][number] * 25
            + components["cold"][number] * 15
            + components["pair"][number] * 15
            + components["bonus"][number] * 10
            for number in range(1, number_max + 1)
        }
    if key == "ai_improved":
        return {
            number: components["freq"][number] * 22
            + components["recent"][number] * 24
            + components["cold"][number] * 12
            + components["pair"][number] * 12
            + components["markov"][number] * 12
            + components["bonus"][number] * 8
            + (10 if number in high_band else 0)
            for number in range(1, number_max + 1)
        }
    if key == "non_overlap":
        fixed = {4, 9, 13}
        digit_counts = Counter(number % 10 for row in rows[-100:] for number in row)
        return {
            number: components["cold"][number] * 18
            + (14 if number in high_band else 0)
            + (8 if number in fixed else 0)
            + max(0, 8 - digit_counts[number % 10])
            for number in range(1, number_max + 1)
        }
    if key == "fixed_numbers":
        fixed = {4, 9, 13}
        return {number: components["freq"][number] + (10 if number in fixed else 0) for number in range(1, number_max + 1)}
    if key == "variable_numbers":
        return {number: components["cold"][number] * 12 + components["recent"][number] * 8 for number in range(1, number_max + 1)}
    if key == "hybrid":
        return {
            number: components["freq"][number] * 16
            + components["recent"][number] * 16
            + components["cold"][number] * 12
            + components["pair"][number] * 12
            + components["markov"][number] * 10
            + components["recent"][number] * 10
            + components["bonus"][number] * 8
            for number in range(1, number_max + 1)
        }
    if key in ("hot10", "hot20", "hot50"):
        limit = int(key.replace("hot", ""))
        hot_items = Counter(number for row in rows[-limit:] for number in row)
        return {number: float(hot_items[number]) for number in range(1, number_max + 1)}
    if key == "cold_revival":
        return {number: gaps[number] * 1.4 + (1.0 / max(counts[number], 1)) for number in range(1, number_max + 1)}
    if key == "bonus_promotion":
        return {number: components["bonus"][number] * 20 + components["recent"][number] * 6 for number in range(1, number_max + 1)}

    return frequency_scores(rows, number_max)


def build_model_support_map(predicted, number_rows, number_max, draw_size, target_round=0, bonus_rows=None, selected_model=None):
    support_map = {int(number): [] for number in predicted}
    candidate_window = max(10, draw_size * 2)
    for model_key, label in ARL_MODEL_LABELS.items():
        scores = build_model_scores(number_rows, model_key, number_max, draw_size, target_round, bonus_rows)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:candidate_window]
        top_numbers = {number for number, value in ranked if value > 0}
        for number in support_map:
            if number in top_numbers:
                support_map[number].append(label)
    if selected_model:
        label = model_label(selected_model)
        for number in support_map:
            if label not in support_map[number]:
                support_map[number].append(label)
    return support_map


def json_text(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def parse_json_text(value, fallback=None):
    if fallback is None:
        fallback = {}
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass
    try:
        return json.loads(str(value))
    except Exception:
        return fallback


def game_low_limit(number_max):
    return 21 if number_max >= 43 else 18


def game_balance(numbers, number_max):
    low_limit = game_low_limit(number_max)
    tails = Counter(number % 10 for number in numbers)
    return {
        "odd": sum(number % 2 for number in numbers),
        "even": len(numbers) - sum(number % 2 for number in numbers),
        "low": sum(number <= low_limit for number in numbers),
        "high": sum(number > low_limit for number in numbers),
        "sum": sum(numbers),
        "consecutive": count_consecutive_pairs(numbers),
        "same_tail": sum(1 for count in tails.values() if count >= 2),
    }


def rank_map_from_scores(scores):
    ranked = sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)
    return {number: index + 1 for index, (number, _) in enumerate(ranked)}


def top_score_numbers(scores, limit):
    ranked = sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)
    return sorted(number for number, value in ranked[:limit] if value > 0)


def summarize_condition_shift(predicted, actual, number_max):
    predicted_summary = game_balance(predicted, number_max)
    actual_summary = game_balance(actual, number_max)
    parts = []
    for key, label in [
        ("odd", "奇数"),
        ("low", "低数字"),
        ("high", "高数字"),
        ("consecutive", "連番"),
        ("same_tail", "同じ下一桁"),
    ]:
        diff = actual_summary[key] - predicted_summary[key]
        if diff > 0:
            parts.append(f"{label}を{diff}枠増やす条件")
        elif diff < 0:
            parts.append(f"{label}を{abs(diff)}枠減らす条件")
    sum_diff = actual_summary["sum"] - predicted_summary["sum"]
    if abs(sum_diff) >= 10:
        direction = "上げる" if sum_diff > 0 else "下げる"
        parts.append(f"合計値を{abs(sum_diff)}程度{direction}条件")
    return " / ".join(parts) if parts else "構造条件は近く、候補数字の入れ替え幅を検証"


def describe_sum_band(actual, number_rows):
    if not number_rows:
        return "履歴不足のため合計値帯は未判定"
    sums = [sum(row) for row in number_rows if row]
    if not sums:
        return "履歴不足のため合計値帯は未判定"
    average = sum(sums) / len(sums)
    if len(sums) > 1:
        variance = sum((value - average) ** 2 for value in sums) / (len(sums) - 1)
        deviation = variance ** 0.5
    else:
        deviation = max(10.0, average * 0.1)
    actual_sum = sum(actual)
    if average - deviation <= actual_sum <= average + deviation:
        return f"過去平均帯内({actual_sum}, 平均{average:.1f})"
    direction = "高い" if actual_sum > average else "低い"
    return f"過去平均より{direction}合計({actual_sum}, 平均{average:.1f})"


def build_number_feature_profiles(predicted, actual, number_rows, bonus_rows, number_max, draw_size, target_round=0):
    rows = clean_number_rows(number_rows, number_max)
    bonus_rows = clean_number_rows(bonus_rows or [], number_max)
    candidate_window = max(10, draw_size * 2)
    counts = frequency_scores(rows, number_max)
    recent = recent_scores(rows, number_max)
    cold = cold_scores(rows, number_max)
    gaps = last_seen_gaps(rows, number_max) if rows else {number: 0 for number in range(1, number_max + 1)}
    bonus_counts = frequency_scores(bonus_rows, number_max) if bonus_rows else {number: 0.0 for number in range(1, number_max + 1)}
    freq_rank = rank_map_from_scores(counts)
    recent_rank = rank_map_from_scores(recent)
    gap_rank = rank_map_from_scores(gaps)
    bonus_rank = rank_map_from_scores(bonus_counts)
    pair_candidates = set(top_score_numbers(build_model_scores(rows, "pair_analysis", number_max, draw_size, target_round, bonus_rows), candidate_window))
    triple_candidates = set(top_score_numbers(build_model_scores(rows, "triple_analysis", number_max, draw_size, target_round, bonus_rows), candidate_window))
    bonus_candidates = set(top_score_numbers(build_model_scores(rows, "bonus_analysis", number_max, draw_size, target_round, bonus_rows), candidate_window))
    support_map = build_model_support_map(actual, rows, number_max, draw_size, target_round, bonus_rows)
    previous_numbers = set(rows[-1]) if rows else set()
    actual_set = set(actual)
    predicted_set = set(predicted)
    actual_tails = Counter(number % 10 for number in actual)
    sum_band = describe_sum_band(actual, rows)
    low_limit = game_low_limit(number_max)

    profiles = []
    for number in sorted(actual):
        not_seen_count = max(gaps.get(number, 0) - 1, 0) if rows else 0
        profile = {
            "数字": f"{number:02d}",
            "予測に含まれていたか": "はい" if number in predicted_set else "いいえ",
            "一致数改善に必要だったか": "はい" if number not in predicted_set else "既に選択",
            "出現回数": int(counts.get(number, 0)),
            "出現頻度順位": int(freq_rank.get(number, 0)),
            "出現頻度上位": freq_rank.get(number, number_max) <= candidate_window,
            "ホットナンバー": recent_rank.get(number, number_max) <= candidate_window and recent.get(number, 0) > 0,
            "コールドナンバー": gap_rank.get(number, number_max) <= candidate_window and not_seen_count >= 2,
            "直近未出現回数": int(not_seen_count),
            "前回からの継続数字": number in previous_numbers,
            "連番条件": "当選番号内で連番" if any(abs(number - other) == 1 for other in actual_set if other != number) else "単独数字",
            "下一桁条件": "同じ下一桁あり" if actual_tails[number % 10] >= 2 else "下一桁の重複なし",
            "高低条件": "低数字" if number <= low_limit else "高数字",
            "奇数偶数条件": "奇数" if number % 2 else "偶数",
            "合計値条件": sum_band,
            "ペア分析候補": number in pair_candidates,
            "トリプル分析候補": number in triple_candidates,
            "ボーナス傾向候補": number in bonus_candidates or bonus_counts.get(number, 0) > 0,
            "ボーナス出現回数": int(bonus_counts.get(number, 0)),
            "ボーナス頻度順位": int(bonus_rank.get(number, 0)),
            "候補入りモデル": support_map.get(number, []),
        }
        profiles.append(profile)
    return profiles


def model_candidate_numbers(scores, draw_size):
    ranked = sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)
    return sorted(number for number, _ in ranked[:draw_size])


def build_model_needed_condition(model_name, model_numbers, actual, number_max):
    include_numbers = sorted(set(actual) - set(model_numbers))
    exclude_numbers = sorted(set(model_numbers) - set(actual))
    parts = []
    if include_numbers:
        parts.append(f"追加候補 {numbers_to_text(include_numbers)} を上位に残す")
    if exclude_numbers:
        parts.append(f"除外候補 {numbers_to_text(exclude_numbers)} の重みを抑える")
    parts.append(summarize_condition_shift(model_numbers, actual, number_max))
    return f"{model_name}: " + " / ".join(parts)


def build_model_improvement_rows(lottery_type, draw_no, prediction_id, actual, number_rows, bonus_rows, number_max, draw_size, target_round, created_at):
    rows = clean_number_rows(number_rows, number_max)
    bonus_rows = clean_number_rows(bonus_rows or [], number_max)
    score_cache = {}
    model_rows = []
    actual_set = set(actual)
    for model_key, model_name in ARL_MODEL_LABELS.items():
        scores = build_model_scores(rows, model_key, number_max, draw_size, target_round, bonus_rows)
        score_cache[model_key] = scores
        model_numbers = model_candidate_numbers(scores, draw_size)
        matched = sorted(set(model_numbers) & actual_set)
        missed = sorted(set(model_numbers) - actual_set)
        should_include = sorted(actual_set - set(model_numbers))
        should_exclude = missed
        needed_conditions = build_model_needed_condition(model_name, model_numbers, actual, number_max)
        hypothesis = (
            f"{model_name}では、{numbers_to_text(should_include) if should_include else '追加不要'}を候補に残す条件と、"
            f"{numbers_to_text(should_exclude) if should_exclude else '除外不要'}の過採用抑制を次回検証する"
        )
        model_rows.append(
            {
                "lottery_type": lottery_type,
                "draw_no": int(draw_no),
                "prediction_id": str(prediction_id),
                "model_key": model_key,
                "model_name": model_name,
                "predicted_numbers": numbers_to_text(model_numbers),
                "actual_numbers": numbers_to_text(actual),
                "matched_numbers": numbers_to_text(matched) if matched else "",
                "missed_numbers": numbers_to_text(missed) if missed else "",
                "should_have_included_numbers": numbers_to_text(should_include) if should_include else "",
                "should_have_excluded_numbers": numbers_to_text(should_exclude) if should_exclude else "",
                "matched_count": len(matched),
                "needed_conditions": needed_conditions,
                "next_hypothesis": hypothesis,
                "created_at": created_at,
            }
        )
    return model_rows, score_cache


def build_best_ensemble_analysis(score_cache, actual, number_max, draw_size):
    if not score_cache:
        return {}
    actual_set = set(actual)
    normalized = {model_key: normalize_scores(scores, number_max) for model_key, scores in score_cache.items()}
    best = None
    model_keys = list(normalized.keys())
    for size in (2, 3):
        for model_combo in combinations(model_keys, size):
            combined = {
                number: sum(normalized[model_key].get(number, 0.0) for model_key in model_combo)
                for number in range(1, number_max + 1)
            }
            candidates = model_candidate_numbers(combined, draw_size)
            matched = sorted(set(candidates) & actual_set)
            score = (len(matched), -len(set(candidates) - actual_set), sum(combined[number] for number in candidates))
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "models": [ARL_MODEL_LABELS.get(model_key, model_key) for model_key in model_combo],
                    "candidate_numbers": candidates,
                    "matched_numbers": matched,
                }
    if not best:
        return {}
    candidate_set = set(best["candidate_numbers"])
    return {
        "組み合わせ": best["models"],
        "候補数字": numbers_to_text(best["candidate_numbers"]),
        "一致数": len(best["matched_numbers"]),
        "一致数字": numbers_to_text(best["matched_numbers"]) if best["matched_numbers"] else "",
        "残すべき候補数字": numbers_to_text(best["matched_numbers"]) if best["matched_numbers"] else "",
        "切るべき候補数字": numbers_to_text(sorted(candidate_set - actual_set)) if candidate_set - actual_set else "",
        "追加すべき候補数字": numbers_to_text(sorted(actual_set - candidate_set)) if actual_set - candidate_set else "",
    }


def build_posthoc_optimization(model_rows, score_cache, predicted, actual, number_max, draw_size):
    current_count = len(set(predicted) & set(actual))
    ranked = sorted(model_rows, key=lambda row: (int(row["matched_count"]), -len(parse_numbers(row["should_have_excluded_numbers"], number_max))), reverse=True)
    max_count = int(ranked[0]["matched_count"]) if ranked else 0
    min_count = int(ranked[-1]["matched_count"]) if ranked else 0
    useful = [row["model_name"] for row in ranked if int(row["matched_count"]) == max_count][:5]
    weak = [row["model_name"] for row in reversed(ranked) if int(row["matched_count"]) == min_count][:5]
    weight_up = [row["model_name"] for row in ranked if int(row["matched_count"]) > current_count][:5] or useful[:3]
    weight_down = [row["model_name"] for row in reversed(ranked) if int(row["matched_count"]) < current_count][:5] or weak[:3]
    ensemble = build_best_ensemble_analysis(score_cache, actual, number_max, draw_size)
    return {
        "useful_models": useful,
        "weak_models": weak,
        "weight_up_models": weight_up,
        "weight_down_models": weight_down,
        "keep_numbers": numbers_to_text(sorted(set(predicted) & set(actual))) if set(predicted) & set(actual) else "",
        "cut_numbers": numbers_to_text(sorted(set(predicted) - set(actual))) if set(predicted) - set(actual) else "",
        "include_numbers": numbers_to_text(sorted(set(actual) - set(predicted))) if set(actual) - set(predicted) else "",
        "ensemble_analysis": ensemble,
    }


def build_winning_condition_report(
    lottery_type,
    draw_no,
    prediction_id,
    prediction_date,
    predicted,
    actual,
    bonus_numbers,
    number_rows,
    bonus_rows,
    number_max,
    draw_size,
    selected_model="",
    failure_reason="",
    created_at=None,
):
    created_at = created_at or datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    predicted = parse_numbers(numbers_to_text(predicted), number_max)
    actual = parse_numbers(numbers_to_text(actual), number_max)
    bonus_numbers = parse_numbers(numbers_to_text(bonus_numbers), number_max)
    matched = sorted(set(predicted) & set(actual))
    missed = sorted(set(predicted) - set(actual))
    should_include = sorted(set(actual) - set(predicted))
    should_exclude = missed
    model_rows, score_cache = build_model_improvement_rows(
        lottery_type,
        draw_no,
        prediction_id,
        actual,
        number_rows,
        bonus_rows,
        number_max,
        draw_size,
        draw_no,
        created_at,
    )
    optimization = build_posthoc_optimization(model_rows, score_cache, predicted, actual, number_max, draw_size)
    number_profiles = build_number_feature_profiles(predicted, actual, number_rows, bonus_rows, number_max, draw_size, draw_no)
    winning_conditions = {
        "選択できていれば一致数が増えた数字": numbers_to_text(should_include) if should_include else "",
        "除外すべきだった数字": numbers_to_text(should_exclude) if should_exclude else "",
        "数字別特徴": number_profiles,
        "必要だった条件": summarize_condition_shift(predicted, actual, number_max),
        "ボーナス数字": numbers_to_text(bonus_numbers) if bonus_numbers else "",
        "後追い最適化": optimization,
        "選択モデル": selected_model,
    }
    model_analysis = [
        {
            key: value
            for key, value in row.items()
            if key in MODEL_IMPROVEMENT_HISTORY_COLUMNS or key in ("model_key",)
        }
        for row in model_rows
    ]
    hypothesis = (
        f"次回は追加候補({numbers_to_text(should_include) if should_include else 'なし'})を拾えたモデル条件を比較し、"
        f"除外候補({numbers_to_text(should_exclude) if should_exclude else 'なし'})の重みを下げる検証を行う。"
        "これは研究・検証目的の仮説であり、将来の当選を保証するものではありません。"
    )
    main_row = {
        "lottery_type": lottery_type,
        "draw_no": int(draw_no),
        "prediction_id": str(prediction_id),
        "prediction_date": str(prediction_date or ""),
        "predicted_numbers": numbers_to_text(predicted),
        "actual_numbers": numbers_to_text(actual),
        "matched_numbers": numbers_to_text(matched) if matched else "",
        "missed_numbers": numbers_to_text(missed) if missed else "",
        "should_have_included_numbers": numbers_to_text(should_include) if should_include else "",
        "should_have_excluded_numbers": numbers_to_text(should_exclude) if should_exclude else "",
        "matched_count": len(matched),
        "failure_reason": failure_reason,
        "winning_condition_analysis": json_text(winning_conditions),
        "useful_models": " / ".join(optimization["useful_models"]),
        "weak_models": " / ".join(optimization["weak_models"]),
        "weight_up_models": " / ".join(optimization["weight_up_models"]),
        "weight_down_models": " / ".join(optimization["weight_down_models"]),
        "model_improvement_analysis": json_text(model_analysis),
        "ensemble_analysis": json_text(optimization["ensemble_analysis"]),
        "next_hypothesis": hypothesis,
        "created_at": created_at,
    }
    return main_row, model_rows


def read_history_csv(path, columns):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding).reindex(columns=columns)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path).reindex(columns=columns)


def read_history_jsonl(path, columns):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except Exception:
                continue
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns)


def write_history_jsonl(df, path):
    json_columns = {"winning_condition_analysis", "model_improvement_analysis", "ensemble_analysis"}
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in df.to_dict("records"):
            cleaned = {}
            for key, value in record.items():
                try:
                    if pd.isna(value):
                        value = ""
                except TypeError:
                    pass
                if key in json_columns and value:
                    cleaned[key] = parse_json_text(value, value)
                else:
                    cleaned[key] = value
            handle.write(json.dumps(cleaned, ensure_ascii=False, default=str) + "\n")


def record_key_set(df, key_columns):
    if df is None or df.empty:
        return set()
    frame = df.reindex(columns=key_columns)
    return {
        tuple(str(row.get(column, "")).strip() for column in key_columns)
        for _, row in frame.iterrows()
    }


def filter_existing_records(incoming, existing, key_columns):
    if incoming is None or incoming.empty:
        return incoming
    existing_keys = record_key_set(existing, key_columns)
    if not existing_keys:
        return incoming
    return incoming[
        incoming.apply(
            lambda row: tuple(str(row.get(column, "")).strip() for column in key_columns) not in existing_keys,
            axis=1,
        )
    ]


def append_winning_condition_history(history_dir, main_rows, model_rows):
    if not main_rows and not model_rows:
        return
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    main_path = history_dir / "winning_condition_history.csv"
    model_path = history_dir / "model_improvement_history.csv"
    jsonl_path = history_dir / "winning_condition_history.jsonl"

    if main_rows:
        existing = read_history_csv(main_path, WINNING_CONDITION_HISTORY_COLUMNS)
        incoming = pd.DataFrame(main_rows, columns=WINNING_CONDITION_HISTORY_COLUMNS)
        key_columns = ["lottery_type", "draw_no", "prediction_id"]
        incoming = filter_existing_records(incoming, existing, key_columns)
        if not incoming.empty:
            combined = pd.concat([existing, incoming], ignore_index=True).reindex(columns=WINNING_CONDITION_HISTORY_COLUMNS)
            combined = combined.drop_duplicates(key_columns, keep="first")
            combined.to_csv(main_path, index=False, encoding="utf-8-sig")
            write_history_jsonl(combined, jsonl_path)

    if model_rows:
        existing_models = read_history_csv(model_path, MODEL_IMPROVEMENT_HISTORY_COLUMNS)
        normalized_rows = [
            {column: row.get(column, "") for column in MODEL_IMPROVEMENT_HISTORY_COLUMNS}
            for row in model_rows
        ]
        incoming_models = pd.DataFrame(normalized_rows, columns=MODEL_IMPROVEMENT_HISTORY_COLUMNS)
        key_columns = ["lottery_type", "draw_no", "prediction_id", "model_name"]
        incoming_models = filter_existing_records(incoming_models, existing_models, key_columns)
        if not incoming_models.empty:
            combined_models = pd.concat([existing_models, incoming_models], ignore_index=True).reindex(columns=MODEL_IMPROVEMENT_HISTORY_COLUMNS)
            combined_models = combined_models.drop_duplicates(key_columns, keep="first")
            combined_models.to_csv(model_path, index=False, encoding="utf-8-sig")


def load_winning_condition_history(history_dir, lottery_type=None):
    history_dir = Path(history_dir)
    try:
        main_df = read_history_csv(history_dir / "winning_condition_history.csv", WINNING_CONDITION_HISTORY_COLUMNS)
    except Exception:
        main_df = read_history_jsonl(history_dir / "winning_condition_history.jsonl", WINNING_CONDITION_HISTORY_COLUMNS)
    if main_df.empty:
        main_df = read_history_jsonl(history_dir / "winning_condition_history.jsonl", WINNING_CONDITION_HISTORY_COLUMNS)
    try:
        model_df = read_history_csv(history_dir / "model_improvement_history.csv", MODEL_IMPROVEMENT_HISTORY_COLUMNS)
    except Exception:
        model_df = pd.DataFrame(columns=MODEL_IMPROVEMENT_HISTORY_COLUMNS)
    if lottery_type:
        main_df = main_df[main_df["lottery_type"].astype(str) == str(lottery_type)] if not main_df.empty else main_df
        model_df = model_df[model_df["lottery_type"].astype(str) == str(lottery_type)] if not model_df.empty else model_df
    return main_df, model_df


def canonical_model_from_name(model_name):
    text = str(model_name or "").strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return None, ""
    if text.startswith("custom:"):
        return text, text.split("custom:", 1)[1]
    key = canonical_model_key(text)
    if key in ARL_MODEL_LABELS:
        return key, ARL_MODEL_LABELS[key]
    for model_key, label in ARL_MODEL_LABELS.items():
        if text == label:
            return model_key, label
    for alias, model_key in MODEL_ALIASES.items():
        if text == alias:
            return model_key, ARL_MODEL_LABELS.get(model_key, text)
    return f"custom:{text}", text


def split_model_names(value):
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, (list, tuple, set)):
        raw_names = list(value)
    else:
        parsed = parse_json_text(value, None)
        if isinstance(parsed, (list, tuple, set)):
            raw_names = list(parsed)
        else:
            text = str(value or "")
            for separator in (" / ", "/", "、", ",", "\n", "\t"):
                text = text.replace(separator, "|")
            raw_names = text.split("|")
    names = []
    for name in raw_names:
        text = str(name or "").strip()
        if text and text.lower() not in {"nan", "none", "-"}:
            names.append(text)
    return names


def model_counter_from_column(df, column):
    counter = Counter()
    if df is None or df.empty or column not in df:
        return counter
    for value in df[column].fillna("").tolist():
        for name in split_model_names(value):
            key, _ = canonical_model_from_name(name)
            if key:
                counter[key] += 1
    return counter


def sorted_improvement_history(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=WINNING_CONDITION_HISTORY_COLUMNS)
    sorted_df = df.copy()
    sorted_df["_draw_sort"] = pd.to_numeric(sorted_df.get("draw_no"), errors="coerce").fillna(0)
    sorted_df["_created_sort"] = pd.to_datetime(sorted_df.get("created_at"), errors="coerce")
    return sorted_df.sort_values(["_draw_sort", "_created_sort"], ascending=[True, True])


def effective_model_counters(model_df, limit=5):
    useful = Counter()
    weak = Counter()
    if model_df is None or model_df.empty or "model_name" not in model_df:
        return useful, weak
    model_scores = model_df.copy()
    model_scores["matched_count"] = pd.to_numeric(model_scores.get("matched_count"), errors="coerce").fillna(0)
    grouped = (
        model_scores.groupby("model_name", dropna=False)["matched_count"]
        .agg(["mean", "count"])
        .reset_index()
    )
    if grouped.empty:
        return useful, weak
    top = grouped.sort_values(["mean", "count"], ascending=[False, False]).head(limit)
    bottom = grouped.sort_values(["mean", "count"], ascending=[True, False]).head(limit)
    for _, row in top.iterrows():
        key, _ = canonical_model_from_name(row["model_name"])
        if key:
            useful[key] += int(row["count"])
    for _, row in bottom.iterrows():
        key, _ = canonical_model_from_name(row["model_name"])
        if key:
            weak[key] += int(row["count"])
    return useful, weak


def clamp_float(value, lower, upper):
    return max(lower, min(upper, float(value)))


def counter_to_records(counter, limit=6):
    rows = []
    for key, count in counter.most_common(limit):
        _, label = canonical_model_from_name(key)
        rows.append({"model_key": key, "model_name": label or model_label(key), "count": int(count)})
    return rows


def format_model_counter(counter, limit=5):
    if not counter:
        return "-"
    parts = []
    for key, count in counter.most_common(limit):
        _, label = canonical_model_from_name(key)
        parts.append(f"{label or model_label(key)}({int(count)})")
    return " / ".join(parts) if parts else "-"


def build_ai_improvement_weight_summary(history_dir, lottery_type=None):
    warnings = []
    try:
        history, model_history = load_winning_condition_history(history_dir, lottery_type)
    except Exception as exc:
        return {
            "lottery_type": lottery_type or "",
            "available": False,
            "warnings": [f"AI改善履歴を読み込めませんでした: {exc}"],
            "history_count": 0,
            "model_history_count": 0,
            "latest_draw_no": "-",
            "latest_created_at": "-",
            "latest_hypothesis": "-",
            "model_weights": {},
            "recent5_weight_up": Counter(),
            "recent10_weight_up": Counter(),
            "recent5_weight_down": Counter(),
            "recent10_weight_down": Counter(),
            "long_useful": Counter(),
            "long_weak": Counter(),
        }

    if history.empty and model_history.empty:
        warnings.append("AI改善履歴がまだありません。通常の候補スコア活用予測にフォールバックします。")

    history = sorted_improvement_history(history)
    recent5 = history.tail(5)
    recent10 = history.tail(10)

    recent5_weight_up = model_counter_from_column(recent5, "weight_up_models")
    recent10_weight_up = model_counter_from_column(recent10, "weight_up_models")
    recent5_weight_down = model_counter_from_column(recent5, "weight_down_models")
    recent10_weight_down = model_counter_from_column(recent10, "weight_down_models")
    recent5_useful = model_counter_from_column(recent5, "useful_models")
    recent10_useful = model_counter_from_column(recent10, "useful_models")
    recent5_weak = model_counter_from_column(recent5, "weak_models")
    recent10_weak = model_counter_from_column(recent10, "weak_models")
    all_useful = model_counter_from_column(history, "useful_models")
    all_weak = model_counter_from_column(history, "weak_models")
    effective_useful, effective_weak = effective_model_counters(model_history)

    long_useful = all_useful + effective_useful
    long_weak = all_weak + effective_weak
    weight_scores = Counter()
    for key, count in recent5_weight_up.items():
        weight_scores[key] += count * 0.055
    for key, count in recent10_weight_up.items():
        weight_scores[key] += count * 0.025
    for key, count in recent5_useful.items():
        weight_scores[key] += count * 0.02
    for key, count in recent10_useful.items():
        weight_scores[key] += count * 0.01
    for key, count in long_useful.items():
        weight_scores[key] += count * 0.004
    for key, count in recent5_weight_down.items():
        weight_scores[key] -= count * 0.055
    for key, count in recent10_weight_down.items():
        weight_scores[key] -= count * 0.025
    for key, count in recent5_weak.items():
        weight_scores[key] -= count * 0.02
    for key, count in recent10_weak.items():
        weight_scores[key] -= count * 0.01
    for key, count in long_weak.items():
        weight_scores[key] -= count * 0.004

    model_weights = {
        key: round(clamp_float(score, -0.28, 0.28), 4)
        for key, score in weight_scores.items()
        if abs(score) >= 0.005
    }
    if history.empty:
        latest = {}
    else:
        latest = history.iloc[-1].to_dict()
    return {
        "lottery_type": lottery_type or "",
        "available": bool(model_weights),
        "warnings": warnings,
        "history_count": int(len(history)),
        "model_history_count": int(len(model_history)),
        "latest_draw_no": latest.get("draw_no", "-") if latest else "-",
        "latest_created_at": latest.get("created_at", "-") if latest else "-",
        "latest_hypothesis": latest.get("next_hypothesis", "-") if latest else "-",
        "model_weights": model_weights,
        "recent5_weight_up": recent5_weight_up,
        "recent10_weight_up": recent10_weight_up,
        "recent5_weight_down": recent5_weight_down,
        "recent10_weight_down": recent10_weight_down,
        "long_useful": long_useful,
        "long_weak": long_weak,
        "weight_up_records": counter_to_records(recent10_weight_up),
        "weight_down_records": counter_to_records(recent10_weight_down),
    }


def ai_improvement_weight_rows(summary):
    summary = summary or {}
    return pd.DataFrame(
        [
            {
                "区分": "重み上げ候補",
                "直近5回": format_model_counter(summary.get("recent5_weight_up", Counter())),
                "直近10回": format_model_counter(summary.get("recent10_weight_up", Counter())),
                "長期": format_model_counter(summary.get("long_useful", Counter())),
            },
            {
                "区分": "重み下げ候補",
                "直近5回": format_model_counter(summary.get("recent5_weight_down", Counter())),
                "直近10回": format_model_counter(summary.get("recent10_weight_down", Counter())),
                "長期": format_model_counter(summary.get("long_weak", Counter())),
            },
        ]
    )


def weighted_model_text(model_weights, positive=True, limit=5):
    if not model_weights:
        return "-"
    items = [
        (key, weight)
        for key, weight in model_weights.items()
        if (weight > 0 if positive else weight < 0)
    ]
    items = sorted(items, key=lambda item: abs(item[1]), reverse=True)[:limit]
    if not items:
        return "-"
    parts = []
    for key, weight in items:
        _, label = canonical_model_from_name(key)
        parts.append(f"{label or model_label(key)}({weight:+.3f})")
    return " / ".join(parts)


def score_row_model_adjustment(row, model_key, model_weight):
    columns = AI_MODEL_SCORE_COLUMNS.get(canonical_model_key(model_key), ["総合スコア"])
    values = []
    for column in columns:
        if column in row:
            try:
                values.append(float(row.get(column, 0) or 0))
            except Exception:
                values.append(0.0)
    if not values:
        return 0.0
    return (sum(values) / len(values)) * float(model_weight)


def apply_ai_improvement_weights(score_df, weight_summary):
    if score_df is None or score_df.empty:
        return pd.DataFrame()
    weighted = score_df.copy()
    if "総合スコア" not in weighted.columns:
        weighted["総合スコア"] = pd.to_numeric(weighted.get("スコア", 0), errors="coerce").fillna(0)
    model_weights = (weight_summary or {}).get("model_weights", {})
    if not model_weights:
        weighted["AI改善加点"] = 0.0
        weighted["AI改善後スコア"] = pd.to_numeric(weighted["総合スコア"], errors="coerce").fillna(0).round(3)
        weighted["AI改善理由"] = ""
        weighted["AI改善関連モデル"] = ""
        weighted["AI改善抑制モデル"] = ""
        return weighted

    adjustments = []
    reasons = []
    positive_models = []
    negative_models = []
    for _, row in weighted.iterrows():
        raw_adjustment = 0.0
        positive = []
        negative = []
        for model_key, model_weight in model_weights.items():
            contribution = score_row_model_adjustment(row, model_key, model_weight)
            raw_adjustment += contribution
            _, label = canonical_model_from_name(model_key)
            label = label or model_label(model_key)
            if contribution >= 0.4:
                positive.append(label)
            elif contribution <= -0.4:
                negative.append(label)
        adjustment = round(clamp_float(raw_adjustment, -15.0, 15.0), 3)
        adjustments.append(adjustment)
        positive_models.append(" / ".join(dict.fromkeys(positive[:4])))
        negative_models.append(" / ".join(dict.fromkeys(negative[:4])))
        reason_parts = []
        if positive:
            reason_parts.append("加点: " + " / ".join(dict.fromkeys(positive[:4])))
        if negative:
            reason_parts.append("減点: " + " / ".join(dict.fromkeys(negative[:4])))
        reasons.append("、".join(reason_parts))
    weighted["AI改善加点"] = adjustments
    weighted["AI改善後スコア"] = (
        pd.to_numeric(weighted["総合スコア"], errors="coerce").fillna(0) + weighted["AI改善加点"]
    ).clip(lower=0, upper=120).round(3)
    weighted["AI改善理由"] = reasons
    weighted["AI改善関連モデル"] = positive_models
    weighted["AI改善抑制モデル"] = negative_models
    weighted = weighted.sort_values(["AI改善後スコア", "総合スコア", "数字"], ascending=[False, False, True]).reset_index(drop=True)
    weighted["AI改善順位"] = range(1, len(weighted) + 1)
    return weighted


def primary_support(number, support_map):
    supports = support_map.get(int(number), [])
    return supports[0] if supports else "未特定モデル"


def build_hit_factor_summary(predicted, actual, support_map):
    matched = sorted(set(predicted) & set(actual))
    if not matched:
        return "一致数字なし。的中要因は次回検証対象として保留。"
    return " / ".join(f"{number:02d} -> {primary_support(number, support_map)}由来" for number in matched)


def build_effective_conditions(predicted, actual, support_map):
    matched = sorted(set(predicted) & set(actual))
    labels = []
    for number in matched:
        labels.extend(support_map.get(number, []))
    if not labels:
        return "有効条件なし"
    counts = Counter(labels)
    return " / ".join(label for label, _ in counts.most_common(5))


def condition_counts(numbers, number_max):
    high_floor = 32
    tails = Counter(number % 10 for number in numbers)
    return {
        "high": sum(number >= high_floor for number in numbers),
        "odd": sum(number % 2 for number in numbers),
        "consecutive": count_consecutive_pairs(numbers),
        "same_tail": sum(1 for count in tails.values() if count >= 2),
        "fixed": sum(number in {4, 9, 13} for number in numbers),
        "sum": sum(numbers),
    }


def build_missing_excess_conditions(predicted, actual, number_max):
    predicted_counts = condition_counts(predicted, number_max)
    actual_counts = condition_counts(actual, number_max)
    missing = []
    excess = []
    labels = [
        ("high", "32以上"),
        ("odd", "奇数"),
        ("consecutive", "連番"),
        ("same_tail", "同末尾"),
        ("fixed", "4・9・13候補"),
    ]
    for key, label in labels:
        diff = actual_counts[key] - predicted_counts[key]
        if diff > 0:
            missing.append(f"{label}+{diff}")
        elif diff < 0:
            excess.append(f"{label}{diff}")
    sum_diff = actual_counts["sum"] - predicted_counts["sum"]
    if abs(sum_diff) >= 15:
        if sum_diff > 0:
            missing.append(f"合計値+{sum_diff}")
        else:
            excess.append(f"合計値{sum_diff}")
    return {
        "missing": " / ".join(missing) if missing else "大きな不足条件なし",
        "excess": " / ".join(excess) if excess else "大きな過剰条件なし",
    }


def format_contribution_detail(predicted, actual, support_map):
    details = []
    actual_set = set(actual)
    for number in sorted(predicted):
        hit = "的中" if number in actual_set else "外れ"
        supports = "・".join(support_map.get(number, ["未特定モデル"]))
        details.append(f"{number:02d}:{hit}:{supports}")
    return " / ".join(details)


def build_contribution_rows(game_name, prediction_row, predicted, actual, support_map, saved_at=None):
    saved_at = saved_at or datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    rows = []
    actual_set = set(actual)
    for number in sorted(predicted):
        supports = support_map.get(number) or ["未特定モデル"]
        hit = number in actual_set
        score = round(1 / len(supports), 4) if hit else 0.0
        for support in supports:
            rows.append(
                {
                    "研究所": game_name,
                    "予想ID": str(prediction_row.get("予想ID", "")),
                    "開催回": safe_int(prediction_row.get("開催回")),
                    "候補番号": safe_int(prediction_row.get("候補番号")),
                    "予想数字": f"{number:02d}",
                    "選出モデル": support,
                    "複数モデル一致": "あり" if len(supports) >= 2 else "なし",
                    "的中": "あり" if hit else "なし",
                    "寄与スコア": score,
                    "保存日時": saved_at,
                }
            )
    return rows


def merge_contribution_rows(existing, rows):
    incoming = pd.DataFrame(rows, columns=CONTRIBUTION_COLUMNS)
    if incoming.empty:
        return existing.reindex(columns=CONTRIBUTION_COLUMNS)
    if existing is None or existing.empty:
        return incoming
    existing = existing.reindex(columns=CONTRIBUTION_COLUMNS)
    incoming = filter_existing_records(incoming, existing, ["予想ID"])
    if incoming.empty:
        return existing
    return pd.concat([existing, incoming], ignore_index=True).reindex(columns=CONTRIBUTION_COLUMNS)


def build_research_cycle_rows(game_name, report_rows, saved_at=None):
    saved_at = saved_at or datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    rows = []
    for report in report_rows:
        match_text = f"一致数 {report.get('本数字一致数', '-')}"
        if "ボーナス一致" in report:
            match_text += f" / ボーナス {report.get('ボーナス一致', '-')}"
        if "ボーナス一致数" in report:
            match_text += f" / ボーナス {report.get('ボーナス一致数', '-')}"
        rows.append(
            {
                "研究所": game_name,
                "予想ID": report.get("予想ID", ""),
                "開催回": safe_int(report.get("開催回")),
                "仮説": report.get("次回の仮説", ""),
                "分析": report.get("的中要因", report.get("有効条件", "")),
                "予想": report.get("予想番号", ""),
                "結果": report.get("実際の当選番号", ""),
                "検証": match_text,
                "改善": report.get("改善案", ""),
                "再予想": report.get("次回の仮説", ""),
                "保存日時": saved_at,
            }
        )
    return rows


def merge_research_cycle_rows(existing, rows):
    incoming = pd.DataFrame(rows, columns=RESEARCH_CYCLE_COLUMNS)
    if incoming.empty:
        return existing.reindex(columns=RESEARCH_CYCLE_COLUMNS)
    if existing is None or existing.empty:
        return incoming
    existing = existing.reindex(columns=RESEARCH_CYCLE_COLUMNS)
    incoming = filter_existing_records(incoming, existing, ["予想ID"])
    if incoming.empty:
        return existing
    return pd.concat([existing, incoming], ignore_index=True).reindex(columns=RESEARCH_CYCLE_COLUMNS)


def build_contribution_ranking(contributions):
    if contributions is None or contributions.empty or "的中" not in contributions:
        return pd.DataFrame(columns=["順位", "モデル", "平均寄与率", "寄与スコア", "的中数字数"])
    df = contributions.copy()
    df["寄与スコア"] = pd.to_numeric(df["寄与スコア"], errors="coerce").fillna(0)
    hit_df = df[df["的中"].astype(str) == "あり"]
    if hit_df.empty:
        return pd.DataFrame(columns=["順位", "モデル", "平均寄与率", "寄与スコア", "的中数字数"])
    grouped = hit_df.groupby("選出モデル").agg(寄与スコア=("寄与スコア", "sum"), 的中数字数=("予想数字", "count")).reset_index()
    total = grouped["寄与スコア"].sum() or 1
    grouped["平均寄与率"] = (grouped["寄与スコア"] / total * 100).round(1).astype(str) + "%"
    grouped = grouped.sort_values("寄与スコア", ascending=False).reset_index(drop=True)
    grouped.insert(0, "順位", grouped.index + 1)
    return grouped.rename(columns={"選出モデル": "モデル"})


def add_verification_metrics(reports, draw_size):
    if reports is None or reports.empty:
        return reports
    df = reports.copy()
    match_count = pd.to_numeric(
        df["本数字一致数"] if "本数字一致数" in df else pd.Series([0] * len(df), index=df.index),
        errors="coerce",
    ).fillna(0)

    if "ボーナス一致数" in df:
        bonus_score = pd.to_numeric(df["ボーナス一致数"], errors="coerce").fillna(0) * 0.25
    elif "ボーナス一致" in df:
        bonus_score = df["ボーナス一致"].astype(str).map(lambda value: 0.5 if value == "あり" else 0.0)
    else:
        bonus_score = pd.Series([0.0] * len(df), index=df.index)

    hit_rate = (match_count / max(draw_size, 1) * 100).round(1)
    grade = df["等級判定"].astype(str) if "等級判定" in df else pd.Series(["該当なし"] * len(df), index=df.index)
    win_rate = grade.map(lambda value: 0.0 if value in ("", "nan", "該当なし") else 100.0)
    expected_value = ((match_count + bonus_score) / max(draw_size, 1)).round(3)

    for column, values in [("的中率", hit_rate), ("勝率", win_rate), ("期待値", expected_value)]:
        existing = pd.to_numeric(df[column], errors="coerce") if column in df else pd.Series([pd.NA] * len(df), index=df.index)
        df[column] = existing.fillna(values)
    return df


def ensure_purchase_columns(purchases):
    if purchases is None or purchases.empty:
        return pd.DataFrame(columns=PURCHASE_COLUMNS)
    df = purchases.copy()
    for column in PURCHASE_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df.reindex(columns=PURCHASE_COLUMNS)


def purchase_grade(lottery_type, matched_count, bonus_matched_count):
    lottery_type = str(lottery_type)
    matched_count = safe_int(matched_count)
    bonus_matched_count = safe_int(bonus_matched_count)
    if lottery_type == "loto7":
        if matched_count == 7:
            return "1等"
        if matched_count == 6 and bonus_matched_count >= 1:
            return "2等"
        if matched_count == 6:
            return "3等"
        if matched_count == 5:
            return "4等"
        if matched_count == 4:
            return "5等"
        if matched_count == 3 and bonus_matched_count >= 1:
            return "6等"
        return "該当なし"
    if matched_count == 6:
        return "1等"
    if matched_count == 5 and bonus_matched_count >= 1:
        return "2等"
    if matched_count == 5:
        return "3等"
    if matched_count == 4:
        return "4等"
    if matched_count == 3:
        return "5等"
    return "該当なし"


def result_numbers_from_row(result_row, lottery_type):
    if result_row is None:
        return [], []
    if lottery_type == "loto7":
        main_columns = [f"第{index}数字" for index in range(1, 8)]
        bonus_columns = ["BONUS数字1", "BONUS数字2"]
        number_max = 37
    else:
        main_columns = [f"第{index}数字" for index in range(1, 7)]
        bonus_columns = ["BONUS数字"]
        number_max = 43
    main_numbers = [safe_int(result_row.get(column), None) for column in main_columns if safe_int(result_row.get(column), None) is not None]
    bonus_numbers = [safe_int(result_row.get(column), None) for column in bonus_columns if safe_int(result_row.get(column), None) is not None]
    if not main_numbers and "本数字" in result_row:
        main_numbers = parse_numbers(result_row.get("本数字"), number_max)
    if not bonus_numbers and "ボーナス数字" in result_row:
        bonus_numbers = parse_numbers(result_row.get("ボーナス数字"), number_max)
    return sorted(main_numbers), sorted(bonus_numbers)


def numeric_money(value, default=None):
    try:
        text = str(value).replace(",", "").replace("円", "").strip()
        if text == "" or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except Exception:
        return default


def prize_payout_from_result(result_row, prize_rank, ticket_count=1, manual_payout=None):
    if prize_rank in ("", "-", "該当なし", "結果待ち"):
        return 0.0, True
    manual_value = numeric_money(manual_payout, None)
    column = f"{prize_rank}賞金"
    prize_value = numeric_money(result_row.get(column, "") if result_row is not None else "", None)
    if prize_value is None:
        return (manual_value, True) if manual_value is not None else ("", False)
    return float(prize_value) * max(safe_int(ticket_count, 1), 1), True


def evaluate_purchase_history(purchases, results, lottery_type, draw_size, number_max):
    purchases = ensure_purchase_columns(purchases)
    if purchases.empty:
        return purchases
    game_rows = purchases[purchases["lottery_type"].astype(str) == str(lottery_type)].copy()
    other_rows = purchases[purchases["lottery_type"].astype(str) != str(lottery_type)].copy()
    if game_rows.empty:
        return purchases

    result_df = results.copy() if results is not None else pd.DataFrame()
    if not result_df.empty and "開催回" in result_df.columns:
        result_df["開催回"] = pd.to_numeric(result_df["開催回"], errors="coerce").fillna(0).astype(int)

    evaluated_rows = []
    for _, row in game_rows.iterrows():
        updated = row.to_dict()
        draw_no = safe_int(updated.get("draw_no"))
        ticket_numbers = parse_numbers(updated.get("numbers"), number_max)
        ticket_count = max(safe_int(updated.get("ticket_count"), 1), 1)
        cost = numeric_money(updated.get("cost"), 0.0) or 0.0
        result_row = None
        if not result_df.empty and "開催回" in result_df.columns:
            matches = result_df[result_df["開催回"] == int(draw_no)]
            if not matches.empty:
                result_row = matches.tail(1).iloc[0]
        if result_row is None:
            updated.update(
                {
                    "result_numbers": "",
                    "bonus_numbers": "",
                    "matched_count": "",
                    "bonus_matched_count": "",
                    "prize_rank": "結果待ち",
                    "payout": "",
                    "profit_loss": "",
                    "status": "結果待ち",
                }
            )
            evaluated_rows.append(updated)
            continue
        result_numbers, bonus_numbers = result_numbers_from_row(result_row, lottery_type)
        matched = sorted(set(ticket_numbers) & set(result_numbers))
        bonus_matched = sorted(set(ticket_numbers) & set(bonus_numbers))
        prize_rank = purchase_grade(lottery_type, len(matched), len(bonus_matched))
        payout, payout_confirmed = prize_payout_from_result(result_row, prize_rank, ticket_count, updated.get("payout"))
        if payout_confirmed:
            payout_value = numeric_money(payout, 0.0) or 0.0
            profit_loss = payout_value - cost
            status = "未当選" if prize_rank == "該当なし" else "照合済み"
        else:
            payout_value = ""
            profit_loss = ""
            status = "払戻未確定"
        updated.update(
            {
                "result_numbers": numbers_to_text(result_numbers),
                "bonus_numbers": numbers_to_text(bonus_numbers),
                "matched_count": len(matched),
                "bonus_matched_count": len(bonus_matched),
                "prize_rank": prize_rank,
                "payout": payout_value,
                "profit_loss": profit_loss,
                "status": status,
            }
        )
        evaluated_rows.append(updated)
    combined = pd.concat([other_rows, pd.DataFrame(evaluated_rows)], ignore_index=True)
    return ensure_purchase_columns(combined)


def purchase_display_df(purchases, lottery_type=None):
    purchases = ensure_purchase_columns(purchases)
    if lottery_type:
        purchases = purchases[purchases["lottery_type"].astype(str) == str(lottery_type)].copy()
    if purchases.empty:
        return pd.DataFrame(columns=PURCHASE_DISPLAY_COLUMNS)
    display = pd.DataFrame(
        {
            "開催回": pd.to_numeric(purchases["draw_no"], errors="coerce").fillna(0).astype(int),
            "購入日": purchases["purchase_date"],
            "購入番号": purchases["numbers"],
            "予測方式": purchases["prediction_method"],
            "モデル名": purchases["model_name"],
            "口数": pd.to_numeric(purchases["ticket_count"], errors="coerce").fillna(0).astype(int),
            "購入金額": pd.to_numeric(purchases["cost"], errors="coerce").fillna(0).round(0).astype(int),
            "一致数": purchases["matched_count"],
            "BONUS一致数": purchases["bonus_matched_count"],
            "当選等級": purchases["prize_rank"],
            "払戻金": purchases["payout"],
            "収支": purchases["profit_loss"],
            "状態": purchases["status"].replace("", "未照合"),
            "メモ": purchases["notes"],
        }
    )
    return display.sort_values(["開催回", "購入日"], ascending=[False, False]).reset_index(drop=True)


def build_purchase_summary(purchases, lottery_type=None):
    purchases = ensure_purchase_columns(purchases)
    if lottery_type:
        purchases = purchases[purchases["lottery_type"].astype(str) == str(lottery_type)].copy()
    total_cost = float(pd.to_numeric(purchases.get("cost", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not purchases.empty else 0.0
    total_payout = float(pd.to_numeric(purchases.get("payout", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not purchases.empty else 0.0
    profit_loss = total_payout - total_cost
    return_rate = round(total_payout / total_cost * 100, 1) if total_cost else 0.0
    hit_count = 0
    max_payout = 0.0
    best_method = "-"
    if not purchases.empty:
        prize = purchases["prize_rank"].astype(str)
        hit_count = int((~prize.isin(["", "nan", "該当なし", "結果待ち"])).sum())
        max_payout = float(pd.to_numeric(purchases["payout"], errors="coerce").fillna(0).max())
        method_summary = build_purchase_group_summary(purchases, "prediction_method", "予測方式")
        if not method_summary.empty:
            best_method = str(method_summary.iloc[0]["予測方式"])
    return {
        "総購入金額": total_cost,
        "総払戻金": total_payout,
        "累計収支": profit_loss,
        "回収率": return_rate,
        "的中回数": hit_count,
        "最高払戻金": max_payout,
        "実戦成績が良い予測方式": best_method,
    }


def build_purchase_group_summary(purchases, group_column, label_column):
    purchases = ensure_purchase_columns(purchases)
    columns = [label_column, "件数", "総購入金額", "総払戻金", "収支", "回収率", "的中回数", "最高払戻金"]
    if purchases.empty or group_column not in purchases.columns:
        return pd.DataFrame(columns=columns)
    df = purchases.copy()
    df[group_column] = df[group_column].fillna("").replace("", "未記録")
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0)
    df["payout"] = pd.to_numeric(df["payout"], errors="coerce").fillna(0)
    hit_mask = ~df["prize_rank"].astype(str).isin(["", "nan", "該当なし", "結果待ち"])
    rows = []
    for value, group in df.groupby(group_column):
        cost = float(group["cost"].sum())
        payout = float(group["payout"].sum())
        rows.append(
            {
                label_column: value,
                "件数": int(len(group)),
                "総購入金額": int(round(cost)),
                "総払戻金": int(round(payout)),
                "収支": int(round(payout - cost)),
                "回収率": round(payout / cost * 100, 1) if cost else 0.0,
                "的中回数": int(hit_mask.loc[group.index].sum()),
                "最高払戻金": int(round(float(group["payout"].max()))),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["収支", "回収率", "総払戻金"], ascending=False).reset_index(drop=True)


def ranking_target_models():
    labels = list(ARL_MODEL_LABELS.values())
    for label in ("候補スコア活用予測", "AI改善反映予測"):
        if label not in labels:
            labels.append(label)
    for label in (CHAMINI6_GOD_MODE_LABEL,):
        if label not in labels:
            labels.append(label)
    return labels


def ranking_model_type(model_name):
    if model_name in (CHAMINI6_GOD_MODE_KEY, CHAMINI6_GOD_MODE_LABEL):
        return "Chamini6 God Mode"
    if model_name in (ANTI_POPULAR_EXPECTED_VALUE_KEY, ANTI_POPULAR_EXPECTED_VALUE_LABEL):
        return "補助モデル"
    if model_name in ("候補スコア活用予測", "AI改善反映予測"):
        return "予測方式"
    if model_name == ARL_MODEL_LABELS.get("random_baseline"):
        return "基準モデル"
    if model_name in ARL_MODEL_LABELS.values():
        return "分析モデル"
    return "保存済み予測"


def safe_mean(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def safe_max(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def rounded_or_none(value, digits=3):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return round(float(value), digits)


def model_history_ai_scores(model_history):
    if model_history is None or model_history.empty or "model_name" not in model_history or "matched_count" not in model_history:
        return {}
    df = model_history.copy()
    df["matched_count"] = pd.to_numeric(df["matched_count"], errors="coerce")
    grouped = df.dropna(subset=["matched_count"]).groupby("model_name")["matched_count"].mean()
    return {str(model): float(score) for model, score in grouped.items()}


def build_model_dashboard(
    reports,
    match_column="本数字一致数",
    model_column="使用モデル",
    draw_size=None,
    model_history=None,
    include_target_models=False,
):
    columns = [
        "順位",
        "モデル",
        "種別",
        "検証数",
        "平均一致数",
        "最大一致数",
        "直近5回成績",
        "直近10回成績",
        "長期成績",
        "安定性",
        "AI改善後成績",
        "改善前後の差",
        "勝率",
        "期待値",
        "総合評価",
        "状態",
    ]
    target_models = ranking_target_models() if include_target_models else []
    if reports is None or reports.empty or match_column not in reports:
        if not target_models:
            return pd.DataFrame(columns=columns)
        rows = [
            {
                "順位": "-",
                "モデル": model,
                "種別": ranking_model_type(model),
                "検証数": 0,
                "平均一致数": None,
                "最大一致数": None,
                "直近5回成績": None,
                "直近10回成績": None,
                "長期成績": None,
                "安定性": None,
                "AI改善後成績": None,
                "改善前後の差": None,
                "勝率": None,
                "期待値": None,
                "総合評価": None,
                "状態": "検証待ち",
            }
            for model in target_models
        ]
        return pd.DataFrame(rows, columns=columns)
    if draw_size is None:
        lengths = reports["予想番号"].map(lambda value: len(parse_numbers(value))) if "予想番号" in reports else pd.Series([0])
        draw_size = int(lengths.max()) if len(lengths) and int(lengths.max()) > 0 else 1
    df = add_verification_metrics(reports, draw_size)
    df[match_column] = pd.to_numeric(df[match_column], errors="coerce").fillna(0)
    df["期待値"] = pd.to_numeric(df["期待値"], errors="coerce").fillna(0)
    if model_column not in df:
        df[model_column] = "未記録"
    df[model_column] = df[model_column].fillna("").replace("", "未記録")
    if "開催回" in df:
        df["開催回"] = pd.to_numeric(df["開催回"], errors="coerce").fillna(0)
        df = df.sort_values("開催回")
    ai_scores = model_history_ai_scores(model_history)
    if "AI改善反映予測" in set(df[model_column].astype(str)):
        ai_scores["AI改善反映予測"] = float(df[df[model_column].astype(str) == "AI改善反映予測"][match_column].mean())
    base_score_method_mean = None
    if "候補スコア活用予測" in set(df[model_column].astype(str)):
        base_score_method_mean = float(df[df[model_column].astype(str) == "候補スコア活用予測"][match_column].mean())
    rows = []
    all_models = list(dict.fromkeys(target_models + sorted(df[model_column].astype(str).unique().tolist())))
    for model in all_models:
        model_rows = df[df[model_column].astype(str) == str(model)].copy()
        if model_rows.empty:
            rows.append(
                {
                    "順位": "-",
                    "モデル": model,
                    "種別": ranking_model_type(model),
                    "検証数": 0,
                    "平均一致数": None,
                    "最大一致数": None,
                    "直近5回成績": None,
                    "直近10回成績": None,
                    "長期成績": None,
                    "安定性": None,
                    "AI改善後成績": rounded_or_none(ai_scores.get(str(model))),
                    "改善前後の差": None,
                    "勝率": None,
                    "期待値": None,
                    "総合評価": None,
                    "状態": "検証待ち",
                }
            )
            continue
        matches = pd.to_numeric(model_rows[match_column], errors="coerce").fillna(0)
        match_std = float(matches.std()) if len(model_rows) > 1 else 0.0
        stability = max(0.0, 100.0 - (match_std / max(draw_size, 1) * 100))
        avg_match = float(matches.mean())
        recent5 = float(matches.tail(5).mean())
        recent10 = float(matches.tail(10).mean())
        long_score = avg_match
        ai_after = ai_scores.get(str(model))
        previous_rows = matches.iloc[:-5]
        previous_avg = float(previous_rows.mean()) if len(previous_rows) else None
        if model == "AI改善反映予測" and base_score_method_mean is not None:
            improvement_delta = avg_match - base_score_method_mean
        elif ai_after is not None:
            improvement_delta = ai_after - long_score
        elif previous_avg is not None:
            improvement_delta = recent5 - previous_avg
        else:
            improvement_delta = 0.0
        win_rate = safe_mean(model_rows["勝率"]) if "勝率" in model_rows else None
        expected_value = safe_mean(model_rows["期待値"]) if "期待値" in model_rows else None
        evaluation_score = (
            avg_match * 35
            + recent5 * 12
            + recent10 * 8
            + stability / 100 * 12
            + (win_rate or 0) / 100 * 10
            + (expected_value or 0) * 12
            + max(improvement_delta, -1.0) * 6
        )
        rows.append(
            {
                "順位": 0,
                "モデル": model,
                "種別": ranking_model_type(model),
                "検証数": int(len(model_rows)),
                "平均一致数": rounded_or_none(avg_match),
                "最大一致数": int(safe_max(matches) or 0),
                "直近5回成績": rounded_or_none(recent5),
                "直近10回成績": rounded_or_none(recent10),
                "長期成績": rounded_or_none(long_score),
                "安定性": rounded_or_none(stability, 1),
                "AI改善後成績": rounded_or_none(ai_after),
                "改善前後の差": rounded_or_none(improvement_delta),
                "勝率": rounded_or_none(win_rate, 1),
                "期待値": rounded_or_none(expected_value),
                "総合評価": rounded_or_none(evaluation_score, 3),
                "状態": "評価中",
            }
        )
    dashboard = pd.DataFrame(rows, columns=columns)
    if dashboard.empty:
        return dashboard
    dashboard["_sort_score"] = pd.to_numeric(dashboard["総合評価"], errors="coerce").fillna(-1)
    dashboard["_sort_avg"] = pd.to_numeric(dashboard["平均一致数"], errors="coerce").fillna(-1)
    dashboard["_sort_stability"] = pd.to_numeric(dashboard["安定性"], errors="coerce").fillna(-1)
    dashboard = dashboard.sort_values(["_sort_score", "_sort_avg", "_sort_stability", "モデル"], ascending=[False, False, False, True]).reset_index(drop=True)
    ranked_mask = dashboard["検証数"].fillna(0).astype(int) > 0
    dashboard.loc[ranked_mask, "順位"] = range(1, int(ranked_mask.sum()) + 1)
    dashboard.loc[~ranked_mask, "順位"] = "-"
    return dashboard.drop(columns=["_sort_score", "_sort_avg", "_sort_stability"])


def _clamp(value, low=0.0, high=100.0):
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return default


def _column_at(df, index):
    if df is None or df.empty or len(df.columns) <= index:
        return None
    return df.columns[index]


def _numeric_series(df, column):
    if df is None or df.empty or column is None or column not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _bonus_match_series(df, column):
    if df is None or df.empty or column is None or column not in df:
        return pd.Series([0.0] * len(df), index=df.index)
    numeric = pd.to_numeric(df[column], errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0.0)

    def text_to_score(value):
        text = str(value).strip().lower()
        if text in ("", "nan", "none", "-", "0", "0.0", "false", "no", "なし"):
            return 0.0
        return 1.0

    return df[column].map(text_to_score).astype(float)


def _verification_column_set(reports):
    return {
        "draw": _column_at(reports, 1),
        "model": _column_at(reports, 3),
        "prediction": _column_at(reports, 4),
        "actual": _column_at(reports, 5),
        "bonus": _column_at(reports, 6),
        "match": _column_at(reports, 7),
        "bonus_match": _column_at(reports, 9),
        "failure": _column_at(reports, 30),
        "improvement": _column_at(reports, 32),
        "hypothesis": _column_at(reports, 33),
    }


def build_model_performance_weights(reports, draw_size, lottery_type="", model_history=None):
    created_at = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    if reports is None or reports.empty:
        return pd.DataFrame(
            [
                {
                    "lottery_type": lottery_type,
                    "created_at": created_at,
                    "model_key": model_key,
                    "model_name": label,
                    "evaluation_count": 0,
                    "average_match": None,
                    "recent5_score": None,
                    "recent10_score": None,
                    "long_score": None,
                    "stability": None,
                    "expected_value": None,
                    "bonus_match_rate": None,
                    "raw_score": 0.0,
                    "applied_weight": 1.0,
                    "status": "検証待ち",
                }
                for model_key, label in ARL_MODEL_LABELS.items()
            ],
            columns=MODEL_WEIGHT_HISTORY_COLUMNS,
        )

    df = add_verification_metrics(reports, draw_size)
    columns = _verification_column_set(df)
    model_column = columns["model"]
    match_column = columns["match"]
    bonus_column = columns["bonus_match"]
    if model_column is None or match_column is None:
        return build_model_performance_weights(pd.DataFrame(), draw_size, lottery_type, model_history)

    df = df.copy()
    df[model_column] = df[model_column].fillna("").astype(str)
    ai_scores = model_history_ai_scores(model_history)
    weight_rows = []
    for model_key, label in ARL_MODEL_LABELS.items():
        model_aliases = {str(label), str(model_key)}
        model_rows = df[df[model_column].isin(model_aliases)].copy()
        if model_rows.empty:
            ai_after = ai_scores.get(str(label), ai_scores.get(str(model_key)))
            raw_score = (ai_after or 0.0) * 18
            weight_rows.append(
                {
                    "lottery_type": lottery_type,
                    "created_at": created_at,
                    "model_key": model_key,
                    "model_name": label,
                    "evaluation_count": 0,
                    "average_match": None,
                    "recent5_score": None,
                    "recent10_score": None,
                    "long_score": None,
                    "stability": None,
                    "expected_value": rounded_or_none((ai_after or 0) / max(draw_size, 1), 3) if ai_after is not None else None,
                    "bonus_match_rate": None,
                    "raw_score": rounded_or_none(raw_score, 3),
                    "applied_weight": 1.0,
                    "status": "検証待ち",
                }
            )
            continue
        matches = _numeric_series(model_rows, match_column)
        bonus_matches = _bonus_match_series(model_rows, bonus_column)
        avg_match = float(matches.mean()) if not matches.empty else 0.0
        recent5 = float(matches.tail(5).mean()) if not matches.empty else 0.0
        recent10 = float(matches.tail(10).mean()) if not matches.empty else 0.0
        long_score = avg_match
        std = float(matches.std()) if len(matches) > 1 else 0.0
        stability = _clamp(100.0 - (std / max(draw_size, 1) * 100.0))
        expected_value = float(((matches + bonus_matches * 0.25) / max(draw_size, 1)).mean())
        bonus_rate = float((bonus_matches > 0).mean() * 100.0) if len(bonus_matches) else 0.0
        ai_after = ai_scores.get(str(label), ai_scores.get(str(model_key), 0.0))
        raw_score = (
            avg_match * 30
            + recent5 * 18
            + recent10 * 10
            + long_score * 8
            + stability / 100 * 10
            + expected_value * 18
            + bonus_rate / 100 * 6
            + ai_after * 6
        )
        weight_rows.append(
            {
                "lottery_type": lottery_type,
                "created_at": created_at,
                "model_key": model_key,
                "model_name": label,
                "evaluation_count": int(len(model_rows)),
                "average_match": rounded_or_none(avg_match, 3),
                "recent5_score": rounded_or_none(recent5, 3),
                "recent10_score": rounded_or_none(recent10, 3),
                "long_score": rounded_or_none(long_score, 3),
                "stability": rounded_or_none(stability, 1),
                "expected_value": rounded_or_none(expected_value, 3),
                "bonus_match_rate": rounded_or_none(bonus_rate, 1),
                "raw_score": rounded_or_none(raw_score, 3),
                "applied_weight": 1.0,
                "status": "評価中",
            }
        )

    scored = [row for row in weight_rows if safe_int(row["evaluation_count"]) > 0 or _safe_float(row["raw_score"], 0.0) > 0]
    raw_values = [_safe_float(row["raw_score"], 0.0) for row in scored]
    low = min(raw_values) if raw_values else 0.0
    high = max(raw_values) if raw_values else 0.0
    for row in weight_rows:
        if not scored or high == low:
            weight = 1.0 if safe_int(row["evaluation_count"]) > 0 else 0.9
        else:
            normalized = (_safe_float(row["raw_score"], 0.0) - low) / (high - low)
            weight = 0.65 + normalized * 0.95
        if safe_int(row["evaluation_count"]) == 0 and _safe_float(row["raw_score"], 0.0) == 0:
            weight = 0.85
        weight = _clamp(weight, 0.45, 1.75)
        row["applied_weight"] = rounded_or_none(weight, 3)
        if row["status"] != "検証待ち":
            if weight >= 1.22:
                row["status"] = "強める"
            elif weight <= 0.82:
                row["status"] = "弱める"
            else:
                row["status"] = "標準"
    return pd.DataFrame(weight_rows, columns=MODEL_WEIGHT_HISTORY_COLUMNS)


def build_weighted_ensemble_score_map(number_rows, bonus_rows, number_max, draw_size, target_round=0, reports=None, model_history=None, lottery_type=""):
    rows = clean_number_rows(number_rows, number_max)
    bonus_rows = clean_number_rows(bonus_rows or [], number_max)
    weight_df = build_model_performance_weights(reports, draw_size, lottery_type, model_history)
    weight_by_key = {
        str(row["model_key"]): _safe_float(row["applied_weight"], 1.0)
        for _, row in weight_df.iterrows()
    }
    combined = {number: 0.0 for number in range(1, number_max + 1)}
    model_score_maps = {}
    for model_key in ARL_MODEL_LABELS:
        raw_scores = build_model_scores(rows, model_key, number_max, draw_size, target_round, bonus_rows)
        normalized = normalize_scores(raw_scores, number_max)
        model_score_maps[model_key] = normalized
        weight = weight_by_key.get(model_key, 1.0)
        for number, score in normalized.items():
            combined[number] += score * weight
    return combined, weight_df, model_score_maps


def _history_sum_stats(number_rows, draw_size, number_max, precleaned=False):
    rows = number_rows if precleaned else clean_number_rows(number_rows, number_max)
    sums = [sum(row) for row in rows]
    if not sums:
        average = (number_max + 1) / 2 * draw_size
        return average, max(18.0, average * 0.18)
    average = sum(sums) / len(sums)
    if len(sums) <= 1:
        return average, max(18.0, average * 0.18)
    variance = sum((value - average) ** 2 for value in sums) / (len(sums) - 1)
    return average, max(variance ** 0.5, 1.0)


def _balance_score(numbers, number_rows, number_max, draw_size, sum_stats=None):
    summary = game_balance(numbers, number_max)
    target_odd = draw_size / 2
    target_low = draw_size / 2
    sum_average, sum_std = sum_stats or _history_sum_stats(number_rows, draw_size, number_max)
    odd_penalty = abs(summary["odd"] - target_odd) * 11
    low_penalty = abs(summary["low"] - target_low) * 9
    sum_penalty = min(abs(summary["sum"] - sum_average) / max(sum_std, 1) * 14, 28)
    consecutive_penalty = max(0, summary["consecutive"] - 1) * 12
    tail_penalty = max(0, summary["same_tail"] - 2) * 8
    return _clamp(100 - odd_penalty - low_penalty - sum_penalty - consecutive_penalty - tail_penalty)


def evaluate_low_popularity(numbers, number_rows, number_max, draw_size, precleaned=False, sum_stats=None, similarity_rows=None):
    numbers = sorted({safe_int(number) for number in numbers if 1 <= safe_int(number) <= number_max})
    rows = number_rows if precleaned else clean_number_rows(number_rows, number_max)
    if len(numbers) != draw_size:
        return {
            "low_popularity_score": 0.0,
            "birthday_bias_score": 0.0,
            "consecutive_score": 0.0,
            "regularity_score": 0.0,
            "last_digit_score": 0.0,
            "sum_balance_score": 0.0,
            "past_similarity_score": 0.0,
            "visual_neatness_score": 0.0,
            "past_similarity": 100.0,
            "risk_level": "高",
        }
    birthday_count = sum(number <= 31 for number in numbers)
    over_birthday = draw_size - birthday_count
    birthday_heavy = max(0, birthday_count - max(draw_size - 2, 1))
    birthday_score = _clamp(62 + over_birthday * (38 / max(draw_size, 1)) - birthday_heavy * 11)

    consecutive = count_consecutive_pairs(numbers)
    consecutive_score = _clamp(100 - max(0, consecutive - 1) * 28)

    gaps = [right - left for left, right in zip(numbers, numbers[1:])]
    repeated_gap_count = max(Counter(gaps).values()) if gaps else 0
    regular_penalty = 0
    if repeated_gap_count >= max(2, draw_size - 3):
        regular_penalty += 26
    regular_penalty += sum(12 for gap in gaps if gap in (5, 7))
    if len(set(gaps)) == 1 and gaps:
        regular_penalty += 32
    regularity_score = _clamp(100 - regular_penalty)

    tail_counts = Counter(number % 10 for number in numbers)
    max_tail = max(tail_counts.values()) if tail_counts else 0
    last_digit_score = _clamp(100 - max(0, max_tail - 2) * 24)

    sum_average, sum_std = sum_stats or _history_sum_stats(rows, draw_size, number_max, precleaned=True)
    z_score = abs(sum(numbers) - sum_average) / max(sum_std, 1)
    sum_balance_score = _clamp(100 - max(0.0, z_score - 0.8) * 28)

    similarity_rows = similarity_rows if similarity_rows is not None else (rows[-300:] if len(rows) > 300 else rows)
    max_overlap = max((len(set(numbers) & set(row)) for row in similarity_rows), default=0)
    past_similarity = max_overlap / max(draw_size, 1) * 100
    past_similarity_score = _clamp(100 - past_similarity * 0.92)

    tens_counts = Counter(number // 10 for number in numbers)
    neat_penalty = max(0, max(tens_counts.values()) - 3) * 16 if tens_counts else 0
    if all(number % 5 == 0 for number in numbers[: max(2, min(3, len(numbers)))]) and len(numbers) >= 3:
        neat_penalty += 18
    if len(set(gaps)) <= 2 and len(gaps) >= 4:
        neat_penalty += 20
    visual_neatness_score = _clamp(100 - neat_penalty)

    total_score = (
        birthday_score * 0.16
        + consecutive_score * 0.14
        + regularity_score * 0.15
        + last_digit_score * 0.12
        + sum_balance_score * 0.17
        + past_similarity_score * 0.16
        + visual_neatness_score * 0.10
    )
    if total_score >= 72 and z_score <= 1.8 and consecutive <= 1:
        risk_level = "低"
    elif total_score >= 52 and z_score <= 2.5:
        risk_level = "中"
    else:
        risk_level = "高"
    return {
        "low_popularity_score": rounded_or_none(total_score, 3),
        "birthday_bias_score": rounded_or_none(birthday_score, 3),
        "consecutive_score": rounded_or_none(consecutive_score, 3),
        "regularity_score": rounded_or_none(regularity_score, 3),
        "last_digit_score": rounded_or_none(last_digit_score, 3),
        "sum_balance_score": rounded_or_none(sum_balance_score, 3),
        "past_similarity_score": rounded_or_none(past_similarity_score, 3),
        "visual_neatness_score": rounded_or_none(visual_neatness_score, 3),
        "past_similarity": rounded_or_none(past_similarity, 1),
        "risk_level": risk_level,
    }


def _top_models_for_numbers(numbers, model_score_maps, weight_df, limit=4):
    rows = []
    weight_by_key = {
        str(row["model_key"]): _safe_float(row["applied_weight"], 1.0)
        for _, row in weight_df.iterrows()
    } if weight_df is not None and not weight_df.empty else {}
    for model_key, label in ARL_MODEL_LABELS.items():
        score_map = model_score_maps.get(model_key, {})
        average_score = sum(score_map.get(number, 0.0) for number in numbers) / max(len(numbers), 1)
        rows.append((average_score * weight_by_key.get(model_key, 1.0), label))
    return [label for _, label in sorted(rows, reverse=True)[:limit]]


def _role_reason(role):
    if role == "本命・安定型":
        return "過去成績の重みが高いモデルを中心に、奇数偶数・高低・合計値の安定性を優先"
    if role == "直近トレンド型":
        return "直近傾向、マルコフ系の移り変わり、ホット傾向を重めに評価"
    return "予測スコアを維持しつつ、誕生日数字偏りや規則的な並びを避ける低人気評価を加点"


def build_high_prize_ticket_strategy(number_rows, bonus_rows, number_max, draw_size, target_round=0, reports=None, model_history=None, lottery_type="", candidate_pool_limit=None):
    rows = clean_number_rows(number_rows, number_max)
    bonus_rows = clean_number_rows(bonus_rows or [], number_max)
    if not rows:
        return [], build_model_performance_weights(reports, draw_size, lottery_type, model_history)
    sum_stats = _history_sum_stats(rows, draw_size, number_max, precleaned=True)
    similarity_rows = rows[-300:] if len(rows) > 300 else rows
    ensemble_scores, weight_df, model_score_maps = build_weighted_ensemble_score_map(
        rows,
        bonus_rows,
        number_max,
        draw_size,
        target_round,
        reports,
        model_history,
        lottery_type,
    )
    ensemble_norm = normalize_scores(ensemble_scores, number_max)
    hot_norm = normalize_scores(build_model_scores(rows, "hot_analysis", number_max, draw_size, target_round, bonus_rows), number_max)
    cold_norm = normalize_scores(build_model_scores(rows, "cold_analysis", number_max, draw_size, target_round, bonus_rows), number_max)
    markov_norm = normalize_scores(build_model_scores(rows, "markov_chain", number_max, draw_size, target_round, bonus_rows), number_max)

    ranked = sorted(ensemble_scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)
    pool_limit = candidate_pool_limit or max(14, draw_size + 8)
    initial_limit = min(pool_limit, max(12, draw_size + 5))
    pool = [number for number, _ in ranked[:initial_limit]]
    for source in (hot_norm, cold_norm, markov_norm):
        for number, _ in sorted(source.items(), key=lambda item: (item[1], -item[0]), reverse=True)[: draw_size]:
            if number not in pool:
                pool.append(number)
    pool = sorted(pool, key=lambda number: ensemble_scores.get(number, 0.0), reverse=True)[: min(number_max, pool_limit)]
    if len(pool) < draw_size:
        pool = list(range(1, min(number_max, draw_size) + 1))

    candidates = []
    for combo in combinations(sorted(pool), draw_size):
        numbers = tuple(sorted(combo))
        popularity = evaluate_low_popularity(numbers, rows, number_max, draw_size, precleaned=True, sum_stats=sum_stats, similarity_rows=similarity_rows)
        prediction_score = sum(ensemble_norm.get(number, 0.0) for number in numbers) / draw_size * 100
        balance = _balance_score(numbers, rows, number_max, draw_size, sum_stats=sum_stats)
        trend_score = sum((hot_norm.get(number, 0.0) + markov_norm.get(number, 0.0)) / 2 for number in numbers) / draw_size * 100
        cold_score = sum(cold_norm.get(number, 0.0) for number in numbers) / draw_size * 100
        low_popularity = _safe_float(popularity["low_popularity_score"], 0.0)
        stable_score = prediction_score * 0.62 + balance * 0.26 + low_popularity * 0.12
        trend_role_score = prediction_score * 0.50 + trend_score * 0.33 + balance * 0.10 + low_popularity * 0.07
        jackpot_score = prediction_score * 0.38 + low_popularity * 0.40 + cold_score * 0.14 + balance * 0.08
        candidates.append(
            {
                "numbers": numbers,
                "prediction_score": prediction_score,
                "balance_score": balance,
                "trend_score": trend_score,
                "cold_score": cold_score,
                "popularity": popularity,
                "stable_score": stable_score,
                "trend_role_score": trend_role_score,
                "jackpot_score": jackpot_score,
            }
        )
    if not candidates:
        numbers = tuple(sorted(number for number, _ in ranked[:draw_size]))
        popularity = evaluate_low_popularity(numbers, rows, number_max, draw_size, precleaned=True, sum_stats=sum_stats, similarity_rows=similarity_rows)
        candidates.append(
            {
                "numbers": numbers,
                "prediction_score": 0.0,
                "balance_score": 0.0,
                "trend_score": 0.0,
                "cold_score": 0.0,
                "popularity": popularity,
                "stable_score": 0.0,
                "trend_role_score": 0.0,
                "jackpot_score": 0.0,
            }
        )

    role_keys = {
        "本命・安定型": "stable_score",
        "直近トレンド型": "trend_role_score",
        "高額当選狙い・低人気型": "jackpot_score",
    }
    selected = []
    selected_sets = []
    for ticket_no, role in enumerate(HIGH_PRIZE_TICKET_ROLES, start=1):
        sorted_candidates = sorted(candidates, key=lambda row: row[role_keys[role]], reverse=True)
        chosen = None
        for candidate in sorted_candidates:
            candidate_set = set(candidate["numbers"])
            if any(candidate_set == existing for existing in selected_sets):
                continue
            if selected_sets and all(len(candidate_set & existing) > draw_size - 2 for existing in selected_sets):
                continue
            chosen = candidate
            break
        if chosen is None:
            chosen = sorted_candidates[0]
        selected_sets.append(set(chosen["numbers"]))
        models = _top_models_for_numbers(chosen["numbers"], model_score_maps, weight_df, 4)
        popularity = chosen["popularity"]
        selected.append(
            {
                "ticket_no": ticket_no,
                "ticket_role": role,
                "numbers": tuple(chosen["numbers"]),
                "adopted_models": models,
                "selection_reason": _role_reason(role),
                "prediction_score": rounded_or_none(chosen["prediction_score"], 3),
                "expected_score": rounded_or_none(chosen[role_keys[role]], 3),
                "low_popularity_score": rounded_or_none(popularity["low_popularity_score"], 3),
                "past_similarity": rounded_or_none(popularity["past_similarity"], 1),
                "risk_level": popularity["risk_level"],
                "popularity_detail": popularity,
            }
        )
    return selected, weight_df


def ticket_strategy_display_frame(tickets):
    rows = []
    for ticket in tickets or []:
        rows.append(
            {
                "買い目番号": ticket.get("ticket_no"),
                "役割": ticket.get("ticket_role"),
                "買い目": numbers_to_text(ticket.get("numbers", [])),
                "採用モデル": " / ".join(ticket.get("adopted_models", [])),
                "選定理由": ticket.get("selection_reason", ""),
                "予測スコア": ticket.get("prediction_score"),
                "期待スコア": ticket.get("expected_score"),
                "低人気スコア": ticket.get("low_popularity_score"),
                "過去類似度": ticket.get("past_similarity"),
                "リスク評価": ticket.get("risk_level"),
            }
        )
    return pd.DataFrame(rows)


def build_ensemble_prediction_rows(tickets, lottery_type, target_round, prediction_date=None, created_at=None):
    prediction_date = prediction_date or datetime.now().strftime("%Y/%m/%d")
    created_at = created_at or datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    rows = []
    for ticket in tickets or []:
        rows.append(
            {
                "lottery_type": lottery_type,
                "target_draw": safe_int(target_round),
                "prediction_date": prediction_date,
                "ticket_no": safe_int(ticket.get("ticket_no")),
                "ticket_role": ticket.get("ticket_role", ""),
                "numbers": numbers_to_text(ticket.get("numbers", [])),
                "adopted_models": " / ".join(ticket.get("adopted_models", [])),
                "selection_reason": ticket.get("selection_reason", ""),
                "prediction_score": ticket.get("prediction_score", ""),
                "expected_score": ticket.get("expected_score", ""),
                "low_popularity_score": ticket.get("low_popularity_score", ""),
                "past_similarity": ticket.get("past_similarity", ""),
                "risk_level": ticket.get("risk_level", ""),
                "created_at": created_at,
            }
        )
    return rows


def build_ticket_strategy_rows(tickets, lottery_type, target_round, prediction_date=None, created_at=None):
    prediction_rows = build_ensemble_prediction_rows(tickets, lottery_type, target_round, prediction_date, created_at)
    return [{column: row.get(column, "") for column in TICKET_STRATEGY_COLUMNS} for row in prediction_rows]


def build_popularity_score_rows(tickets, lottery_type, target_round, prediction_date=None, created_at=None):
    prediction_date = prediction_date or datetime.now().strftime("%Y/%m/%d")
    created_at = created_at or datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    rows = []
    for ticket in tickets or []:
        detail = ticket.get("popularity_detail", {}) or {}
        rows.append(
            {
                "lottery_type": lottery_type,
                "target_draw": safe_int(target_round),
                "prediction_date": prediction_date,
                "ticket_no": safe_int(ticket.get("ticket_no")),
                "numbers": numbers_to_text(ticket.get("numbers", [])),
                "low_popularity_score": detail.get("low_popularity_score", ticket.get("low_popularity_score", "")),
                "birthday_bias_score": detail.get("birthday_bias_score", ""),
                "consecutive_score": detail.get("consecutive_score", ""),
                "regularity_score": detail.get("regularity_score", ""),
                "last_digit_score": detail.get("last_digit_score", ""),
                "sum_balance_score": detail.get("sum_balance_score", ""),
                "past_similarity_score": detail.get("past_similarity_score", ""),
                "visual_neatness_score": detail.get("visual_neatness_score", ""),
                "risk_level": ticket.get("risk_level", detail.get("risk_level", "")),
                "created_at": created_at,
            }
        )
    return rows


def build_continuous_win_research_rows(tickets, lottery_type, target_round, actual_numbers=None, bonus_numbers=None, prediction_date=None, created_at=None):
    prediction_date = prediction_date or datetime.now().strftime("%Y/%m/%d")
    created_at = created_at or datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    actual_numbers = sorted(actual_numbers or [])
    bonus_numbers = sorted(bonus_numbers or [])
    rows = []
    for ticket in tickets or []:
        numbers = sorted(ticket.get("numbers", []))
        hit_numbers = sorted(set(numbers) & set(actual_numbers)) if actual_numbers else []
        missed_numbers = sorted(set(numbers) - set(actual_numbers)) if actual_numbers else []
        bonus_hit = sorted(set(numbers) & set(bonus_numbers)) if bonus_numbers else []
        if actual_numbers:
            failure_reason = "一致数と外れた数字を次回検証へ回す"
            improvement_plan = "弱いモデルを下げ、近かったモデルと不足条件を次回重みに反映"
        else:
            failure_reason = "結果待ち"
            improvement_plan = "抽選結果登録後に自動検証"
        rows.append(
            {
                "prediction_date": prediction_date,
                "target_draw": safe_int(target_round),
                "lottery_type": lottery_type,
                "numbers": numbers_to_text(numbers),
                "ticket_role": ticket.get("ticket_role", ""),
                "adopted_models": " / ".join(ticket.get("adopted_models", [])),
                "prediction_score": ticket.get("prediction_score", ""),
                "low_popularity_score": ticket.get("low_popularity_score", ""),
                "actual_numbers": numbers_to_text(actual_numbers) if actual_numbers else "",
                "bonus_numbers": numbers_to_text(bonus_numbers) if bonus_numbers else "",
                "matched_count": len(hit_numbers) if actual_numbers else "",
                "bonus_matched_count": len(bonus_hit) if bonus_numbers else "",
                "missed_numbers": numbers_to_text(missed_numbers) if missed_numbers else "",
                "hit_numbers": numbers_to_text(hit_numbers) if hit_numbers else "",
                "expected_value": rounded_or_none((_safe_float(ticket.get("expected_score"), 0.0) / 100.0), 3),
                "failure_reason": failure_reason,
                "improvement_plan": improvement_plan,
                "next_hypothesis": f"{ticket.get('ticket_role', '')}の重みと低人気条件を次回も検証",
                "created_at": created_at,
            }
        )
    return rows


def build_high_prize_backtest_summary(number_rows, bonus_rows, number_max, draw_size, lottery_type="", periods=None, reports=None, model_history=None, min_training_rounds=20):
    rows = clean_number_rows(number_rows, number_max)
    bonus_rows = clean_number_rows(bonus_rows or [], number_max)
    if periods is None:
        periods = [30, 50, 100, "all"]
    if len(rows) <= min_training_rounds:
        return pd.DataFrame(columns=BACKTEST_SUMMARY_COLUMNS)
    created_at = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    detail_rows = []
    model_defs = [(model_key, label, "既存分析モデル") for model_key, label in ARL_MODEL_LABELS.items()]
    special_defs = [
        ("ensemble", "アンサンブルモデル", "強化モデル"),
        ("high_prize", "高額当選狙いモデル", "強化モデル"),
        ("continuous_win", "連続当選狙いモデル", "強化モデル"),
    ]
    for period in periods:
        period_label = str(period)
        start_index = min_training_rounds if period == "all" else max(min_training_rounds, len(rows) - int(period))
        for target_index in range(start_index, len(rows)):
            training_rows = rows[:target_index]
            training_bonus = bonus_rows[:target_index] if bonus_rows else []
            actual = set(rows[target_index])
            bonus_actual = set(bonus_rows[target_index]) if target_index < len(bonus_rows) else set()
            target_round = target_index + 1
            special_tickets, _ = build_high_prize_ticket_strategy(
                training_rows,
                training_bonus,
                number_max,
                draw_size,
                target_round,
                reports,
                model_history,
                lottery_type,
                candidate_pool_limit=max(draw_size + 5, 12),
            )
            special_by_role = {ticket["ticket_role"]: ticket["numbers"] for ticket in special_tickets}
            for model_key, model_name, model_type in model_defs:
                scores = build_model_scores(training_rows, model_key, number_max, draw_size, target_round, training_bonus)
                predicted = model_candidate_numbers(scores, draw_size)
                matched = len(set(predicted) & actual)
                bonus_hit = len(set(predicted) & bonus_actual)
                detail_rows.append(
                    {
                        "period": period_label,
                        "model_name": model_name,
                        "model_type": model_type if model_key != "random_baseline" else "ランダム予測モデル",
                        "matched": matched,
                        "bonus_hit": bonus_hit,
                    }
                )
            for special_key, model_name, model_type in special_defs:
                if special_key == "ensemble":
                    predicted = special_by_role.get("本命・安定型", [])
                elif special_key == "high_prize":
                    predicted = special_by_role.get("高額当選狙い・低人気型", [])
                else:
                    predicted = special_by_role.get("直近トレンド型", [])
                matched = len(set(predicted) & actual)
                bonus_hit = len(set(predicted) & bonus_actual)
                detail_rows.append(
                    {
                        "period": period_label,
                        "model_name": model_name,
                        "model_type": model_type,
                        "matched": matched,
                        "bonus_hit": bonus_hit,
                    }
                )
    if not detail_rows:
        return pd.DataFrame(columns=BACKTEST_SUMMARY_COLUMNS)
    detail_df = pd.DataFrame(detail_rows)
    summary_rows = []
    for (period, model_name), group in detail_df.groupby(["period", "model_name"], sort=False):
        matches = pd.to_numeric(group["matched"], errors="coerce").fillna(0)
        bonus_hits = pd.to_numeric(group["bonus_hit"], errors="coerce").fillna(0)
        random_group = detail_df[(detail_df["period"] == period) & (detail_df["model_type"] == "ランダム予測モデル")]
        random_avg = float(pd.to_numeric(random_group["matched"], errors="coerce").mean()) if not random_group.empty else 0.0
        std = float(matches.std()) if len(matches) > 1 else 0.0
        stability = _clamp(100.0 - (std / max(draw_size, 1) * 100.0))
        summary_rows.append(
            {
                "lottery_type": lottery_type,
                "period": period,
                "model_name": model_name,
                "model_type": str(group.iloc[0]["model_type"]),
                "evaluated_draws": int(len(group)),
                "average_match": rounded_or_none(float(matches.mean()), 3),
                "max_match": int(matches.max()) if not matches.empty else 0,
                "match3_rate": rounded_or_none(float((matches >= 3).mean() * 100), 1),
                "match4_rate": rounded_or_none(float((matches >= 4).mean() * 100), 1),
                "bonus_match_rate": rounded_or_none(float((bonus_hits > 0).mean() * 100), 1),
                "stability": rounded_or_none(stability, 1),
                "recent_score": rounded_or_none(float(matches.tail(5).mean()), 3),
                "long_score": rounded_or_none(float(matches.mean()), 3),
                "random_delta": rounded_or_none(float(matches.mean()) - random_avg, 3),
                "expected_value": rounded_or_none(float(((matches + bonus_hits * 0.25) / max(draw_size, 1)).mean()), 3),
                "created_at": created_at,
            }
        )
    return pd.DataFrame(summary_rows, columns=BACKTEST_SUMMARY_COLUMNS).sort_values(
        ["period", "expected_value", "average_match", "stability"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def build_enhanced_ai_improvement_report(reports, weight_df=None, draw_size=None):
    columns = ["項目", "内容"]
    if reports is None or reports.empty:
        return pd.DataFrame(
            [
                {"項目": "なぜ外れたか", "内容": "検証レポートがまだないため、結果登録後に分析します"},
                {"項目": "次回の仮説", "内容": "予想、結果、検証、改善の履歴を蓄積して重みを調整します"},
            ],
            columns=columns,
        )
    df = reports.copy()
    report_cols = _verification_column_set(df)
    draw_column = report_cols["draw"]
    if draw_column and draw_column in df:
        df[draw_column] = pd.to_numeric(df[draw_column], errors="coerce").fillna(0)
        df = df.sort_values(draw_column, ascending=False)
    latest = df.iloc[0]
    match_count = _safe_float(latest.get(report_cols["match"], 0) if report_cols["match"] else 0)
    failure_text = latest.get(report_cols["failure"], "") if report_cols["failure"] else ""
    improvement_text = latest.get(report_cols["improvement"], "") if report_cols["improvement"] else ""
    hypothesis_text = latest.get(report_cols["hypothesis"], "") if report_cols["hypothesis"] else ""
    if weight_df is not None and not weight_df.empty:
        sorted_weights = weight_df.copy()
        sorted_weights["applied_weight"] = pd.to_numeric(sorted_weights["applied_weight"], errors="coerce").fillna(1.0)
        close_models = sorted_weights.sort_values("applied_weight", ascending=False)["model_name"].head(5).tolist()
        weak_models = sorted_weights.sort_values("applied_weight", ascending=True)["model_name"].head(5).tolist()
        weight_up = sorted_weights[sorted_weights["applied_weight"] >= 1.15]["model_name"].head(5).tolist()
        weight_down = sorted_weights[sorted_weights["applied_weight"] <= 0.85]["model_name"].head(5).tolist()
    else:
        close_models = []
        weak_models = []
        weight_up = []
        weight_down = []
    if draw_size and match_count >= draw_size:
        miss_reason = "全一致のため、次回は過学習しないよう同じ条件を継続検証"
    elif failure_text:
        miss_reason = str(failure_text)
    else:
        miss_reason = "当選番号の高低、合計値、直近トレンドのいずれかが予想条件から外れた可能性を検証"
    rows = [
        {"項目": "なぜ外れたか", "内容": miss_reason},
        {"項目": "弱かったモデル", "内容": " / ".join(weak_models) if weak_models else "検証待ち"},
        {"項目": "比較的近かったモデル", "内容": " / ".join(close_models) if close_models else "検証待ち"},
        {"項目": "次回重みを上げるモデル", "内容": " / ".join(weight_up) if weight_up else "該当なし"},
        {"項目": "次回重みを下げるモデル", "内容": " / ".join(weight_down) if weight_down else "該当なし"},
        {"項目": "数字選定ルールの改善案", "内容": str(improvement_text or "低人気条件、合計値帯、直近トレンドの重みを分けて検証")},
        {"項目": "次回の仮説", "内容": str(hypothesis_text or "成績上位モデルを軸に、低人気スコアを補助指標として使う")},
    ]
    return pd.DataFrame(rows, columns=columns)


def build_condition_success_table(reports, number_max, match_column="本数字一致数", success_threshold=3):
    if reports is None or reports.empty or "予想番号" not in reports or match_column not in reports:
        return pd.DataFrame(columns=["条件", "対象件数", "成功件数", "成功率"])
    rows = []
    checks = {
        "32以上を含む": lambda nums: any(number >= 32 for number in nums),
        "32以上2個以上": lambda nums: sum(number >= 32 for number in nums) >= 2,
        "連番あり": lambda nums: count_consecutive_pairs(nums) >= 1,
        "同末尾あり": lambda nums: any(count >= 2 for count in Counter(number % 10 for number in nums).values()),
        "ボーナス数字条件": lambda nums: True,
        "4・9・13候補": lambda nums: any(number in {4, 9, 13} for number in nums),
    }
    df = reports.copy()
    df[match_column] = pd.to_numeric(df[match_column], errors="coerce").fillna(0)
    for label, check in checks.items():
        target = df[df["予想番号"].map(lambda value: check(parse_numbers(value, number_max)))]
        total = len(target)
        success = int((target[match_column] >= success_threshold).sum()) if total else 0
        rows.append(
            {
                "条件": label,
                "対象件数": total,
                "成功件数": success,
                "成功率": f"{round(success / total * 100, 1)}%" if total else "0%",
            }
        )
    return pd.DataFrame(rows)


def build_ai_improvement_summary(reports):
    if reports is None or reports.empty:
        return {
            "予想": "-",
            "結果": "-",
            "一致数": "-",
            "的中要因": "検証レポートがまだありません。",
            "外れ要因": "検証後に自動作成します。",
            "改善案": "予想、結果、検証を蓄積します。",
            "次回仮説": "履歴分析後に生成します。",
        }
    sorted_reports = reports.copy()
    if "開催回" in sorted_reports:
        sorted_reports["開催回"] = pd.to_numeric(sorted_reports["開催回"], errors="coerce").fillna(0)
        sorted_reports = sorted_reports.sort_values("開催回", ascending=False)
    latest = sorted_reports.iloc[0]
    return {
        "予想": latest.get("予想番号", "-"),
        "結果": latest.get("実際の当選番号", "-"),
        "一致数": latest.get("本数字一致数", "-"),
        "的中要因": latest.get("的中要因", "一致数字ごとの由来を検証中"),
        "外れ要因": latest.get("失敗要因", "-"),
        "改善案": latest.get("改善案", "-"),
        "次回仮説": latest.get("次回の仮説", "-"),
    }


def build_research_flow_table():
    return pd.DataFrame(
        [
            {"順番": 1, "工程": "当選番号追加"},
            {"順番": 2, "工程": "結果分析"},
            {"順番": 3, "工程": "反省履歴保存"},
            {"順番": 4, "工程": "履歴分析"},
            {"順番": 5, "工程": "モデル評価"},
            {"順番": 6, "工程": "改善条件抽出"},
            {"順番": 7, "工程": "次回予想生成"},
        ]
    )


def extract_video_hypothesis(video_name, transcript_text, result_text="未バックテスト"):
    text = str(transcript_text or "").strip()
    law_parts = []
    keywords = [
        ("高数字", "32以上または高数字帯の採用条件を検証"),
        ("連番", "連番を除外せず許容条件として検証"),
        ("末尾", "同末尾・下一桁偏りを条件化"),
        ("ボーナス", "ボーナス数字から本数字昇格を検証"),
        ("奇数", "奇数偶数バランスを検証"),
        ("偶数", "奇数偶数バランスを検証"),
        ("合計", "合計値レンジを検証"),
        ("ホット", "ホット数字の継続を検証"),
        ("コールド", "コールド数字の復活を検証"),
    ]
    for keyword, law in keywords:
        if keyword in text and law not in law_parts:
            law_parts.append(law)
    if not law_parts:
        law_parts.append("文字起こし内の数字傾向を仮説化してバックテスト")
    hypothesis = f"{video_name or '動画仮説'}: " + " / ".join(law_parts[:3])
    return {
        "動画名": video_name or "未入力動画",
        "仮説": hypothesis,
        "法則抽出": " / ".join(law_parts),
        "成績": result_text or "未バックテスト",
        "採用可否": "要検証",
        "保存日時": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
    }
