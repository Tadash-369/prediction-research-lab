from collections import Counter, OrderedDict
from datetime import datetime
import random

import pandas as pd


PROJECT_JAPANESE_NAME = "分析研究所"
PROJECT_ENGLISH_NAME = "Prediction Research Lab"
PROJECT_SHORT_NAME = "PRL"

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
    ids = set(incoming["予想ID"].astype(str))
    existing = existing[~existing["予想ID"].astype(str).isin(ids)]
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
    ids = set(incoming["予想ID"].astype(str))
    existing = existing[~existing["予想ID"].astype(str).isin(ids)]
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


def build_model_dashboard(reports, match_column="本数字一致数", model_column="使用モデル", draw_size=None):
    if reports is None or reports.empty or match_column not in reports:
        return pd.DataFrame(columns=["モデル", "平均一致数", "安定性", "直近成績", "長期成績", "期待値", "最大一致数", "全期間"])
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
    rows = []
    for model, model_rows in df.groupby(model_column):
        match_std = float(model_rows[match_column].std()) if len(model_rows) > 1 else 0.0
        stability = round(max(0.0, 100.0 - match_std * 25), 1)
        rows.append(
            {
                "モデル": model,
                "平均一致数": round(float(model_rows[match_column].mean()), 3),
                "安定性": stability,
                "直近成績": round(float(model_rows.tail(10)[match_column].mean()), 3),
                "長期成績": round(float(model_rows[match_column].mean()), 3),
                "期待値": round(float(model_rows["期待値"].mean()), 3),
                "最大一致数": int(model_rows[match_column].max()),
                "全期間": int(len(model_rows)),
            }
        )
    return pd.DataFrame(rows).sort_values(["平均一致数", "安定性", "期待値"], ascending=False)


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
