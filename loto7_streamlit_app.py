from datetime import date, datetime
from itertools import combinations
from pathlib import Path

import pandas as pd
import streamlit as st

from arl_research_engine import (
    ARL_MODEL_LABELS,
    CONTRIBUTION_COLUMNS,
    RESEARCH_CYCLE_COLUMNS,
    VIDEO_HYPOTHESIS_COLUMNS,
    add_verification_metrics,
    build_ai_improvement_summary,
    build_condition_success_table,
    build_contribution_ranking,
    build_contribution_rows,
    build_research_cycle_rows,
    build_effective_conditions,
    build_hit_factor_summary,
    build_missing_excess_conditions,
    build_model_dashboard,
    build_model_scores,
    build_model_support_map,
    build_research_flow_table,
    extract_video_hypothesis,
    format_contribution_detail,
    merge_contribution_rows,
    merge_research_cycle_rows,
)


BASE_DIR = Path(__file__).resolve().parent
RESULTS_CSV = BASE_DIR / "loto7.csv"
PREDICTIONS_CSV = BASE_DIR / "loto7_predictions.csv"
OFFICIAL_RESULTS_CSV = BASE_DIR / "loto7_results.csv"
VERIFICATION_REPORTS_CSV = BASE_DIR / "loto7_verification_reports.csv"
MODEL_SETTINGS_CSV = BASE_DIR / "loto7_model_settings.csv"
CONTRIBUTIONS_CSV = BASE_DIR / "loto7_model_contributions.csv"
RESEARCH_CYCLES_CSV = BASE_DIR / "loto7_research_cycles.csv"
VIDEO_HYPOTHESES_CSV = BASE_DIR / "video_hypotheses.csv"

NUMBER_COLUMNS = ["第1数字", "第2数字", "第3数字", "第4数字", "第5数字", "第6数字", "第7数字"]
BONUS_COLUMNS = ["BONUS数字1", "BONUS数字2"]
PREDICTION_COLUMNS = ["予想ID", "開催回", "予想日", "候補番号", "予想番号", "使用モデル", "予想理由", "保存日時"]
OFFICIAL_RESULT_COLUMNS = ["開催回", "抽せん日", "本数字", "ボーナス数字", "登録元", "保存日時"]
VERIFICATION_COLUMNS = [
    "予想ID",
    "開催回",
    "検証日",
    "使用モデル",
    "予想番号",
    "実際の当選番号",
    "ボーナス数字",
    "本数字一致数",
    "的中率",
    "ボーナス一致数",
    "勝率",
    "期待値",
    "等級判定",
    "近接一致数",
    "合計値差",
    "奇数偶数差",
    "高低差",
    "外れた数字の差分",
    "奇数偶数バランスのズレ",
    "高低バランスのズレ",
    "合計値のズレ",
    "連番の有無",
    "下一桁の偏り",
    "ペア一致",
    "的中要因",
    "有効条件",
    "足りなかった条件",
    "過剰だった条件",
    "モデル貢献度詳細",
    "失敗要因",
    "逆算分析",
    "改善案",
    "次回の仮説",
]
MODEL_LABELS = dict(ARL_MODEL_LABELS)
MODEL_SETTING_COLUMNS = ["設定名", "モデルキー", "モデル名", "根拠", "更新日時"]
AI_IMPROVEMENT_CONTEXT = {
    "actual_numbers": [11, 14, 17, 23, 30, 35, 37],
    "hit_numbers": [17, 30, 35],
    "near_pairs": [(15, 14), (22, 23), (34, 35)],
    "low_max": 2,
    "mid_high_min": 3,
}


st.set_page_config(page_title="ロト7分析", layout="wide")
st.title("ロト7 分析ダッシュボード")


def read_csv(path, columns=None):
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def save_csv(df, path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def model_key_from_name(model_name):
    for key, name in MODEL_LABELS.items():
        if name == model_name:
            return key
    return None


def read_active_model_setting():
    settings = read_csv(MODEL_SETTINGS_CSV, MODEL_SETTING_COLUMNS)
    if settings.empty:
        return None
    active = settings[settings["設定名"] == "active_loto7_prediction"]
    if active.empty:
        return None
    row = active.tail(1).iloc[0]
    return {
        "model_key": str(row["モデルキー"]),
        "model_name": str(row["モデル名"]),
        "reason": str(row["根拠"]),
    }


def save_active_model_setting(model_key, reason):
    model_name = MODEL_LABELS.get(model_key, model_key)
    settings = read_csv(MODEL_SETTINGS_CSV, MODEL_SETTING_COLUMNS)
    settings = settings[settings["設定名"] != "active_loto7_prediction"] if not settings.empty else settings
    row = {
        "設定名": "active_loto7_prediction",
        "モデルキー": model_key,
        "モデル名": model_name,
        "根拠": reason,
        "更新日時": now_text(),
    }
    settings = pd.concat([settings, pd.DataFrame([row])], ignore_index=True)
    save_csv(settings, MODEL_SETTINGS_CSV, MODEL_SETTING_COLUMNS)


def to_int(value, default=0):
    try:
        return int(str(value).replace(",", "").strip())
    except Exception:
        return default


def now_text():
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def today_text():
    return date.today().strftime("%Y/%m/%d")


def numbers_to_text(numbers):
    return "-".join(f"{int(number):02d}" for number in sorted(numbers))


def parse_number_text(value):
    numbers = []
    for part in str(value).replace(",", "-").replace(" ", "-").split("-"):
        number = to_int(part, None)
        if number and 1 <= number <= 37:
            numbers.append(number)
    return sorted(numbers)


def row_numbers(row):
    return sorted(to_int(row[column]) for column in NUMBER_COLUMNS)


def row_bonus_numbers(row):
    return sorted(to_int(row[column]) for column in BONUS_COLUMNS)


def merge_official_results(results):
    official = read_csv(OFFICIAL_RESULTS_CSV, OFFICIAL_RESULT_COLUMNS)
    if official.empty:
        return results

    merged = results.copy()
    if merged.empty:
        merged = pd.DataFrame(columns=results.columns if len(results.columns) else ["開催回", "日付", *NUMBER_COLUMNS, *BONUS_COLUMNS])

    merged["開催回"] = merged["開催回"].map(to_int)
    existing_rounds = set(merged["開催回"].tolist())
    rows = []
    for _, row in official.iterrows():
        round_no = to_int(row["開催回"])
        main_numbers = parse_number_text(row["本数字"])
        bonus_numbers = parse_number_text(row["ボーナス数字"])
        if round_no in existing_rounds or len(main_numbers) != 7 or len(bonus_numbers) != 2:
            continue
        result_row = {
            "開催回": round_no,
            "日付": row["抽せん日"],
            "第1数字": main_numbers[0],
            "第2数字": main_numbers[1],
            "第3数字": main_numbers[2],
            "第4数字": main_numbers[3],
            "第5数字": main_numbers[4],
            "第6数字": main_numbers[5],
            "第7数字": main_numbers[6],
            "BONUS数字1": bonus_numbers[0],
            "BONUS数字2": bonus_numbers[1],
            "キャリーオーバー": "",
        }
        rows.append(result_row)

    if rows:
        merged = pd.concat([merged, pd.DataFrame(rows)], ignore_index=True)
    merged["開催回"] = merged["開催回"].map(to_int)
    return merged.sort_values("開催回")


def count_consecutive_pairs(numbers):
    ordered = sorted(numbers)
    return sum(1 for left, right in zip(ordered, ordered[1:]) if right - left == 1)


def balance_summary(numbers):
    odd = sum(number % 2 for number in numbers)
    low = sum(number <= 18 for number in numbers)
    return {
        "odd": odd,
        "even": len(numbers) - odd,
        "low": low,
        "high": len(numbers) - low,
        "sum": sum(numbers),
        "consecutive": count_consecutive_pairs(numbers),
        "last_digits": [number % 10 for number in numbers],
    }


def normalize_score_map(score_map):
    values = list(score_map.values())
    if not values:
        return {number: 0.0 for number in range(1, 38)}
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return {number: 0.0 for number in score_map}
    return {number: (value - min_value) / (max_value - min_value) for number, value in score_map.items()}


def build_near_correction_scores():
    scores = {number: 0.0 for number in range(1, 38)}
    anchors = set(AI_IMPROVEMENT_CONTEXT["actual_numbers"] + AI_IMPROVEMENT_CONTEXT["hit_numbers"])
    for predicted, actual in AI_IMPROVEMENT_CONTEXT["near_pairs"]:
        anchors.add(predicted)
        anchors.add(actual)
    for number in range(1, 38):
        for anchor in anchors:
            distance = abs(number - anchor)
            if distance == 0:
                scores[number] += 1.0
            elif distance == 1:
                scores[number] += 0.75
            elif distance == 2:
                scores[number] += 0.4
    return scores


def build_high_band_scores():
    scores = {number: 0.0 for number in range(1, 38)}
    for number in range(1, 38):
        if 28 <= number <= 37:
            scores[number] = 1.0
        elif 20 <= number <= 27:
            scores[number] = 0.65
        elif 11 <= number <= 19:
            scores[number] = 0.35
    return scores


def build_historical_scores(results, model_key, target_round):
    if results.empty:
        return {number: 0.0 for number in range(1, 38)}

    history = results.sort_values("開催回").copy()
    number_rows = [row_numbers(row) for _, row in history.iterrows()]
    bonus_rows = [row_bonus_numbers(row) for _, row in history.iterrows()]
    return build_model_scores(number_rows, model_key, 37, 7, target_round, bonus_rows)


def historical_context_before_round(round_no):
    history = merge_official_results(read_csv(RESULTS_CSV))
    if history.empty:
        return [], []
    history = history.copy()
    history["開催回"] = history["開催回"].map(to_int)
    history = history[history["開催回"] < int(round_no)].sort_values("開催回")
    number_rows = [row_numbers(row) for _, row in history.iterrows()]
    bonus_rows = [row_bonus_numbers(row) for _, row in history.iterrows()]
    return number_rows, bonus_rows


def build_support_map_for_prediction(predicted, round_no, selected_model=None):
    number_rows, bonus_rows = historical_context_before_round(round_no)
    return build_model_support_map(
        predicted,
        number_rows,
        number_max=37,
        draw_size=7,
        target_round=round_no,
        bonus_rows=bonus_rows,
        selected_model=selected_model,
    )


def build_candidate_scores(results, model_key="recent_trend"):
    if results.empty:
        return pd.DataFrame(columns=["数字", "スコア"])
    frequency = normalize_score_map(build_historical_scores(results, "frequency_balance", 0))
    recent = normalize_score_map(build_historical_scores(results, "recent_trend", 0))
    overdue = normalize_score_map(build_historical_scores(results, "overdue_interval", 0))
    model = normalize_score_map(build_historical_scores(results, model_key, int(results["開催回"].max()) + 1))
    near = normalize_score_map(build_near_correction_scores())
    high_band = normalize_score_map(build_high_band_scores())

    rows = []
    for number in range(1, 38):
        if model_key == "machine_learning":
            score = (
                frequency[number] * 20
                + recent[number] * 25
                + overdue[number] * 10
                + near[number] * 25
                + high_band[number] * 20
            )
        else:
            score = frequency[number] * 35 + recent[number] * 30 + overdue[number] * 20 + model[number] * 15
        rows.append({"数字": number, "スコア": round(score, 3)})
    return pd.DataFrame(rows).sort_values("スコア", ascending=False)


def build_pick_reason(numbers, target_sum, model_name):
    summary = balance_summary(numbers)
    extra = ""
    if model_name == MODEL_LABELS["machine_learning"]:
        near_count = count_ai_near_signals(numbers)
        high_band_count = sum(1 for number in numbers if 28 <= number <= 37)
        extra = f"近接補正候補{near_count}個、20番台後半〜30番台{high_band_count}個を含め、"
        return (
            f"{model_name}を反映し、奇数{summary['odd']}・偶数{summary['even']}、"
            f"低数字{summary['low']}・高数字{summary['high']}、{extra}"
            f"合計{summary['sum']}の中高数字寄せ構成として検証。連番は{summary['consecutive']}組に抑えています。"
        )
    return (
        f"{model_name}を反映し、奇数{summary['odd']}・偶数{summary['even']}、"
        f"低数字{summary['low']}・高数字{summary['high']}、{extra}合計{summary['sum']}を"
        f"過去平均{target_sum:.1f}付近に調整。連番は{summary['consecutive']}組に抑えています。"
    )


def count_ai_near_signals(numbers):
    anchors = set(AI_IMPROVEMENT_CONTEXT["actual_numbers"] + AI_IMPROVEMENT_CONTEXT["hit_numbers"])
    for predicted, actual in AI_IMPROVEMENT_CONTEXT["near_pairs"]:
        anchors.add(predicted)
        anchors.add(actual)
    return sum(1 for number in numbers if any(abs(number - anchor) <= 2 for anchor in anchors))


def describe_last_digit_bias(numbers):
    counts = pd.Series([number % 10 for number in numbers]).value_counts()
    biased_digits = [f"{int(digit)}が{int(count)}個" for digit, count in counts.items() if count >= 2]
    return " / ".join(biased_digits) if biased_digits else "大きな偏りなし"


def build_reverse_analysis(predicted, actual):
    predicted_set = set(predicted)
    actual_set = set(actual)
    keep_numbers = sorted(predicted_set & actual_set)
    replace_numbers = sorted(predicted_set - actual_set)
    add_numbers = sorted(actual_set - predicted_set)
    replace_text = numbers_to_text(replace_numbers) if replace_numbers else "なし"
    add_text = numbers_to_text(add_numbers) if add_numbers else "なし"
    keep_text = numbers_to_text(keep_numbers) if keep_numbers else "なし"
    return f"残す候補: {keep_text} / 入れ替え候補: {replace_text} / 近づける追加候補: {add_text}"


def build_next_hypothesis(predicted, actual):
    matched = sorted(set(predicted) & set(actual))
    actual_summary = balance_summary(actual)
    matched_text = numbers_to_text(matched) if matched else "なし"
    return (
        f"一致数字({matched_text})の周辺±2を候補群として観察し、"
        f"合計{actual_summary['sum']}前後・奇数{actual_summary['odd']}個・低数字{actual_summary['low']}個を次回比較軸にする"
    )


def generate_prediction_picks(results, model_key="recent_trend", pick_count=3, candidate_limit=16):
    if results.empty:
        return []

    results = results.copy()
    results["開催回"] = results["開催回"].map(to_int)
    scores = build_candidate_scores(results, model_key)
    score_map = dict(zip(scores["数字"], scores["スコア"]))
    candidate_numbers = scores.head(candidate_limit)["数字"].tolist()
    sums = results[NUMBER_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    target_sum = float(sums.mean())
    sum_std = float(sums.std()) if len(sums) > 1 else 25.0
    if pd.isna(sum_std) or sum_std <= 0:
        sum_std = 25.0

    candidates = []
    for combo in combinations(candidate_numbers, 7):
        numbers = tuple(sorted(combo))
        summary = balance_summary(numbers)
        if model_key != "non_overlap" and summary["consecutive"] > 1:
            continue
        if model_key == "machine_learning":
            low_count = sum(1 for number in numbers if number <= 18)
            upper_band_count = sum(1 for number in numbers if 28 <= number <= 37)
            mid_high_count = sum(1 for number in numbers if 20 <= number <= 37)
            near_count = count_ai_near_signals(numbers)
            if summary["odd"] not in (3, 4):
                continue
            if low_count > AI_IMPROVEMENT_CONTEXT["low_max"]:
                continue
            if upper_band_count < 2 or mid_high_count < AI_IMPROVEMENT_CONTEXT["mid_high_min"]:
                continue
            if near_count < 3:
                continue
        elif model_key == "non_overlap":
            upper_band_count = sum(32 <= number <= 37 for number in numbers)
            if upper_band_count < 1:
                continue
        elif summary["odd"] not in (3, 4) or summary["low"] not in (3, 4):
            continue
        ranking_score = (
            sum(score_map[number] for number in numbers)
            - abs(summary["sum"] - target_sum) / max(sum_std, 1) * 12
            - (0 if model_key == "non_overlap" else summary["consecutive"] * 4)
        )
        if model_key == "machine_learning":
            ranking_score += count_ai_near_signals(numbers) * 3
            ranking_score += sum(1 for number in numbers if 28 <= number <= 37) * 2
        if model_key == "non_overlap":
            ranking_score += sum(32 <= number <= 37 for number in numbers) * 3
        candidates.append((ranking_score, numbers))

    candidates.sort(reverse=True, key=lambda item: item[0])
    picks = []
    used_numbers = set()
    for _, numbers in candidates:
        if picks and len(set(numbers) & used_numbers) > 4:
            continue
        picks.append({"numbers": numbers, "reason": build_pick_reason(numbers, target_sum, MODEL_LABELS[model_key])})
        used_numbers.update(numbers)
        if len(picks) == pick_count:
            break
    return picks


def run_backtest(results, lookback_rounds=30, min_training_rounds=40):
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()

    history = results.copy()
    history["開催回"] = history["開催回"].map(to_int)
    history = history.sort_values("開催回").reset_index(drop=True)
    if len(history) <= min_training_rounds:
        return pd.DataFrame(), pd.DataFrame()

    target_rows = history.iloc[min_training_rounds:].tail(int(lookback_rounds))
    detail_rows = []
    for _, target_row in target_rows.iterrows():
        target_round = to_int(target_row["開催回"])
        train_history = history[history["開催回"] < target_round]
        official_row = {
            "開催回": target_round,
            "本数字": numbers_to_text(row_numbers(target_row)),
            "ボーナス数字": numbers_to_text(row_bonus_numbers(target_row)),
        }

        for model_key, model_name in MODEL_LABELS.items():
            picks = generate_prediction_picks(train_history, model_key=model_key, candidate_limit=12)
            for index, pick in enumerate(picks, start=1):
                prediction_row = {
                    "予想ID": f"L7BT-{model_key}-{target_round}-{index}",
                    "開催回": target_round,
                    "予想番号": numbers_to_text(pick["numbers"]),
                    "使用モデル": model_name,
                }
                report = build_report_row(prediction_row, official_row)
                detail_rows.append(
                    {
                        "開催回": target_round,
                        "モデル": model_name,
                        "候補番号": index,
                        "予想番号": report["予想番号"],
                        "当選番号": report["実際の当選番号"],
                        "一致数": report["本数字一致数"],
                        "的中率": report["的中率"],
                        "ボーナス一致数": report["ボーナス一致数"],
                        "勝率": report["勝率"],
                        "期待値": report["期待値"],
                        "近接一致数": report["近接一致数"],
                        "合計値差": report["合計値差"],
                        "奇数偶数差": report["奇数偶数差"],
                        "高低差": report["高低差"],
                        "等級判定": report["等級判定"],
                    }
                )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return pd.DataFrame(), detail_df

    summary_rows = []
    for model_name, model_rows in detail_df.groupby("モデル"):
        round_best = model_rows.groupby("開催回")["一致数"].max()
        round_near_best = model_rows.groupby("開催回")["近接一致数"].max()
        match_std = float(model_rows["一致数"].std()) if len(model_rows) > 1 else 0.0
        summary_rows.append(
            {
                "モデル": model_name,
                "検証回数": int(round_best.count()),
                "買い目数": int(len(model_rows)),
                "平均一致数": round(float(model_rows["一致数"].mean()), 3),
                "安定性": round(max(0.0, 100.0 - match_std * 25), 1),
                "直近成績": round(float(model_rows.tail(10)["一致数"].mean()), 3),
                "長期成績": round(float(model_rows["一致数"].mean()), 3),
                "期待値": round(float(model_rows["期待値"].mean()), 3),
                "3口内最高一致平均": round(float(round_best.mean()), 3),
                "3個以上一致率": round(float((model_rows["一致数"] >= 3).mean() * 100), 1),
                "平均近接一致数": round(float(model_rows["近接一致数"].mean()), 3),
                "3口内最高近接平均": round(float(round_near_best.mean()), 3),
                "平均合計値差": round(float(model_rows["合計値差"].abs().mean()), 3),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["3口内最高一致平均", "3個以上一致率", "3口内最高近接平均"],
        ascending=False,
    )
    return summary_df, detail_df.sort_values(["開催回", "モデル", "候補番号"], ascending=[False, True, True])


def run_pre_prediction_research(results):
    if results.empty or len(results) <= 40:
        return pd.DataFrame(), pd.DataFrame(), None
    lookback_rounds = min(5, max(3, len(results) - 40))
    summary_df, detail_df = run_backtest(results, lookback_rounds=lookback_rounds)
    best_key = best_actionable_model_key(summary_df)
    if best_key:
        save_active_model_setting(best_key, f"予想生成前の履歴分析・直近{lookback_rounds}回評価")
    return summary_df, detail_df, best_key


def get_pre_prediction_research(results):
    if results.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    latest_round = int(results["開催回"].map(to_int).max())
    cache_key = f"{latest_round}:{len(results)}"
    if st.session_state.get("loto7_pre_research_key") != cache_key:
        summary_df, detail_df, best_key = run_pre_prediction_research(results)
        st.session_state["loto7_pre_research_key"] = cache_key
        st.session_state["loto7_pre_research_summary"] = summary_df
        st.session_state["loto7_pre_research_detail"] = detail_df
        st.session_state["loto7_pre_research_best"] = best_key
    return (
        st.session_state.get("loto7_pre_research_summary", pd.DataFrame()),
        st.session_state.get("loto7_pre_research_detail", pd.DataFrame()),
        st.session_state.get("loto7_pre_research_best"),
    )


def best_actionable_model_key(summary_df):
    if summary_df.empty:
        return None
    usable = summary_df[summary_df["モデル"] != MODEL_LABELS["random_baseline"]]
    if usable.empty:
        return None
    return model_key_from_name(str(usable.iloc[0]["モデル"]))


def safe_model_id(model_key):
    return str(model_key).replace(" ", "_").replace("/", "_")


def save_prediction_picks(picks, target_round, model_key, model_name):
    predictions = read_csv(PREDICTIONS_CSV, PREDICTION_COLUMNS)
    existing = set()
    if not predictions.empty:
        for _, row in predictions.iterrows():
            existing.add((to_int(row["開催回"]), to_int(row["候補番号"]), str(row["予想番号"]), str(row["使用モデル"])))

    rows = []
    prediction_date = today_text()
    for index, pick in enumerate(picks, start=1):
        number_text = numbers_to_text(pick["numbers"])
        key = (target_round, index, number_text, model_name)
        if key in existing:
            continue
        rows.append(
            {
                "予想ID": f"L7-{target_round}-{prediction_date.replace('/', '')}-{safe_model_id(model_key)}-{index}",
                "開催回": target_round,
                "予想日": prediction_date,
                "候補番号": index,
                "予想番号": number_text,
                "使用モデル": model_name,
                "予想理由": pick["reason"],
                "保存日時": now_text(),
            }
        )
    if rows:
        predictions = pd.concat([predictions, pd.DataFrame(rows)], ignore_index=True)
        predictions["開催回"] = predictions["開催回"].map(to_int)
        predictions["候補番号"] = predictions["候補番号"].map(to_int)
        save_csv(predictions.sort_values(["開催回", "候補番号", "使用モデル", "保存日時"]), PREDICTIONS_CSV, PREDICTION_COLUMNS)
    return len(rows)


def determine_grade(match_count, bonus_count):
    if match_count == 7:
        return "1等"
    if match_count == 6 and bonus_count >= 1:
        return "2等"
    if match_count == 6:
        return "3等"
    if match_count == 5:
        return "4等"
    if match_count == 4:
        return "5等"
    if match_count == 3 and bonus_count >= 1:
        return "6等"
    return "該当なし"


def count_near_matches(predicted, actual):
    actual_only = set(actual) - set(predicted)
    predicted_only = set(predicted) - set(actual)
    used_actual = set()
    near_matches = 0
    for number in predicted_only:
        near = [actual_number for actual_number in actual_only if abs(number - actual_number) <= 2 and actual_number not in used_actual]
        if near:
            closest = min(near, key=lambda actual_number: abs(number - actual_number))
            used_actual.add(closest)
            near_matches += 1
    return near_matches


def build_report_row(prediction_row, official_row, support_map=None):
    predicted = parse_number_text(prediction_row["予想番号"])
    actual = parse_number_text(official_row["本数字"])
    bonus = parse_number_text(official_row["ボーナス数字"])
    support_map = support_map or {
        number: [str(prediction_row.get("使用モデル", "未特定モデル"))] for number in predicted
    }
    predicted_summary = balance_summary(predicted)
    actual_summary = balance_summary(actual)
    matched = sorted(set(predicted) & set(actual))
    missed = sorted(set(predicted) - set(actual))
    actual_missing = sorted(set(actual) - set(predicted))
    differences = []
    for number in missed:
        nearest = min(actual_missing or actual, key=lambda actual_number: abs(number - actual_number))
        differences.append(f"{number:02d}->{nearest:02d}({nearest - number:+d})")
    match_count = len(matched)
    bonus_count = len(set(predicted) & set(bonus))
    grade = determine_grade(match_count, bonus_count)
    hit_rate = round(match_count / max(len(actual), 1) * 100, 1)
    win_rate = 0.0 if grade == "該当なし" else 100.0
    expected_value = round((match_count + bonus_count * 0.25) / 7, 3)
    sum_gap = actual_summary["sum"] - predicted_summary["sum"]
    odd_gap = actual_summary["odd"] - predicted_summary["odd"]
    low_gap = actual_summary["low"] - predicted_summary["low"]
    factors = []
    if match_count <= 2:
        factors.append("中心候補と実際の当選数字の重なりが少ない")
    if abs(sum_gap) >= 20:
        factors.append("合計値の目標帯が実際と離れた")
    if odd_gap:
        factors.append("奇数偶数バランスに差がある")
    if low_gap:
        factors.append("高低バランスに差がある")
    improvements = []
    if abs(sum_gap) >= 10:
        improvements.append("合計値の目標帯を再調整する")
    if odd_gap:
        improvements.append("奇数偶数の配分を1枠ずらして比較する")
    if low_gap:
        improvements.append("低数字と高数字の採用枠を調整する")
    if not improvements:
        improvements.append("現行条件を維持し、候補数字の入れ替え幅を小さく検証する")
    condition_gap = build_missing_excess_conditions(predicted, actual, 37)
    pair_match_count = len(list(combinations(matched, 2))) if match_count >= 2 else 0
    odd_even_gap = f"予想 {predicted_summary['odd']}:{predicted_summary['even']} / 実際 {actual_summary['odd']}:{actual_summary['even']}"
    high_low_gap = f"予想 低{predicted_summary['low']}:高{predicted_summary['high']} / 実際 低{actual_summary['low']}:高{actual_summary['high']}"
    consecutive_text = f"予想{predicted_summary['consecutive']}組 / 実際{actual_summary['consecutive']}組"

    return {
        "予想ID": prediction_row["予想ID"],
        "開催回": to_int(prediction_row["開催回"]),
        "検証日": today_text(),
        "使用モデル": prediction_row.get("使用モデル", ""),
        "予想番号": numbers_to_text(predicted),
        "実際の当選番号": numbers_to_text(actual),
        "ボーナス数字": numbers_to_text(bonus),
        "本数字一致数": match_count,
        "的中率": hit_rate,
        "ボーナス一致数": bonus_count,
        "勝率": win_rate,
        "期待値": expected_value,
        "等級判定": grade,
        "近接一致数": count_near_matches(predicted, actual),
        "合計値差": sum_gap,
        "奇数偶数差": odd_gap,
        "高低差": low_gap,
        "外れた数字の差分": " / ".join(differences) if differences else "なし",
        "奇数偶数バランスのズレ": odd_even_gap,
        "高低バランスのズレ": high_low_gap,
        "合計値のズレ": sum_gap,
        "連番の有無": consecutive_text,
        "下一桁の偏り": f"予想: {describe_last_digit_bias(predicted)} / 実際: {describe_last_digit_bias(actual)}",
        "ペア一致": pair_match_count,
        "的中要因": build_hit_factor_summary(predicted, actual, support_map),
        "有効条件": build_effective_conditions(predicted, actual, support_map),
        "足りなかった条件": condition_gap["missing"],
        "過剰だった条件": condition_gap["excess"],
        "モデル貢献度詳細": format_contribution_detail(predicted, actual, support_map),
        "失敗要因": " / ".join(factors) if factors else "大きな構造差は小さく、候補選択の微差が影響",
        "逆算分析": build_reverse_analysis(predicted, actual),
        "改善案": " / ".join(improvements),
        "次回の仮説": build_next_hypothesis(predicted, actual),
    }


def upsert_official_result(round_no, draw_date, main_numbers, bonus_numbers, source):
    official = read_csv(OFFICIAL_RESULTS_CSV, OFFICIAL_RESULT_COLUMNS)
    row = {
        "開催回": int(round_no),
        "抽せん日": draw_date.strftime("%Y/%m/%d") if hasattr(draw_date, "strftime") else str(draw_date),
        "本数字": numbers_to_text(main_numbers),
        "ボーナス数字": numbers_to_text(bonus_numbers),
        "登録元": source,
        "保存日時": now_text(),
    }
    if official.empty:
        official = pd.DataFrame([row])
    else:
        official["開催回"] = official["開催回"].map(to_int)
        same_round = official["開催回"] == int(round_no)
        if same_round.any():
            for key, value in row.items():
                official.loc[same_round, key] = value
        else:
            official = pd.concat([official, pd.DataFrame([row])], ignore_index=True)
    official["開催回"] = official["開催回"].map(to_int)
    save_csv(official.sort_values("開催回"), OFFICIAL_RESULTS_CSV, OFFICIAL_RESULT_COLUMNS)


def import_latest_result_to_official_log():
    results = merge_official_results(read_csv(RESULTS_CSV))
    if results.empty:
        return "loto7.csv に抽せん結果がありません。"
    results["開催回"] = results["開催回"].map(to_int)
    latest = results.sort_values("開催回").tail(1).iloc[0]
    upsert_official_result(
        to_int(latest["開催回"]),
        latest["日付"],
        row_numbers(latest),
        row_bonus_numbers(latest),
        "loto7.csv",
    )
    verified_count = verify_predictions(to_int(latest["開催回"]))
    return f"第{to_int(latest['開催回'])}回を公式結果ログへ保存し、検証レポート{verified_count}件を更新しました。"


def verify_predictions(round_no=None):
    predictions = read_csv(PREDICTIONS_CSV, PREDICTION_COLUMNS)
    official = read_csv(OFFICIAL_RESULTS_CSV, OFFICIAL_RESULT_COLUMNS)
    if predictions.empty or official.empty:
        return 0
    predictions["開催回"] = predictions["開催回"].map(to_int)
    official["開催回"] = official["開催回"].map(to_int)
    if round_no is not None:
        predictions = predictions[predictions["開催回"] == int(round_no)]
        official = official[official["開催回"] == int(round_no)]

    rows = []
    contribution_rows = []
    for _, prediction in predictions.iterrows():
        matches = official[official["開催回"] == to_int(prediction["開催回"])]
        if not matches.empty:
            official_row = matches.tail(1).iloc[0]
            predicted = parse_number_text(prediction["予想番号"])
            actual = parse_number_text(official_row["本数字"])
            support_map = build_support_map_for_prediction(
                predicted,
                to_int(prediction["開催回"]),
                selected_model=prediction.get("使用モデル", ""),
            )
            rows.append(build_report_row(prediction, official_row, support_map))
            contribution_rows.extend(
                build_contribution_rows(
                    "ロト7研究所",
                    prediction,
                    predicted,
                    actual,
                    support_map,
                    saved_at=now_text(),
                )
            )

    reports = read_csv(VERIFICATION_REPORTS_CSV, VERIFICATION_COLUMNS)
    if not reports.empty and rows:
        ids = {str(row["予想ID"]) for row in rows}
        reports = reports[~reports["予想ID"].astype(str).isin(ids)]
    if rows:
        reports = pd.concat([reports, pd.DataFrame(rows)], ignore_index=True)
        reports = add_verification_metrics(reports, draw_size=7)
        reports["開催回"] = reports["開催回"].map(to_int)
        save_csv(reports.sort_values(["開催回", "予想ID"]), VERIFICATION_REPORTS_CSV, VERIFICATION_COLUMNS)
    if contribution_rows:
        contributions = read_csv(CONTRIBUTIONS_CSV, CONTRIBUTION_COLUMNS)
        contributions = merge_contribution_rows(contributions, contribution_rows)
        save_csv(contributions, CONTRIBUTIONS_CSV, CONTRIBUTION_COLUMNS)
    if rows:
        cycles = read_csv(RESEARCH_CYCLES_CSV, RESEARCH_CYCLE_COLUMNS)
        cycles = merge_research_cycle_rows(cycles, build_research_cycle_rows("ロト7研究所", rows, saved_at=now_text()))
        save_csv(cycles, RESEARCH_CYCLES_CSV, RESEARCH_CYCLE_COLUMNS)
    return len(rows)


def render_registration_form():
    st.sidebar.subheader("新しい当選結果を登録")
    st.sidebar.caption("実際の当選結果はここに入力します。数字欄へ本数字7個とボーナス2個を入れてください。")
    if st.sidebar.button("表示中の最新回を公式結果ログに保存", key="import_latest_loto7_result"):
        try:
            st.session_state["registration_message"] = import_latest_result_to_official_log()
            st.rerun()
        except Exception as exc:
            st.sidebar.error(str(exc))
    results = merge_official_results(read_csv(RESULTS_CSV))
    next_round = 1 if results.empty else int(results["開催回"].map(to_int).max()) + 1
    with st.sidebar.form("new_loto7_result_form"):
        round_no = st.number_input("開催回", min_value=1, value=next_round, step=1)
        draw_date = st.date_input("日付", value=date.today())
        cols = st.columns(2)
        main_numbers = [
            cols[index % 2].number_input(f"本数字{index + 1}", min_value=1, max_value=37, value=index + 1, step=1, key=f"loto7_main_{index}")
            for index in range(7)
        ]
        bonus_numbers = [
            cols[index % 2].number_input(f"ボーナス数字{index + 1}", min_value=1, max_value=37, value=30 + index, step=1, key=f"loto7_bonus_{index}")
            for index in range(2)
        ]
        submitted = st.form_submit_button("公式結果ログに保存")
    if not submitted:
        return
    main_numbers = [int(number) for number in main_numbers]
    bonus_numbers = [int(number) for number in bonus_numbers]
    if len(set(main_numbers)) != 7:
        st.sidebar.error("本数字7個は重複しない数字で入力してください。")
        return
    if len(set(bonus_numbers)) != 2 or set(main_numbers) & set(bonus_numbers):
        st.sidebar.error("ボーナス数字2個は本数字と重複しない数字で入力してください。")
        return
    upsert_official_result(int(round_no), draw_date, main_numbers, bonus_numbers, "manual")
    verify_predictions(int(round_no))
    st.session_state["registration_message"] = f"第{int(round_no)}回を公式結果ログに保存しました。"
    st.rerun()


def render_prediction_area(results):
    st.subheader("次回候補スコア 上位")
    if results.empty:
        st.info("loto7.csv がありません。")
        return
    results = results.copy()
    results["開催回"] = results["開催回"].map(to_int)
    st.markdown("**研究フロー: 当選番号追加 → 結果分析 → 反省履歴保存 → 履歴分析 → モデル評価 → 改善条件抽出 → 次回予想生成**")
    with st.spinner("予想生成前の履歴分析とモデル評価を実行しています..."):
        pre_summary, _, best_key = get_pre_prediction_research(results)
    model_options = [key for key in MODEL_LABELS if key != "random_baseline"]
    active_setting = read_active_model_setting()
    default_model = best_key if best_key in model_options else active_setting["model_key"] if active_setting and active_setting["model_key"] in model_options else "machine_learning"
    model_key = st.selectbox(
        "研究モデル",
        model_options,
        index=model_options.index(default_model),
        format_func=lambda key: MODEL_LABELS[key],
    )
    if best_key and not pre_summary.empty:
        best_row = pre_summary[pre_summary["モデル"] == MODEL_LABELS[best_key]].iloc[0]
        st.caption(
            f"履歴分析済み: {MODEL_LABELS[best_key]}を暫定採用。"
            f"3口内最高一致平均 {best_row['3口内最高一致平均']} / 3個以上一致率 {best_row['3個以上一致率']}%。"
        )
    elif active_setting:
        st.caption(f"反映中の研究モデル: {active_setting['model_name']}（{active_setting['reason']}）")
    scores = build_candidate_scores(results, model_key)
    st.dataframe(scores.head(20), width="stretch", hide_index=True)

    target_round = int(results["開催回"].max()) + 1
    picks = generate_prediction_picks(results, model_key=model_key)
    st.subheader("次回予想買い目")
    if not picks:
        st.info("予想生成に必要な履歴が不足しています。")
        return
    saved_count = save_prediction_picks(picks, target_round, model_key, MODEL_LABELS[model_key])
    st.caption(f"予測研究所ログ: 第{target_round}回の予想を loto7_predictions.csv に保存しました。新規保存 {saved_count}件。")
    for index, pick in enumerate(picks, start=1):
        st.markdown(f"**第{index}候補：{numbers_to_text(pick['numbers'])}**")
        st.write(f"理由：{pick['reason']}")


def render_ai_improvement_report():
    st.subheader("AI改善レポート")
    st.caption("予想、結果、検証、改善をセットで保存する研究開発メモです。当選や利益を保証するものではありません。")
    reports = read_csv(VERIFICATION_REPORTS_CSV, VERIFICATION_COLUMNS)
    reports = add_verification_metrics(reports, draw_size=7)
    summary = build_ai_improvement_summary(reports)
    cols = st.columns(3)
    cols[0].metric("一致数", summary["一致数"])
    cols[1].metric("予想", summary["予想"])
    cols[2].metric("結果", summary["結果"])
    st.markdown("**的中要因**")
    st.write(summary["的中要因"])
    st.markdown("**外れ方の解釈**")
    st.write(summary["外れ要因"])
    st.markdown("**改善案**")
    st.write(summary["改善案"])
    st.markdown("**次回仮説**")
    st.write(summary["次回仮説"])


def render_lab():
    st.subheader("予測研究所システム")
    st.caption("当選や利益を保証するものではなく、予測手法を検証するための研究開発ログです。")
    st.info(
        "このバックテストは、過去データに対して各予想モデルの傾向を比較する研究目的の検証です。\n\n"
        "表示される一致数や改善指標は、将来の当選や利益を保証するものではありません。\n\n"
        "本システムの目的は、予想、結果、検証、改善のサイクルを通じて、予測手法の有効性を継続的に研究することです。"
    )
    predictions = read_csv(PREDICTIONS_CSV, PREDICTION_COLUMNS)
    official = read_csv(OFFICIAL_RESULTS_CSV, OFFICIAL_RESULT_COLUMNS)
    reports = read_csv(VERIFICATION_REPORTS_CSV, VERIFICATION_COLUMNS)
    reports = add_verification_metrics(reports, draw_size=7)
    contributions = read_csv(CONTRIBUTIONS_CSV, CONTRIBUTION_COLUMNS)
    cols = st.columns(4)
    cols[0].metric("予想履歴", len(predictions))
    cols[1].metric("公式結果ログ", len(official))
    cols[2].metric("検証レポート", len(reports))
    cols[3].metric("貢献度ログ", len(contributions))
    if st.button("予想履歴と公式結果を照合して検証レポート更新"):
        st.success(f"検証レポートを{verify_predictions()}件更新しました。")
        reports = read_csv(VERIFICATION_REPORTS_CSV, VERIFICATION_COLUMNS)
        reports = add_verification_metrics(reports, draw_size=7)
        contributions = read_csv(CONTRIBUTIONS_CSV, CONTRIBUTION_COLUMNS)
    tabs = st.tabs(["バックテスト", "検証レポート", "モデル貢献度", "条件別成功率", "AI改善レポート", "動画仮説研究", "予想履歴", "公式結果"])
    with tabs[0]:
        st.markdown("**新フロー**")
        st.dataframe(build_research_flow_table(), width="stretch", hide_index=True)
        results = merge_official_results(read_csv(RESULTS_CSV))
        if len(results) <= 40:
            st.info("バックテストには最低41回分以上の抽せん結果が必要です。")
        else:
            max_rounds = max(10, min(30, len(results) - 40))
            default_rounds = min(10, max_rounds)
            lookback_rounds = st.slider("バックテスト対象回数", 10, max_rounds, default_rounds, key="loto7_backtest_rounds")
            if st.button("ロト7バックテストを実行"):
                with st.spinner("機械学習モデルを含めてロト7モデルを検証しています..."):
                    summary_df, detail_df = run_backtest(results, lookback_rounds=lookback_rounds)
                st.session_state["loto7_backtest_summary"] = summary_df
                st.session_state["loto7_backtest_detail"] = detail_df
                best_key = best_actionable_model_key(summary_df)
                if best_key:
                    save_active_model_setting(best_key, f"直近{lookback_rounds}回バックテストで最上位")
                    st.session_state["loto7_model_notice"] = f"{MODEL_LABELS[best_key]}を次回予想モデルとして保存しました。"
                    st.rerun()

            summary_df = st.session_state.get("loto7_backtest_summary", pd.DataFrame())
            detail_df = st.session_state.get("loto7_backtest_detail", pd.DataFrame())
            if "loto7_model_notice" in st.session_state:
                st.success(st.session_state.pop("loto7_model_notice"))
            if summary_df.empty:
                st.info("対象回数を選んでバックテストを実行してください。")
            else:
                st.markdown("**モデル別成績表**")
                st.dataframe(summary_df, width="stretch", hide_index=True)
                best_key = best_actionable_model_key(summary_df)
                ml_row = summary_df[summary_df["モデル"] == MODEL_LABELS["machine_learning"]]
                if best_key:
                    st.write(f"現在の比較では「{MODEL_LABELS[best_key]}」が実用モデル内で最上位です。")
                if not ml_row.empty:
                    ml_rank = summary_df["モデル"].tolist().index(MODEL_LABELS["machine_learning"]) + 1
                    st.write(
                        f"機械学習モデルは全{len(summary_df)}モデル中 {ml_rank}位です。"
                        f" 3口内最高一致平均 {ml_row.iloc[0]['3口内最高一致平均']}、"
                        f"平均近接一致数 {ml_row.iloc[0]['平均近接一致数']}。"
                    )
                    if best_key != "machine_learning":
                        st.warning("機械学習モデルは今回のバックテストでは最上位ではありません。AI改善部門の改善対象として扱います。")
                st.caption("この結果は過去データ上の検証であり、将来の当選や利益を保証するものではありません。")
                with st.expander("回別バックテスト詳細"):
                    st.dataframe(detail_df, width="stretch", hide_index=True)
    with tabs[1]:
        if reports.empty:
            st.info("検証レポートはまだありません。")
        else:
            sorted_reports = reports.sort_values("開催回", ascending=False)
            st.dataframe(sorted_reports, width="stretch", hide_index=True)
            latest_report = sorted_reports.iloc[0]
            st.markdown("**直近の検証サマリー**")
            st.write(f"予想番号: {latest_report['予想番号']} / 実際: {latest_report['実際の当選番号']}")
            st.write(f"一致数: {latest_report['本数字一致数']} / ボーナス一致数: {latest_report['ボーナス一致数']} / 判定: {latest_report['等級判定']}")
            st.write(f"的中要因: {latest_report.get('的中要因', '-')}")
            st.write(f"外れ要因: {latest_report['失敗要因']}")
            st.write(f"足りなかった条件: {latest_report.get('足りなかった条件', '-')}")
            st.write(f"過剰だった条件: {latest_report.get('過剰だった条件', '-')}")
            st.write(f"改善案: {latest_report['改善案']}")
            st.write(f"次回の仮説: {latest_report['次回の仮説']}")
    with tabs[2]:
        ranking = build_contribution_ranking(contributions)
        dashboard = build_model_dashboard(reports)
        if ranking.empty and dashboard.empty:
            st.info("モデル貢献度は、予想と結果を照合すると作成されます。")
        else:
            st.markdown("**モデル貢献度ランキング**")
            st.dataframe(ranking, width="stretch", hide_index=True)
            st.markdown("**モデル別成績**")
            st.dataframe(dashboard, width="stretch", hide_index=True)
            with st.expander("数字別モデル貢献度ログ"):
                st.dataframe(contributions.sort_values(["開催回", "予想ID"], ascending=[False, True]), width="stretch", hide_index=True)
    with tabs[3]:
        condition_df = build_condition_success_table(reports, number_max=37)
        if condition_df.empty:
            st.info("条件別成功率は、検証レポート作成後に表示されます。")
        else:
            st.dataframe(condition_df, width="stretch", hide_index=True)
    with tabs[4]:
        summary = build_ai_improvement_summary(reports)
        summary_cols = st.columns(3)
        summary_cols[0].metric("一致数", summary["一致数"])
        summary_cols[1].metric("予想", summary["予想"])
        summary_cols[2].metric("結果", summary["結果"])
        st.write(f"的中要因: {summary['的中要因']}")
        st.write(f"外れ要因: {summary['外れ要因']}")
        st.write(f"改善案: {summary['改善案']}")
        st.write(f"次回仮説: {summary['次回仮説']}")
    with tabs[5]:
        video_logs = read_csv(VIDEO_HYPOTHESES_CSV, VIDEO_HYPOTHESIS_COLUMNS)
        with st.form("loto7_video_hypothesis_form"):
            video_name = st.text_input("動画名")
            transcript = st.text_area("YouTube文字起こし", height=140)
            result_text = st.text_input("バックテスト成績", value="未バックテスト")
            submitted = st.form_submit_button("仮説として保存")
        if submitted:
            row = extract_video_hypothesis(video_name, transcript, result_text)
            video_logs = pd.concat([video_logs, pd.DataFrame([row])], ignore_index=True)
            save_csv(video_logs, VIDEO_HYPOTHESES_CSV, VIDEO_HYPOTHESIS_COLUMNS)
            st.success("動画仮説を保存しました。")
        if video_logs.empty:
            st.info("動画仮説ログはまだありません。")
        else:
            st.dataframe(video_logs.sort_values("保存日時", ascending=False), width="stretch", hide_index=True)
    with tabs[6]:
        if predictions.empty:
            st.info("予想履歴はまだありません。")
        else:
            st.dataframe(predictions.sort_values(["開催回", "候補番号"], ascending=[False, True]), width="stretch", hide_index=True)
    with tabs[7]:
        if official.empty:
            st.info("公式結果ログはまだありません。")
        else:
            st.dataframe(official.sort_values("開催回", ascending=False), width="stretch", hide_index=True)


render_registration_form()
if "registration_message" in st.session_state:
    st.success(st.session_state.pop("registration_message"))

st.info("当選結果を入力する場合は、画面左側のサイドバーにある「新しい当選結果を登録」を使ってください。サイドバーが見えない場合は、左上の矢印で開けます。")

results = merge_official_results(read_csv(RESULTS_CSV))
if results.empty:
    st.info("loto7.csv が見つからないか、データが空です。")
    st.stop()

latest = results.sort_values("開催回").tail(1).iloc[0]
st.subheader("最新回")
cols = st.columns(3)
cols[0].metric("開催回", int(latest["開催回"]))
cols[1].metric("日付", latest["日付"])
cols[2].metric("キャリーオーバー", latest.get("キャリーオーバー", "-"))
st.info(
    f"本数字: {' - '.join(f'{number:02d}' for number in row_numbers(latest))} / "
    f"BONUS: {' - '.join(f'{number:02d}' for number in row_bonus_numbers(latest))}"
)

render_prediction_area(results)
render_ai_improvement_report()
render_lab()

st.subheader("最近の抽せん結果")
st.dataframe(results.sort_values("開催回", ascending=False).head(30), width="stretch", hide_index=True)
