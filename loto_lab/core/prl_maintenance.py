from datetime import datetime
from itertools import combinations
from pathlib import Path
import re
import shutil

import pandas as pd

from arl_research_engine import (
    ARL_MODEL_LABELS,
    CONTRIBUTION_COLUMNS,
    RESEARCH_CYCLE_COLUMNS,
    add_verification_metrics,
    append_winning_condition_history,
    build_contribution_rows,
    build_effective_conditions,
    build_hit_factor_summary,
    build_missing_excess_conditions,
    build_model_support_map,
    build_winning_condition_report,
    build_research_cycle_rows,
    format_contribution_detail,
    merge_contribution_rows,
    merge_research_cycle_rows,
    numbers_to_text,
    parse_numbers,
    safe_int,
)


CORE_DIR = Path(__file__).resolve().parent
LOTO_LAB_DIR = CORE_DIR.parent
PROJECT_ROOT = LOTO_LAB_DIR.parent
DATA_DIR = LOTO_LAB_DIR / "data"
VERIFICATION_DIR = DATA_DIR / "verification"
BACKUP_DIR = DATA_DIR / "backups"
AI_IMPROVEMENT_DIR = DATA_DIR / "ai_improvement"
BASE_DIR = LOTO_LAB_DIR

PREDICTION_COLUMNS = ["予想ID", "開催回", "予想日", "候補番号", "予想番号", "使用モデル", "予想理由", "保存日時"]
LOTO6_OFFICIAL_RESULT_COLUMNS = ["開催回", "抽せん日", "本数字", "ボーナス数字", "球セット", "登録元", "保存日時"]
LOTO7_OFFICIAL_RESULT_COLUMNS = ["開催回", "抽せん日", "本数字", "ボーナス数字", "登録元", "保存日時"]
LOTO6_NUMBER_COLUMNS = ["第1数字", "第2数字", "第3数字", "第4数字", "第5数字", "第6数字"]
LOTO7_NUMBER_COLUMNS = ["第1数字", "第2数字", "第3数字", "第4数字", "第5数字", "第6数字", "第7数字"]
LOTO7_BONUS_COLUMNS = ["BONUS数字1", "BONUS数字2"]

LOTO7_VERIFICATION_COLUMNS = [
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

MODEL_NAME_REPLACEMENTS = {
    "頻出数字バランス": "出現頻度分析",
    "直近トレンド": "ホットナンバー分析",
    "出現間隔": "コールドナンバー分析",
    "ホット分析": "ホットナンバー分析",
    "コールド分析": "コールドナンバー分析",
    "ボーナス分析": "ボーナス数字分析",
    "モンテカルロ": "モンテカルロシミュレーション",
    "機械学習": "機械学習モデル",
    "ランダムモデル": "ランダム予測モデル",
    "ランダム比較": "ランダム予測モデル",
    "AI改善モデル": "機械学習モデル",
    "非重複モデル": "人と被りにくい高額当選狙いモデル",
    "固定数字モデル": "出現頻度分析",
    "変動数字モデル": "コールドナンバー分析",
    "ハイブリッドモデル": "機械学習モデル",
    "ホット10モデル": "ホットナンバー分析",
    "ホット20モデル": "ホットナンバー分析",
    "ホット50モデル": "ホットナンバー分析",
    "コールド復活モデル": "コールドナンバー分析",
    "ボーナス昇格モデル": "ボーナス数字分析",
    "score_balance_v1": "機械学習モデル",
}

MODEL_NAME_TO_KEY = {label: key for key, label in ARL_MODEL_LABELS.items()}
MODEL_NAME_TO_KEY.update(
    {
        "頻出数字バランス": "frequency_analysis",
        "直近トレンド": "hot_analysis",
        "出現間隔": "cold_analysis",
        "ホット分析": "hot_analysis",
        "コールド分析": "cold_analysis",
        "ボーナス分析": "bonus_analysis",
        "モンテカルロ": "monte_carlo",
        "機械学習": "machine_learning",
        "AI改善モデル": "machine_learning",
        "ランダムモデル": "random_baseline",
        "ランダム比較": "random_baseline",
        "score_balance_v1": "machine_learning",
    }
)

TARGETS = [
    {
        "name": "ロト6",
        "draw_size": 6,
        "report": "verification_reports.csv",
        "contribution": "loto6_model_contributions.csv",
        "cycle": "loto6_research_cycles.csv",
        "prediction": "predictions.csv",
        "prediction_prefix": "L6",
        "rewrite_duplicate_prediction_ids": False,
    },
    {
        "name": "ロト7",
        "draw_size": 7,
        "report": "loto7_verification_reports.csv",
        "contribution": "loto7_model_contributions.csv",
        "cycle": "loto7_research_cycles.csv",
        "prediction": "loto7_predictions.csv",
        "prediction_prefix": "L7",
        "rewrite_duplicate_prediction_ids": True,
    },
]

SYNC_TARGETS = [
    ("loto6.csv", "分析研究所/ロト6/loto6.csv"),
    ("results.csv", "分析研究所/ロト6/results.csv"),
    ("predictions.csv", "分析研究所/ロト6/predictions.csv"),
    ("verification_reports.csv", "分析研究所/検証/verification_reports_loto6.csv"),
    ("loto6_model_contributions.csv", "分析研究所/AI改善/loto6_model_contributions.csv"),
    ("loto6_research_cycles.csv", "分析研究所/reports/loto6_research_cycles.csv"),
    ("loto6_next_number_scores.csv", "分析研究所/data/loto6_next_number_scores.csv"),
    ("loto6_ball_sets.csv", "分析研究所/data/loto6_ball_sets.csv"),
    ("purchases.csv", "分析研究所/data/purchases.csv"),
    ("loto7.csv", "分析研究所/ロト7/loto7.csv"),
    ("loto7_next_number_scores.csv", "分析研究所/data/loto7_next_number_scores.csv"),
    ("loto7_results.csv", "分析研究所/ロト7/loto7_results.csv"),
    ("loto7_predictions.csv", "分析研究所/ロト7/loto7_predictions.csv"),
    ("loto7_verification_reports.csv", "分析研究所/検証/verification_reports_loto7.csv"),
    ("loto7_model_contributions.csv", "分析研究所/AI改善/loto7_model_contributions.csv"),
    ("loto7_research_cycles.csv", "分析研究所/reports/loto7_research_cycles.csv"),
    ("video_hypotheses.csv", "分析研究所/reports/video_hypotheses.csv"),
]


def read_csv(path, columns=None):
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def ensure_columns(df, columns):
    df = df.copy()
    for column in columns:
        if column not in df:
            df[column] = ""
    return df.reindex(columns=columns)


def backup_file(path):
    if not path.exists():
        return ""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{path.stem}_{timestamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    return str(backup_path)


def normalize_text(value):
    text = str(value)
    for old, new in MODEL_NAME_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def model_key_from_name(model_name):
    normalized = normalize_text(model_name)
    return MODEL_NAME_TO_KEY.get(normalized, str(model_name))


def prediction_id_date(value):
    digits = re.sub(r"\D", "", str(value or ""))
    return digits or datetime.now().strftime("%Y%m%d")


def unique_prediction_id(row, prefix, used_ids, fallback_index):
    round_no = safe_int(row.get("開催回"))
    candidate_no = safe_int(row.get("候補番号"), fallback_index)
    model_key = model_key_from_name(row.get("使用モデル", ""))
    date_digits = prediction_id_date(row.get("予想日"))
    base_id = f"{prefix}-{round_no}-{date_digits}-{model_key}-{candidate_no}"
    prediction_id = base_id
    suffix = 2
    while prediction_id in used_ids:
        prediction_id = f"{base_id}-{suffix}"
        suffix += 1
    used_ids.add(prediction_id)
    return prediction_id


def normalize_text_columns(df, columns):
    if df.empty:
        return df
    df = df.copy()
    for column in columns:
        if column in df:
            df[column] = df[column].map(normalize_text)
    return df


def migrate_predictions(path, prefix, rewrite_duplicate_ids, apply_changes):
    predictions = read_csv(path, PREDICTION_COLUMNS)
    if predictions.empty:
        return {"file": path.name, "status": "empty", "rows": 0, "backup": ""}
    predictions = ensure_columns(predictions, PREDICTION_COLUMNS)
    predictions = normalize_text_columns(predictions, ["使用モデル", "予想理由"])
    predictions["開催回"] = predictions["開催回"].map(safe_int)
    predictions["候補番号"] = predictions["候補番号"].map(safe_int)

    duplicate_ids = set()
    if "予想ID" in predictions:
        duplicate_ids = {
            str(name)
            for name, count in predictions["予想ID"].astype(str).value_counts().items()
            if count > 1 or name in ("", "nan")
        }
    if rewrite_duplicate_ids and duplicate_ids:
        used_ids = set()
        new_ids = []
        for index, row in predictions.iterrows():
            original_id = str(row.get("予想ID", ""))
            if original_id in duplicate_ids:
                new_ids.append(unique_prediction_id(row, prefix, used_ids, index + 1))
            else:
                candidate_id = original_id
                suffix = 2
                while candidate_id in used_ids:
                    candidate_id = f"{original_id}-{suffix}"
                    suffix += 1
                used_ids.add(candidate_id)
                new_ids.append(candidate_id)
        predictions["予想ID"] = new_ids

    predictions = predictions.sort_values(["開催回", "候補番号", "使用モデル", "保存日時"])
    backup = ""
    if apply_changes:
        backup = backup_file(path)
        save_csv(predictions, path)
    status = "updated" if apply_changes else "dry-run"
    if rewrite_duplicate_ids and duplicate_ids:
        status += f" duplicate ids fixed={len(duplicate_ids)}"
    return {"file": path.name, "status": status, "rows": len(predictions), "backup": backup}


def migrate_report(path, draw_size, apply_changes, columns=None):
    report = read_csv(path)
    if report.empty:
        return {"file": path.name, "status": "empty", "rows": 0, "backup": ""}
    migrated = add_verification_metrics(report, draw_size=draw_size)
    migrated = normalize_text_columns(
        migrated,
        ["使用モデル", "的中要因", "有効条件", "モデル貢献度詳細", "失敗要因", "改善案", "次回の仮説"],
    )
    if columns:
        migrated = ensure_columns(migrated, columns)
    backup = ""
    if apply_changes:
        backup = backup_file(path)
        save_csv(migrated, path)
    return {"file": path.name, "status": "updated" if apply_changes else "dry-run", "rows": len(migrated), "backup": backup}


def migrate_contributions(path, apply_changes):
    contributions = read_csv(path, CONTRIBUTION_COLUMNS)
    if contributions.empty:
        return {"file": path.name, "status": "empty", "rows": 0, "backup": ""}
    migrated = normalize_text_columns(contributions, ["選出モデル"])
    backup = ""
    if apply_changes:
        backup = backup_file(path)
        save_csv(migrated.reindex(columns=CONTRIBUTION_COLUMNS), path)
    return {"file": path.name, "status": "updated" if apply_changes else "dry-run", "rows": len(migrated), "backup": backup}


def count_consecutive_pairs(numbers):
    ordered = sorted(numbers)
    return sum(1 for left, right in zip(ordered, ordered[1:]) if right - left == 1)


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
    }


def describe_last_digit_bias(numbers):
    counts = pd.Series([number % 10 for number in numbers]).value_counts()
    biased_digits = [f"{int(digit)}が{int(count)}個" for digit, count in counts.items() if count >= 2]
    return " / ".join(biased_digits) if biased_digits else "大きな偏りなし"


def determine_loto7_grade(match_count, bonus_count):
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


def build_loto7_report_row(prediction_row, official_row, support_map=None):
    predicted = parse_numbers(prediction_row.get("予想番号", ""), 37)
    actual = parse_numbers(official_row.get("本数字", ""), 37)
    bonus = parse_numbers(official_row.get("ボーナス数字", ""), 37)
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
    grade = determine_loto7_grade(match_count, bonus_count)
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
        "予想ID": prediction_row.get("予想ID", ""),
        "開催回": safe_int(prediction_row.get("開催回")),
        "検証日": datetime.now().strftime("%Y/%m/%d"),
        "使用モデル": normalize_text(prediction_row.get("使用モデル", "")),
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


def historical_loto7_context_before_round(round_no):
    history = read_csv(DATA_DIR / "loto7.csv")
    if history.empty:
        return [], []
    history["開催回"] = history["開催回"].map(safe_int)
    history = history[history["開催回"] < safe_int(round_no)].sort_values("開催回")
    number_rows = []
    bonus_rows = []
    for _, row in history.iterrows():
        number_rows.append([safe_int(row.get(column)) for column in LOTO7_NUMBER_COLUMNS])
        bonus_rows.append([safe_int(row.get(column)) for column in LOTO7_BONUS_COLUMNS])
    return number_rows, bonus_rows


def rebuild_loto7_artifacts(apply_changes):
    prediction_path = DATA_DIR / "loto7_predictions.csv"
    official_path = DATA_DIR / "loto7_results.csv"
    report_path = VERIFICATION_DIR / "loto7_verification_reports.csv"
    contribution_path = VERIFICATION_DIR / "loto7_model_contributions.csv"
    cycle_path = VERIFICATION_DIR / "loto7_research_cycles.csv"

    predictions = read_csv(prediction_path, PREDICTION_COLUMNS)
    official = read_csv(official_path, LOTO7_OFFICIAL_RESULT_COLUMNS)
    if predictions.empty or official.empty:
        return [
            {"file": report_path.name, "status": "missing predictions or official results", "rows": 0, "backup": ""},
            {"file": contribution_path.name, "status": "missing predictions or official results", "rows": 0, "backup": ""},
            {"file": cycle_path.name, "status": "missing predictions or official results", "rows": 0, "backup": ""},
        ]

    predictions = ensure_columns(predictions, PREDICTION_COLUMNS)
    official = ensure_columns(official, LOTO7_OFFICIAL_RESULT_COLUMNS)
    predictions["開催回"] = predictions["開催回"].map(safe_int)
    official["開催回"] = official["開催回"].map(safe_int)
    predictions = normalize_text_columns(predictions, ["使用モデル", "予想理由"])

    report_rows = []
    contribution_rows = []
    winning_condition_rows = []
    model_improvement_rows = []
    saved_at = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    for _, prediction in predictions.iterrows():
        matches = official[official["開催回"] == safe_int(prediction.get("開催回"))]
        if matches.empty:
            continue
        official_row = matches.tail(1).iloc[0]
        draw_no = safe_int(prediction.get("開催回"))
        predicted = parse_numbers(prediction.get("予想番号", ""), 37)
        actual = parse_numbers(official_row.get("本数字", ""), 37)
        bonus = parse_numbers(official_row.get("ボーナス数字", ""), 37)
        number_rows, bonus_rows = historical_loto7_context_before_round(draw_no)
        support_map = build_model_support_map(
            predicted,
            number_rows,
            37,
            7,
            draw_no,
            bonus_rows,
            selected_model=prediction.get("使用モデル", ""),
        )
        report_row = build_loto7_report_row(prediction, official_row, support_map)
        report_rows.append(report_row)
        winning_row, model_rows = build_winning_condition_report(
            lottery_type="loto7",
            draw_no=draw_no,
            prediction_id=prediction.get("予想ID", ""),
            prediction_date=prediction.get("予想日", ""),
            predicted=predicted,
            actual=actual,
            bonus_numbers=bonus,
            number_rows=number_rows,
            bonus_rows=bonus_rows,
            number_max=37,
            draw_size=7,
            selected_model=prediction.get("使用モデル", ""),
            failure_reason=report_row["失敗要因"],
            created_at=saved_at,
        )
        winning_condition_rows.append(winning_row)
        model_improvement_rows.extend(model_rows)
        contribution_rows.extend(
            build_contribution_rows(
                "ロト7研究所",
                prediction,
                predicted,
                actual,
                support_map,
                saved_at=saved_at,
            )
        )

    reports = ensure_columns(pd.DataFrame(report_rows), LOTO7_VERIFICATION_COLUMNS)
    reports = add_verification_metrics(reports, draw_size=7)
    if not reports.empty:
        reports["開催回"] = reports["開催回"].map(safe_int)
        reports = reports.sort_values(["開催回", "予想ID"])

    contributions = pd.DataFrame(contribution_rows, columns=CONTRIBUTION_COLUMNS)
    if not contributions.empty:
        contributions["開催回"] = contributions["開催回"].map(safe_int)
        contributions = contributions.sort_values(["開催回", "予想ID", "予想数字", "選出モデル"])

    cycles = pd.DataFrame(build_research_cycle_rows("ロト7研究所", reports.to_dict("records"), saved_at=saved_at), columns=RESEARCH_CYCLE_COLUMNS)
    if not cycles.empty:
        cycles["開催回"] = cycles["開催回"].map(safe_int)
        cycles = cycles.sort_values(["開催回", "予想ID"])

    results = []
    for path, df, columns in [
        (report_path, reports, LOTO7_VERIFICATION_COLUMNS),
        (contribution_path, contributions, CONTRIBUTION_COLUMNS),
        (cycle_path, cycles, RESEARCH_CYCLE_COLUMNS),
    ]:
        backup = ""
        if apply_changes:
            backup = backup_file(path)
            save_csv(ensure_columns(df, columns), path)
        results.append({"file": path.name, "status": "rebuilt" if apply_changes else "dry-run rebuild", "rows": len(df), "backup": backup})
    if apply_changes and winning_condition_rows:
        append_winning_condition_history(AI_IMPROVEMENT_DIR, winning_condition_rows, model_improvement_rows)
        results.append(
            {
                "file": "data/ai_improvement/winning_condition_history.csv",
                "status": "rebuilt",
                "rows": len(winning_condition_rows),
                "backup": "",
            }
        )
    return results


def rebuild_research_cycles(name, report_path, cycle_path, apply_changes):
    reports = read_csv(report_path)
    if reports.empty:
        return {"file": cycle_path.name, "status": "no reports", "rows": 0, "backup": ""}
    rows = build_research_cycle_rows(f"{name}研究所", reports.to_dict("records"))
    existing = read_csv(cycle_path, RESEARCH_CYCLE_COLUMNS)
    cycles = merge_research_cycle_rows(existing, rows)
    backup = ""
    if apply_changes:
        backup = backup_file(cycle_path)
        save_csv(cycles.reindex(columns=RESEARCH_CYCLE_COLUMNS), cycle_path)
    return {"file": cycle_path.name, "status": "updated" if apply_changes else "dry-run", "rows": len(cycles), "backup": backup}


def resolve_loto_data_path(filename):
    verification_files = {
        "verification_reports.csv",
        "loto6_model_contributions.csv",
        "loto6_research_cycles.csv",
        "loto7_verification_reports.csv",
        "loto7_model_contributions.csv",
        "loto7_research_cycles.csv",
        "video_hypotheses.csv",
    }
    if filename in verification_files:
        return VERIFICATION_DIR / filename
    return DATA_DIR / filename


def sync_project_files(apply_changes):
    results = []
    for source_name, destination_name in SYNC_TARGETS:
        source = resolve_loto_data_path(source_name)
        destination = BASE_DIR / destination_name
        if not source.exists():
            results.append({"file": destination_name, "status": "missing source", "rows": 0, "backup": ""})
            continue
        if apply_changes:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        results.append({"file": destination_name, "status": "synced" if apply_changes else "dry-run", "rows": 1, "backup": ""})
    return results


def run_maintenance(apply_changes=True):
    results = []
    for target in TARGETS:
        prediction_path = DATA_DIR / target["prediction"]
        results.append(
            migrate_predictions(
                prediction_path,
                target["prediction_prefix"],
                target["rewrite_duplicate_prediction_ids"],
                apply_changes,
            )
        )

        if target["name"] == "ロト7":
            results.extend(rebuild_loto7_artifacts(apply_changes))
            continue

        report_path = VERIFICATION_DIR / target["report"]
        contribution_path = VERIFICATION_DIR / target["contribution"]
        cycle_path = VERIFICATION_DIR / target["cycle"]
        results.append(migrate_report(report_path, target["draw_size"], apply_changes))
        results.append(migrate_contributions(contribution_path, apply_changes))
        results.append(rebuild_research_cycles(target["name"], report_path, cycle_path, apply_changes))
    results.extend(sync_project_files(apply_changes))
    return pd.DataFrame(results)


if __name__ == "__main__":
    summary = run_maintenance(apply_changes=True)
    print(summary.to_string(index=False))
