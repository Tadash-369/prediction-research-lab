from datetime import date, datetime
from itertools import combinations
from io import StringIO
from pathlib import Path
import re
import ssl
import subprocess
import sys
import urllib.request

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
RESULTS_CSV = BASE_DIR / "loto6.csv"
SETS_CSV = BASE_DIR / "loto6_ball_sets.csv"
SCORES_CSV = BASE_DIR / "loto6_next_number_scores.csv"
PREDICTIONS_CSV = BASE_DIR / "predictions.csv"
OFFICIAL_RESULTS_CSV = BASE_DIR / "results.csv"
VERIFICATION_REPORTS_CSV = BASE_DIR / "verification_reports.csv"
MODEL_SETTINGS_CSV = BASE_DIR / "model_settings.csv"
CONTRIBUTIONS_CSV = BASE_DIR / "loto6_model_contributions.csv"
RESEARCH_CYCLES_CSV = BASE_DIR / "loto6_research_cycles.csv"
VIDEO_HYPOTHESES_CSV = BASE_DIR / "video_hypotheses.csv"
MIZUHO_LOTO6_URL = "https://www.mizuhobank.co.jp/retail/takarakuji/check/loto/loto6/index.html"

RESULT_COLUMNS = [
    "開催回",
    "日付",
    "第1数字",
    "第2数字",
    "第3数字",
    "第4数字",
    "第5数字",
    "第6数字",
    "BONUS数字",
    "1等口数",
    "2等口数",
    "3等口数",
    "4等口数",
    "5等口数",
    "1等賞金",
    "2等賞金",
    "3等賞金",
    "4等賞金",
    "5等賞金",
    "キャリーオーバー",
]
SET_COLUMNS = ["開催回", "日付", "球セット", "信頼度", "根拠", "動画メタID", "メモ"]
NUMBER_COLUMNS = ["第1数字", "第2数字", "第3数字", "第4数字", "第5数字", "第6数字"]
PREDICTION_COLUMNS = ["予想ID", "開催回", "予想日", "候補番号", "予想番号", "使用モデル", "予想理由", "保存日時"]
OFFICIAL_RESULT_COLUMNS = ["開催回", "抽せん日", "本数字", "ボーナス数字", "球セット", "登録元", "保存日時"]
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
    "ボーナス一致",
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
BACKTEST_MODELS = dict(ARL_MODEL_LABELS)
MODEL_SETTING_COLUMNS = ["設定名", "モデルキー", "モデル名", "根拠", "更新日時"]


st.set_page_config(page_title="ロト6分析", layout="wide")
st.title("ロト6 分析ダッシュボード")


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
    df = df.reindex(columns=columns)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def normalize_score_map(score_map):
    values = list(score_map.values())
    if not values:
        return {number: 0.0 for number in range(1, 44)}
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return {number: 0.0 for number in score_map}
    return {number: (value - min_value) / (max_value - min_value) for number, value in score_map.items()}


def read_active_model_setting():
    settings = read_csv(MODEL_SETTINGS_CSV, MODEL_SETTING_COLUMNS)
    if settings.empty:
        return None
    active = settings[settings["設定名"] == "active_next_prediction"]
    if active.empty:
        return None
    row = active.tail(1).iloc[0]
    return {
        "model_key": str(row["モデルキー"]),
        "model_name": str(row["モデル名"]),
        "reason": str(row["根拠"]),
    }


def save_active_model_setting(model_key, reason):
    model_name = BACKTEST_MODELS.get(model_key, model_key)
    settings = read_csv(MODEL_SETTINGS_CSV, MODEL_SETTING_COLUMNS)
    settings = settings[settings["設定名"] != "active_next_prediction"] if not settings.empty else settings
    row = {
        "設定名": "active_next_prediction",
        "モデルキー": model_key,
        "モデル名": model_name,
        "根拠": reason,
        "更新日時": now_text(),
    }
    settings = pd.concat([settings, pd.DataFrame([row])], ignore_index=True)
    save_csv(settings, MODEL_SETTINGS_CSV, MODEL_SETTING_COLUMNS)


def model_key_from_name(model_name):
    for key, name in BACKTEST_MODELS.items():
        if name == model_name:
            return key
    return None


def best_actionable_model_key(summary_df):
    if summary_df.empty:
        return None
    usable_rows = summary_df[summary_df["モデル"] != BACKTEST_MODELS["random_baseline"]]
    if usable_rows.empty:
        return None
    return model_key_from_name(str(usable_rows.iloc[0]["モデル"]))


def run_analysis():
    scripts = ["analyze_loto6_ball_sets.py", "analyze_loto6_comprehensive.py"]
    logs = []
    for script in scripts:
        completed = subprocess.run(
            [sys.executable, str(BASE_DIR / script)],
            cwd=BASE_DIR,
            text=True,
            capture_output=True,
        )
        logs.append(completed.stdout or completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"{script} の実行に失敗しました。\n{completed.stderr}")
    return "\n".join(logs)


def run_helper_script(script):
    completed = subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{script} の実行に失敗しました。\n{completed.stderr}")
    return completed.stdout or completed.stderr


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def to_int(value, default=0):
    try:
        return int(str(value).replace(",", "").strip())
    except Exception:
        return default


def today_text():
    return date.today().strftime("%Y/%m/%d")


def now_text():
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def numbers_to_text(numbers):
    return "-".join(f"{int(number):02d}" for number in sorted(numbers))


def parse_number_text(value):
    numbers = [to_int(number) for number in re.findall(r"\d+", str(value))]
    return sorted(number for number in numbers if 1 <= number <= 43)


def merge_official_results(results):
    official = read_csv(OFFICIAL_RESULTS_CSV, OFFICIAL_RESULT_COLUMNS)
    if official.empty:
        return results

    merged = results.copy()
    if merged.empty:
        merged = pd.DataFrame(columns=RESULT_COLUMNS)

    merged["開催回"] = merged["開催回"].map(to_int)
    existing_rounds = set(merged["開催回"].tolist())
    rows = []
    for _, row in official.iterrows():
        round_no = to_int(row["開催回"])
        main_numbers = parse_number_text(row["本数字"])
        bonus_numbers = parse_number_text(row["ボーナス数字"])
        if round_no in existing_rounds or len(main_numbers) != 6 or not bonus_numbers:
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
            "BONUS数字": bonus_numbers[0],
            "1等口数": "",
            "2等口数": "",
            "3等口数": "",
            "4等口数": "",
            "5等口数": "",
            "1等賞金": "",
            "2等賞金": "",
            "3等賞金": "",
            "4等賞金": "",
            "5等賞金": "",
            "キャリーオーバー": "",
        }
        rows.append(result_row)

    if rows:
        merged = pd.concat([merged, pd.DataFrame(rows)], ignore_index=True)
    merged["開催回"] = merged["開催回"].map(to_int)
    return merged.sort_values("開催回")


def normalize_digits(text):
    table = str.maketrans("０１２３４５６７８９", "0123456789")
    return str(text).translate(table)


def parse_japanese_date(text):
    normalized = normalize_digits(text)
    match = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", normalized)
    if match:
        year, month, day = map(int, match.groups())
        return date(year, month, day)

    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized.strip(), fmt).date()
        except ValueError:
            pass
    return None


def fetch_web_text(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(request, context=ssl._create_unverified_context(), timeout=30) as response:
        data = response.read()
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def numbers_from_text(text):
    return [int(value) for value in re.findall(r"\d+", normalize_digits(text))]


def extract_latest_loto6_result(page_text):
    tables = pd.read_html(StringIO(page_text))

    for table in tables:
        lines = []
        for _, row in table.iterrows():
            values = [normalize_digits(value) for value in row.tolist() if not pd.isna(value)]
            if values:
                lines.append(" ".join(values))

        whole_text = "\n".join(lines)
        round_match = re.search(r"第\s*(\d+)\s*回", whole_text)
        draw_date = parse_japanese_date(whole_text)
        if not round_match or not draw_date:
            continue

        main_numbers = []
        bonus_number = None
        for line in lines:
            candidates = [n for n in numbers_from_text(line) if 1 <= n <= 43]
            if "ボーナス" in line and candidates:
                bonus_number = candidates[-1]
            if ("本数字" in line or "抽せん数字" in line or "抽選数字" in line) and len(candidates) >= 6:
                main_numbers = candidates[:6]

        if len(main_numbers) == 6 and bonus_number:
            return {
                "round_no": int(round_match.group(1)),
                "draw_date": draw_date,
                "main_numbers": sorted(main_numbers),
                "bonus_number": bonus_number,
            }

    raise RuntimeError("最新当選番号を自動判定できませんでした。手入力フォームで登録してください。")


def upsert_official_result(round_no, draw_date, main_numbers, bonus_number, source, ball_set=""):
    official_results = read_csv(OFFICIAL_RESULTS_CSV, OFFICIAL_RESULT_COLUMNS)
    result_row = {
        "開催回": int(round_no),
        "抽せん日": draw_date.strftime("%Y/%m/%d") if hasattr(draw_date, "strftime") else str(draw_date),
        "本数字": numbers_to_text(main_numbers),
        "ボーナス数字": f"{int(bonus_number):02d}",
        "球セット": str(ball_set).strip().upper(),
        "登録元": source,
        "保存日時": now_text(),
    }

    if official_results.empty:
        official_results = pd.DataFrame([result_row])
    else:
        official_results["開催回"] = official_results["開催回"].map(to_int)
        same_round = official_results["開催回"] == int(round_no)
        if same_round.any():
            for key, value in result_row.items():
                official_results.loc[same_round, key] = value
        else:
            official_results = pd.concat([official_results, pd.DataFrame([result_row])], ignore_index=True)

    official_results["開催回"] = official_results["開催回"].map(to_int)
    official_results = official_results.sort_values("開催回")
    save_csv(official_results, OFFICIAL_RESULTS_CSV, OFFICIAL_RESULT_COLUMNS)


def upsert_result_row(round_no, draw_date, main_numbers, bonus_number):
    results = read_csv(RESULTS_CSV, RESULT_COLUMNS)
    sorted_numbers = sorted(main_numbers)
    result_row = {
        "開催回": round_no,
        "日付": draw_date.strftime("%Y/%m/%d"),
        "第1数字": sorted_numbers[0],
        "第2数字": sorted_numbers[1],
        "第3数字": sorted_numbers[2],
        "第4数字": sorted_numbers[3],
        "第5数字": sorted_numbers[4],
        "第6数字": sorted_numbers[5],
        "BONUS数字": bonus_number,
        "1等口数": "",
        "2等口数": "",
        "3等口数": "",
        "4等口数": "",
        "5等口数": "",
        "1等賞金": "",
        "2等賞金": "",
        "3等賞金": "",
        "4等賞金": "",
        "5等賞金": "",
        "キャリーオーバー": "",
    }

    if results.empty:
        results = pd.DataFrame([result_row])
        changed = True
    else:
        results["開催回"] = results["開催回"].map(to_int)
        same_round = results["開催回"] == round_no
        if same_round.any():
            current = results.loc[same_round].iloc[0]
            current_numbers = [to_int(current[column]) for column in NUMBER_COLUMNS]
            current_bonus = to_int(current["BONUS数字"])
            current_date = str(current["日付"])
            changed = current_numbers != sorted_numbers or current_bonus != bonus_number or current_date != result_row["日付"]
            for key, value in result_row.items():
                results.loc[same_round, key] = value
        else:
            results = pd.concat([results, pd.DataFrame([result_row])], ignore_index=True)
            changed = True

    results["開催回"] = results["開催回"].map(to_int)
    results = results.sort_values("開催回")
    save_csv(results, RESULTS_CSV, RESULT_COLUMNS)
    upsert_official_result(round_no, draw_date, sorted_numbers, bonus_number, "web")
    return changed


def update_latest_result_from_web():
    latest = extract_latest_loto6_result(fetch_web_text(MIZUHO_LOTO6_URL))
    changed = upsert_result_row(
        latest["round_no"],
        latest["draw_date"],
        latest["main_numbers"],
        latest["bonus_number"],
    )

    helper_messages = []
    for script in ("import_dream_backnumber_videos.py", "import_charlie_recent_loto6_sets.py"):
        try:
            run_helper_script(script)
            helper_messages.append(f"{script}: OK")
        except Exception as exc:
            helper_messages.append(f"{script}: {exc}")

    run_analysis()
    verified_count = verify_predictions_for_round(latest["round_no"])

    numbers = " - ".join(f"{number:02d}" for number in latest["main_numbers"])
    status = "更新しました" if changed else "すでに最新でした"
    return (
        f"第{latest['round_no']}回 {latest['draw_date'].strftime('%Y/%m/%d')} "
        f"{numbers} / BONUS {latest['bonus_number']:02d} を確認し、CSVを{status}。"
        f" 検証レポート{verified_count}件を更新しました。"
        + "\n"
        + "\n".join(helper_messages)
    )


def register_new_result(round_no, draw_date, main_numbers, bonus_number, ball_set):
    results = read_csv(RESULTS_CSV, RESULT_COLUMNS)
    sets = read_csv(SETS_CSV, SET_COLUMNS)
    sorted_numbers = sorted(main_numbers)

    if not results.empty and round_no in results["開催回"].map(to_int).tolist():
        upsert_official_result(round_no, draw_date, sorted_numbers, bonus_number, "manual", ball_set)
        return True, f"第{round_no}回は既存データを使って公式結果ログへ保存しました。"

    result_row = {
        "開催回": round_no,
        "日付": draw_date.strftime("%Y/%m/%d"),
        "第1数字": sorted_numbers[0],
        "第2数字": sorted_numbers[1],
        "第3数字": sorted_numbers[2],
        "第4数字": sorted_numbers[3],
        "第5数字": sorted_numbers[4],
        "第6数字": sorted_numbers[5],
        "BONUS数字": bonus_number,
        "1等口数": "",
        "2等口数": "",
        "3等口数": "",
        "4等口数": "",
        "5等口数": "",
        "1等賞金": "",
        "2等賞金": "",
        "3等賞金": "",
        "4等賞金": "",
        "5等賞金": "",
        "キャリーオーバー": "",
    }
    set_row = {
        "開催回": round_no,
        "日付": draw_date.strftime("%Y/%m/%d"),
        "球セット": ball_set.strip().upper(),
        "信頼度": "manual",
        "根拠": "手入力",
        "動画メタID": "",
        "メモ": "",
    }

    results = pd.concat([results, pd.DataFrame([result_row])], ignore_index=True)
    results["開催回"] = results["開催回"].map(to_int)
    results = results.sort_values("開催回")
    save_csv(results, RESULTS_CSV, RESULT_COLUMNS)
    upsert_official_result(round_no, draw_date, sorted_numbers, bonus_number, "manual", ball_set)

    sets = sets[sets["開催回"].map(to_int) != round_no] if not sets.empty else sets
    sets = pd.concat([sets, pd.DataFrame([set_row])], ignore_index=True)
    sets["開催回"] = sets["開催回"].map(to_int)
    sets = sets.sort_values("開催回")
    save_csv(sets, SETS_CSV, SET_COLUMNS)
    return True, f"第{round_no}回を登録しました。"


def import_latest_result_to_official_log():
    results = merge_official_results(read_csv(RESULTS_CSV, RESULT_COLUMNS))
    if results.empty:
        return "loto6.csv に抽せん結果がありません。"
    results["開催回"] = results["開催回"].map(to_int)
    latest = results.sort_values("開催回").tail(1).iloc[0]
    round_no = to_int(latest["開催回"])
    main_numbers = [to_int(latest[column]) for column in NUMBER_COLUMNS]
    bonus_number = to_int(latest["BONUS数字"])
    upsert_official_result(round_no, latest["日付"], main_numbers, bonus_number, "loto6.csv")
    verified_count = verify_predictions_for_round(round_no)
    return f"第{round_no}回を公式結果ログへ保存し、検証レポート{verified_count}件を更新しました。"


def count_consecutive_pairs(numbers):
    ordered = sorted(numbers)
    return sum(1 for left, right in zip(ordered, ordered[1:]) if right - left == 1)


def build_pick_reason(numbers, score_rank, target_sum, active_model_name=None):
    odd = sum(n % 2 for n in numbers)
    low = sum(n <= 21 for n in numbers)
    total = sum(numbers)
    consecutive = count_consecutive_pairs(numbers)
    top_hits = sum(1 for n in numbers if score_rank.get(n, 99) <= 10)
    model_text = f"{active_model_name}を反映し、" if active_model_name else ""
    return (
        f"{model_text}スコア上位数字を{top_hits}個入れ、奇数{odd}・偶数{6 - odd}、"
        f"低数字{low}・高数字{6 - low}、合計{total}を過去平均{target_sum:.1f}付近に調整。"
        f"連番は{consecutive}組に抑えています。"
    )


def generate_prediction_picks(scores, results, pick_count=3, active_model_key=None, target_round=None):
    if scores.empty or results.empty:
        return []

    score_df = scores.copy()
    score_df["数字"] = score_df["数字"].map(to_int)
    score_df["スコア"] = pd.to_numeric(score_df["スコア"], errors="coerce").fillna(0)
    score_df = score_df.sort_values("スコア", ascending=False)

    active_model_name = None
    if active_model_key in BACKTEST_MODELS and active_model_key != "random_baseline":
        active_model_name = BACKTEST_MODELS[active_model_key]
        target_round = target_round or (int(results["開催回"].map(to_int).max()) + 1)
        base_scores = dict(zip(score_df["数字"], score_df["スコア"]))
        history_scores = build_historical_scores(results, active_model_key, target_round)
        base_norm = normalize_score_map(base_scores)
        history_norm = normalize_score_map(history_scores)
        score_df["スコア"] = score_df["数字"].map(
            lambda number: base_norm.get(number, 0.0) * 70 + history_norm.get(number, 0.0) * 30
        )
        score_df = score_df.sort_values("スコア", ascending=False)

    candidate_df = score_df.head(18)
    candidate_numbers = candidate_df["数字"].tolist()
    score_map = dict(zip(candidate_df["数字"], candidate_df["スコア"]))
    score_rank = {num: rank + 1 for rank, num in enumerate(score_df["数字"].tolist())}

    sums = results[NUMBER_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    target_sum = float(sums.mean())
    sum_std = float(sums.std()) if len(sums) > 1 else 25.0
    if pd.isna(sum_std) or sum_std <= 0:
        sum_std = 25.0

    candidates = []
    for combo in combinations(candidate_numbers, 6):
        numbers = tuple(sorted(combo))
        consecutive = count_consecutive_pairs(numbers)
        if active_model_key != "non_overlap" and consecutive > 1:
            continue
        odd = sum(n % 2 for n in numbers)
        low = sum(n <= 21 for n in numbers)
        high_band = sum(32 <= n <= 43 for n in numbers)
        if active_model_key == "non_overlap" and high_band < 1:
            continue
        if active_model_key != "non_overlap" and (odd not in (2, 3, 4) or low not in (2, 3, 4)):
            continue

        total = sum(numbers)
        score_total = sum(score_map[n] for n in numbers)
        balance_penalty = abs(odd - 3) * 5 + abs(low - 3) * 5
        sum_penalty = abs(total - target_sum) / max(sum_std, 1) * 12
        consecutive_penalty = 0 if active_model_key == "non_overlap" else consecutive * 4
        high_bonus = high_band * 3 if active_model_key == "non_overlap" else 0
        ranking_score = score_total - balance_penalty - sum_penalty - consecutive_penalty
        ranking_score += high_bonus
        candidates.append((ranking_score, numbers))

    candidates.sort(reverse=True, key=lambda item: item[0])

    picks = []
    used_numbers = set()
    for _, numbers in candidates:
        overlap = len(set(numbers) & used_numbers)
        if picks and overlap > 3:
            continue
        picks.append(
            {
                "numbers": numbers,
                "reason": build_pick_reason(numbers, score_rank, target_sum, active_model_name),
            }
        )
        used_numbers.update(numbers)
        if len(picks) == pick_count:
            break

    if len(picks) < pick_count:
        existing = {pick["numbers"] for pick in picks}
        for _, numbers in candidates:
            if numbers in existing:
                continue
            picks.append(
                {
                    "numbers": numbers,
                    "reason": build_pick_reason(numbers, score_rank, target_sum, active_model_name),
                }
            )
            if len(picks) == pick_count:
                break

    return picks


def row_main_numbers(row):
    return sorted(to_int(row[column]) for column in NUMBER_COLUMNS)


def build_historical_scores(history, model_key, target_round):
    if history.empty:
        return {number: 0.0 for number in range(1, 44)}

    sorted_history = history.sort_values("開催回").copy()
    number_rows = [row_main_numbers(row) for _, row in sorted_history.iterrows()]
    bonus_rows = [[to_int(row.get("BONUS数字"))] for _, row in sorted_history.iterrows()]
    return build_model_scores(number_rows, model_key, 43, 6, target_round, bonus_rows)


def historical_context_before_round(round_no):
    history = merge_official_results(read_csv(RESULTS_CSV, RESULT_COLUMNS))
    if history.empty:
        return [], []
    history = history.copy()
    history["開催回"] = history["開催回"].map(to_int)
    history = history[history["開催回"] < int(round_no)].sort_values("開催回")
    number_rows = [row_main_numbers(row) for _, row in history.iterrows()]
    bonus_rows = [[to_int(row.get("BONUS数字"))] for _, row in history.iterrows()]
    return number_rows, bonus_rows


def build_support_map_for_prediction(predicted, round_no, selected_model=None):
    number_rows, bonus_rows = historical_context_before_round(round_no)
    return build_model_support_map(
        predicted,
        number_rows,
        number_max=43,
        draw_size=6,
        target_round=round_no,
        bonus_rows=bonus_rows,
        selected_model=selected_model,
    )


def build_model_pick_reason(model_key, numbers, target_sum):
    summary = balance_summary(numbers)
    model_name = BACKTEST_MODELS.get(model_key, model_key)
    return (
        f"{model_name}で候補化し、奇数{summary['odd']}・偶数{summary['even']}、"
        f"低数字{summary['low']}・高数字{summary['high']}、合計{summary['sum']}を"
        f"過去平均{target_sum:.1f}付近として検証。"
    )


def generate_backtest_picks(history, model_key, target_round, pick_count=3):
    if history.empty:
        return []

    number_scores = build_historical_scores(history, model_key, target_round)
    score_items = sorted(number_scores.items(), key=lambda item: item[1], reverse=True)
    candidate_numbers = [number for number, _ in score_items[:12]]
    score_map = dict(score_items)

    sums = history[NUMBER_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    target_sum = float(sums.mean())
    sum_std = float(sums.std()) if len(sums) > 1 else 25.0
    if pd.isna(sum_std) or sum_std <= 0:
        sum_std = 25.0

    candidates = []
    for combo in combinations(candidate_numbers, 6):
        numbers = tuple(sorted(combo))
        consecutive = count_consecutive_pairs(numbers)
        if model_key != "non_overlap" and consecutive > 1:
            continue
        summary = balance_summary(numbers)
        high_band = sum(32 <= n <= 43 for n in numbers)
        if model_key == "non_overlap" and high_band < 1:
            continue
        if model_key != "non_overlap" and (summary["odd"] not in (2, 3, 4) or summary["low"] not in (2, 3, 4)):
            continue

        score_total = sum(score_map[n] for n in numbers)
        balance_penalty = abs(summary["odd"] - 3) * 5 + abs(summary["low"] - 3) * 5
        sum_penalty = abs(summary["sum"] - target_sum) / max(sum_std, 1) * 12
        consecutive_penalty = 0 if model_key == "non_overlap" else consecutive * 4
        high_bonus = high_band * 3 if model_key == "non_overlap" else 0
        ranking_score = score_total - balance_penalty - sum_penalty - consecutive_penalty
        ranking_score += high_bonus
        candidates.append((ranking_score, numbers))

    candidates.sort(reverse=True, key=lambda item: item[0])

    picks = []
    used_numbers = set()
    for _, numbers in candidates:
        if picks and len(set(numbers) & used_numbers) > 3:
            continue
        picks.append(
            {
                "numbers": numbers,
                "reason": build_model_pick_reason(model_key, numbers, target_sum),
            }
        )
        used_numbers.update(numbers)
        if len(picks) == pick_count:
            break

    if len(picks) < pick_count:
        existing = {pick["numbers"] for pick in picks}
        for _, numbers in candidates:
            if numbers in existing:
                continue
            picks.append(
                {
                    "numbers": numbers,
                    "reason": build_model_pick_reason(model_key, numbers, target_sum),
                }
            )
            if len(picks) == pick_count:
                break
    return picks


def run_backtest(results, lookback_rounds=50, min_training_rounds=30):
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
            "本数字": numbers_to_text(row_main_numbers(target_row)),
            "ボーナス数字": f"{to_int(target_row['BONUS数字']):02d}",
        }

        for model_key, model_name in BACKTEST_MODELS.items():
            picks = generate_backtest_picks(train_history, model_key, target_round)
            for index, pick in enumerate(picks, start=1):
                prediction_row = {
                    "予想ID": f"BT-{model_key}-{target_round}-{index}",
                    "開催回": target_round,
                    "予想番号": numbers_to_text(pick["numbers"]),
                    "使用モデル": model_name,
                }
                report = build_verification_report_row(prediction_row, official_row)
                detail_rows.append(
                    {
                        "開催回": target_round,
                        "モデル": model_name,
                        "候補番号": index,
                        "予想番号": report["予想番号"],
                        "当選番号": report["実際の当選番号"],
                        "一致数": report["本数字一致数"],
                        "的中率": report["的中率"],
                        "勝率": report["勝率"],
                        "期待値": report["期待値"],
                        "近接一致数": report["近接一致数"],
                        "ボーナス一致": report["ボーナス一致"],
                        "合計値ズレ": report["合計値のズレ"],
                        "等級判定": report["等級判定"],
                    }
                )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return pd.DataFrame(), detail_df

    summary_rows = []
    for model_name, model_rows in detail_df.groupby("モデル"):
        round_best = model_rows.groupby("開催回")["一致数"].max()
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
                "平均合計値ズレ": round(float(model_rows["合計値ズレ"].abs().mean()), 3),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["3口内最高一致平均", "3個以上一致率", "平均近接一致数"],
        ascending=False,
    )
    return summary_df, detail_df.sort_values(["開催回", "モデル", "候補番号"], ascending=[False, True, True])


def run_pre_prediction_research(results):
    if results.empty or len(results) <= 30:
        return pd.DataFrame(), pd.DataFrame(), None
    lookback_rounds = min(5, max(3, len(results) - 30))
    summary_df, detail_df = run_backtest(results, lookback_rounds=lookback_rounds)
    best_model_key = best_actionable_model_key(summary_df)
    if best_model_key:
        save_active_model_setting(best_model_key, f"予想生成前の履歴分析・直近{lookback_rounds}回評価")
    return summary_df, detail_df, best_model_key


def get_pre_prediction_research(results):
    if results.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    latest_round = int(results["開催回"].map(to_int).max())
    cache_key = f"{latest_round}:{len(results)}"
    if st.session_state.get("loto6_pre_research_key") != cache_key:
        summary_df, detail_df, best_model_key = run_pre_prediction_research(results)
        st.session_state["loto6_pre_research_key"] = cache_key
        st.session_state["loto6_pre_research_summary"] = summary_df
        st.session_state["loto6_pre_research_detail"] = detail_df
        st.session_state["loto6_pre_research_best"] = best_model_key
    return (
        st.session_state.get("loto6_pre_research_summary", pd.DataFrame()),
        st.session_state.get("loto6_pre_research_detail", pd.DataFrame()),
        st.session_state.get("loto6_pre_research_best"),
    )


def save_prediction_picks(picks, target_round, model_name="score_balance_v1"):
    predictions = read_csv(PREDICTIONS_CSV, PREDICTION_COLUMNS)
    existing_keys = set()
    if not predictions.empty:
        for _, row in predictions.iterrows():
            existing_keys.add(
                (
                    to_int(row.get("開催回")),
                    to_int(row.get("候補番号")),
                    str(row.get("予想番号")),
                    str(row.get("使用モデル")),
                )
            )

    rows = []
    prediction_date = today_text()
    for index, pick in enumerate(picks, start=1):
        number_text = numbers_to_text(pick["numbers"])
        key = (int(target_round), index, number_text, model_name)
        if key in existing_keys:
            continue
        rows.append(
            {
                "予想ID": f"{int(target_round)}-{prediction_date.replace('/', '')}-{index}",
                "開催回": int(target_round),
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
        predictions = predictions.sort_values(["開催回", "候補番号", "保存日時"])
        save_csv(predictions, PREDICTIONS_CSV, PREDICTION_COLUMNS)
    return len(rows)


def determine_prize_grade(match_count, bonus_match):
    if match_count == 6:
        return "1等"
    if match_count == 5 and bonus_match:
        return "2等"
    if match_count == 5:
        return "3等"
    if match_count == 4:
        return "4等"
    if match_count == 3:
        return "5等"
    return "該当なし"


def count_near_matches(predicted, actual):
    actual_only = set(actual) - set(predicted)
    predicted_only = set(predicted) - set(actual)
    near_matches = 0
    used_actual = set()
    for number in predicted_only:
        nearest = [actual_number for actual_number in actual_only if abs(number - actual_number) <= 2 and actual_number not in used_actual]
        if nearest:
            closest = min(nearest, key=lambda actual_number: abs(number - actual_number))
            used_actual.add(closest)
            near_matches += 1
    return near_matches


def balance_summary(numbers):
    odd = sum(number % 2 for number in numbers)
    low = sum(number <= 21 for number in numbers)
    return {
        "odd": odd,
        "even": len(numbers) - odd,
        "low": low,
        "high": len(numbers) - low,
        "sum": sum(numbers),
        "consecutive": count_consecutive_pairs(numbers),
        "last_digits": [number % 10 for number in numbers],
    }


def describe_last_digit_bias(numbers):
    counts = pd.Series([number % 10 for number in numbers]).value_counts()
    biased_digits = [f"{int(digit)}が{int(count)}個" for digit, count in counts.items() if count >= 2]
    return " / ".join(biased_digits) if biased_digits else "大きな偏りなし"


def build_failure_factors(predicted, actual, match_count, bonus_match):
    predicted_summary = balance_summary(predicted)
    actual_summary = balance_summary(actual)
    factors = []
    if match_count <= 2:
        factors.append("中心候補と実際の当選数字の重なりが少ない")
    if abs(predicted_summary["sum"] - actual_summary["sum"]) >= 20:
        factors.append("合計値のレンジが実際と離れた")
    if abs(predicted_summary["odd"] - actual_summary["odd"]) >= 2:
        factors.append("奇数偶数バランスが実際とずれた")
    if abs(predicted_summary["low"] - actual_summary["low"]) >= 2:
        factors.append("高低バランスが実際とずれた")
    if predicted_summary["consecutive"] != actual_summary["consecutive"]:
        factors.append("連番パターンを合わせきれなかった")
    if not bonus_match and match_count == 5:
        factors.append("ボーナス数字の補足に届かなかった")
    return " / ".join(factors) if factors else "大きな構造差は小さく、候補選択の微差が影響"


def build_improvement_plan(predicted, actual):
    predicted_summary = balance_summary(predicted)
    actual_summary = balance_summary(actual)
    improvements = []
    sum_gap = actual_summary["sum"] - predicted_summary["sum"]
    if abs(sum_gap) >= 10:
        direction = "上げる" if sum_gap > 0 else "下げる"
        improvements.append(f"合計値の目標帯を{abs(sum_gap)}程度{direction}検証する")
    odd_gap = actual_summary["odd"] - predicted_summary["odd"]
    if odd_gap:
        direction = "奇数寄り" if odd_gap > 0 else "偶数寄り"
        improvements.append(f"奇数偶数を{direction}に1枠調整する仮説を試す")
    low_gap = actual_summary["low"] - predicted_summary["low"]
    if low_gap:
        direction = "低数字寄り" if low_gap > 0 else "高数字寄り"
        improvements.append(f"高低バランスを{direction}に寄せる")
    if actual_summary["consecutive"] > predicted_summary["consecutive"]:
        improvements.append("連番を完全には除外せず、1組を許容する条件を比較する")
    if not improvements:
        improvements.append("現行条件を維持し、スコア上位候補の入れ替え幅を小さく検証する")
    return " / ".join(improvements)


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


def build_verification_report_row(prediction_row, official_row, support_map=None):
    predicted = parse_number_text(prediction_row["予想番号"])
    actual = parse_number_text(official_row["本数字"])
    bonus = parse_number_text(official_row["ボーナス数字"])
    bonus_number = bonus[0] if bonus else 0
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

    bonus_match = bonus_number in predicted and bonus_number not in actual
    match_count = len(matched)
    grade = determine_prize_grade(match_count, bonus_match)
    hit_rate = round(match_count / max(len(actual), 1) * 100, 1)
    win_rate = 0.0 if grade == "該当なし" else 100.0
    expected_value = round((match_count + (0.5 if bonus_match else 0.0)) / 6, 3)
    pair_match_count = len(list(combinations(matched, 2))) if match_count >= 2 else 0
    odd_even_gap = f"予想 {predicted_summary['odd']}:{predicted_summary['even']} / 実際 {actual_summary['odd']}:{actual_summary['even']}"
    high_low_gap = f"予想 低{predicted_summary['low']}:高{predicted_summary['high']} / 実際 低{actual_summary['low']}:高{actual_summary['high']}"
    consecutive_text = f"予想{predicted_summary['consecutive']}組 / 実際{actual_summary['consecutive']}組"
    sum_gap = actual_summary["sum"] - predicted_summary["sum"]
    odd_even_diff = actual_summary["odd"] - predicted_summary["odd"]
    high_low_diff = actual_summary["low"] - predicted_summary["low"]
    condition_gap = build_missing_excess_conditions(predicted, actual, 43)

    return {
        "予想ID": prediction_row["予想ID"],
        "開催回": to_int(prediction_row["開催回"]),
        "検証日": today_text(),
        "使用モデル": prediction_row.get("使用モデル", ""),
        "予想番号": numbers_to_text(predicted),
        "実際の当選番号": numbers_to_text(actual),
        "ボーナス数字": f"{bonus_number:02d}" if bonus_number else "",
        "本数字一致数": match_count,
        "的中率": hit_rate,
        "ボーナス一致": "あり" if bonus_match else "なし",
        "勝率": win_rate,
        "期待値": expected_value,
        "等級判定": grade,
        "近接一致数": count_near_matches(predicted, actual),
        "合計値差": sum_gap,
        "奇数偶数差": odd_even_diff,
        "高低差": high_low_diff,
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
        "失敗要因": build_failure_factors(predicted, actual, match_count, bonus_match),
        "逆算分析": build_reverse_analysis(predicted, actual),
        "改善案": build_improvement_plan(predicted, actual),
        "次回の仮説": build_next_hypothesis(predicted, actual),
    }


def verify_predictions_for_round(round_no=None):
    predictions = read_csv(PREDICTIONS_CSV, PREDICTION_COLUMNS)
    official_results = read_csv(OFFICIAL_RESULTS_CSV, OFFICIAL_RESULT_COLUMNS)
    if predictions.empty or official_results.empty:
        return 0

    predictions["開催回"] = predictions["開催回"].map(to_int)
    official_results["開催回"] = official_results["開催回"].map(to_int)
    if round_no is not None:
        predictions = predictions[predictions["開催回"] == int(round_no)]
        official_results = official_results[official_results["開催回"] == int(round_no)]

    report_rows = []
    contribution_rows = []
    for _, prediction_row in predictions.iterrows():
        result_matches = official_results[official_results["開催回"] == to_int(prediction_row["開催回"])]
        if result_matches.empty:
            continue
        official_row = result_matches.tail(1).iloc[0]
        predicted = parse_number_text(prediction_row["予想番号"])
        actual = parse_number_text(official_row["本数字"])
        support_map = build_support_map_for_prediction(
            predicted,
            to_int(prediction_row["開催回"]),
            selected_model=prediction_row.get("使用モデル", ""),
        )
        report_rows.append(build_verification_report_row(prediction_row, official_row, support_map))
        contribution_rows.extend(
            build_contribution_rows(
                "ロト6研究所",
                prediction_row,
                predicted,
                actual,
                support_map,
                saved_at=now_text(),
            )
        )

    reports = read_csv(VERIFICATION_REPORTS_CSV, VERIFICATION_COLUMNS)
    if not reports.empty and report_rows:
        existing_ids = {str(row["予想ID"]) for _, row in pd.DataFrame(report_rows).iterrows()}
        reports = reports[~reports["予想ID"].astype(str).isin(existing_ids)]

    if report_rows:
        reports = pd.concat([reports, pd.DataFrame(report_rows)], ignore_index=True)
        reports = add_verification_metrics(reports, draw_size=6)
        reports["開催回"] = reports["開催回"].map(to_int)
        reports = reports.sort_values(["開催回", "予想ID"])
        save_csv(reports, VERIFICATION_REPORTS_CSV, VERIFICATION_COLUMNS)
    if contribution_rows:
        contributions = read_csv(CONTRIBUTIONS_CSV, CONTRIBUTION_COLUMNS)
        contributions = merge_contribution_rows(contributions, contribution_rows)
        save_csv(contributions, CONTRIBUTIONS_CSV, CONTRIBUTION_COLUMNS)
    if report_rows:
        cycles = read_csv(RESEARCH_CYCLES_CSV, RESEARCH_CYCLE_COLUMNS)
        cycles = merge_research_cycle_rows(cycles, build_research_cycle_rows("ロト6研究所", report_rows, saved_at=now_text()))
        save_csv(cycles, RESEARCH_CYCLES_CSV, RESEARCH_CYCLE_COLUMNS)
    return len(report_rows)


def render_registration_form():
    st.sidebar.subheader("新しい当選結果を登録")
    st.sidebar.caption("実際の当選結果はここに入力します。球セットは文字入力欄です。")
    if st.sidebar.button("表示中の最新回を公式結果ログに保存", key="import_latest_loto6_result"):
        try:
            st.session_state["registration_message"] = import_latest_result_to_official_log()
            rerun_app()
        except Exception as exc:
            st.sidebar.error(str(exc))
    with st.sidebar.form("new_loto6_result_form"):
        existing_results = merge_official_results(read_csv(RESULTS_CSV, RESULT_COLUMNS))
        next_round = 1
        if not existing_results.empty:
            next_round = int(existing_results["開催回"].map(to_int).max()) + 1

        round_no = st.number_input("開催回", min_value=1, value=next_round, step=1)
        draw_date = st.date_input("日付", value=date.today())
        number_cols = st.columns(3)
        main_numbers = [
            number_cols[index % 3].number_input(
                f"本数字{index + 1}",
                min_value=1,
                max_value=43,
                value=index + 1,
                step=1,
                key=f"main_number_{index + 1}",
            )
            for index in range(6)
        ]
        bonus_number = st.number_input("ボーナス数字", min_value=1, max_value=43, value=7, step=1)
        ball_set = st.text_input("球セット", value="", placeholder="例: A")
        submitted = st.form_submit_button("登録して再分析")

    if not submitted:
        return

    main_numbers = [int(n) for n in main_numbers]
    bonus_number = int(bonus_number)
    if len(set(main_numbers)) != 6:
        st.sidebar.error("本数字6個は重複しない数字で入力してください。")
        return
    if bonus_number in main_numbers:
        st.sidebar.error("ボーナス数字は本数字と重複しない数字で入力してください。")
        return
    if not ball_set.strip():
        st.sidebar.error("球セットを入力してください。")
        return

    try:
        saved, message = register_new_result(int(round_no), draw_date, main_numbers, bonus_number, ball_set)
        if not saved:
            st.sidebar.warning(message)
            return
        with st.spinner("登録後の再分析を実行しています..."):
            run_analysis()
            verify_predictions_for_round(int(round_no))
        st.session_state["registration_message"] = message
        rerun_app()
    except Exception as exc:
        st.sidebar.error(str(exc))


def render_prediction_picks(scores, results, target_round):
    st.markdown("**研究フロー: 当選番号追加 → 結果分析 → 反省履歴保存 → 履歴分析 → モデル評価 → 改善条件抽出 → 次回予想生成**")
    with st.spinner("予想生成前の履歴分析とモデル評価を実行しています..."):
        pre_summary, _, best_model_key = get_pre_prediction_research(results)
    active_setting = read_active_model_setting()
    active_model_key = best_model_key or (active_setting["model_key"] if active_setting else None)
    active_model_name = BACKTEST_MODELS.get(active_model_key, active_setting["model_name"] if active_setting else "score_balance_v1")
    picks = generate_prediction_picks(scores, results, active_model_key=active_model_key, target_round=target_round)
    st.subheader("次回予想買い目")
    if not picks:
        st.info("買い目生成に必要な候補スコアまたは抽せん結果が不足しています。")
        return

    if best_model_key and not pre_summary.empty:
        best_row = pre_summary[pre_summary["モデル"] == BACKTEST_MODELS[best_model_key]].iloc[0]
        st.caption(
            f"履歴分析済み: {active_model_name}を暫定採用。"
            f"3口内最高一致平均 {best_row['3口内最高一致平均']} / 3個以上一致率 {best_row['3個以上一致率']}%。"
        )
    elif active_setting:
        st.caption(f"反映中の研究モデル: {active_model_name}（{active_setting['reason']}）")
    else:
        st.caption("反映中の研究モデル: 標準スコアバランス")

    saved_count = save_prediction_picks(picks, target_round, active_model_name)
    if saved_count:
        st.caption(f"予測研究所ログ: 第{target_round}回の予想{saved_count}件を predictions.csv に保存しました。")
    else:
        st.caption(f"予測研究所ログ: 第{target_round}回の同一予想は保存済みです。")

    for index, pick in enumerate(picks, start=1):
        numbers = " - ".join(f"{n:02d}" for n in pick["numbers"])
        st.markdown(f"**第{index}候補：{numbers}**")
        st.write(f"理由：{pick['reason']}")


def render_prediction_lab():
    st.subheader("予測研究所システム")
    st.caption("当選や利益を保証するものではなく、予測手法を検証するための研究開発ログです。")
    st.info(
        "このバックテストは、過去データに対して各予想モデルの傾向を比較する研究目的の検証です。\n\n"
        "表示される一致数や改善指標は、将来の当選や利益を保証するものではありません。\n\n"
        "本システムの目的は、予想、結果、検証、改善のサイクルを通じて、予測手法の有効性を継続的に研究することです。"
    )

    predictions = read_csv(PREDICTIONS_CSV, PREDICTION_COLUMNS)
    official_results = read_csv(OFFICIAL_RESULTS_CSV, OFFICIAL_RESULT_COLUMNS)
    reports = read_csv(VERIFICATION_REPORTS_CSV, VERIFICATION_COLUMNS)
    reports = add_verification_metrics(reports, draw_size=6)
    contributions = read_csv(CONTRIBUTIONS_CSV, CONTRIBUTION_COLUMNS)

    metric_cols = st.columns(4)
    metric_cols[0].metric("予想履歴", len(predictions))
    metric_cols[1].metric("公式結果ログ", len(official_results))
    metric_cols[2].metric("検証レポート", len(reports))
    metric_cols[3].metric("貢献度ログ", len(contributions))

    if not official_results.empty:
        st.markdown("**最近の公式結果ログ**")
        st.dataframe(official_results.sort_values("開催回", ascending=False).head(10), width="stretch", hide_index=True)

    if st.button("予想履歴と公式結果を照合して検証レポート更新"):
        try:
            updated_count = verify_predictions_for_round()
            st.success(f"検証レポートを{updated_count}件更新しました。")
            reports = read_csv(VERIFICATION_REPORTS_CSV, VERIFICATION_COLUMNS)
            reports = add_verification_metrics(reports, draw_size=6)
        except Exception as exc:
            st.error(str(exc))

    lab_tabs = st.tabs(["バックテスト", "検証レポート", "モデル貢献度", "条件別成功率", "AI改善レポート", "動画仮説研究", "予想履歴", "公式結果"])
    with lab_tabs[0]:
        st.markdown("**新フロー**")
        st.dataframe(build_research_flow_table(), width="stretch", hide_index=True)
        if len(results) <= 30:
            st.info("バックテストには最低31回分以上の抽せん結果が必要です。")
        else:
            max_rounds = max(10, min(50, len(results) - 30))
            default_rounds = min(20, max_rounds)
            lookback_rounds = st.slider("バックテスト対象回数", 10, max_rounds, default_rounds)
            if st.button("バックテストを実行"):
                with st.spinner("過去データでモデル別成績を検証しています..."):
                    summary_df, detail_df = run_backtest(results, lookback_rounds=lookback_rounds)
                st.session_state["backtest_summary"] = summary_df
                st.session_state["backtest_detail"] = detail_df
                best_model_key = best_actionable_model_key(summary_df)
                if best_model_key:
                    reason = f"直近{lookback_rounds}回バックテストで最上位"
                    save_active_model_setting(best_model_key, reason)
                    st.session_state["active_model_notice"] = f"{BACKTEST_MODELS[best_model_key]}を次回予想に反映しました。"
                    rerun_app()
            summary_df = st.session_state.get("backtest_summary", pd.DataFrame())
            detail_df = st.session_state.get("backtest_detail", pd.DataFrame())
            if "active_model_notice" in st.session_state:
                st.success(st.session_state.pop("active_model_notice"))
            if summary_df.empty:
                st.info("対象回数を選んでバックテストを実行してください。")
            else:
                st.markdown("**モデル別成績表**")
                st.dataframe(summary_df, width="stretch", hide_index=True)
                best_model_key = best_actionable_model_key(summary_df)
                best_model = summary_df[summary_df["モデル"] == BACKTEST_MODELS[best_model_key]].iloc[0] if best_model_key else summary_df.iloc[0]
                st.write(
                    f"現在の比較では「{best_model['モデル']}」が、"
                    f"3口内最高一致平均 {best_model['3口内最高一致平均']}、"
                    f"3個以上一致率 {best_model['3個以上一致率']}% で最上位です。"
                )
                active_setting = read_active_model_setting()
                if active_setting:
                    st.write(f"次回予想へ反映中: {active_setting['model_name']}（{active_setting['reason']}）")
                st.caption("この結果は過去データ上の検証であり、将来の当選や利益を保証するものではありません。")
                with st.expander("回別バックテスト詳細"):
                    st.dataframe(detail_df, width="stretch", hide_index=True)
    with lab_tabs[1]:
        if reports.empty:
            st.info("検証レポートはまだありません。予想した開催回の当選結果が登録されると作成できます。")
        else:
            sorted_reports = reports.sort_values("開催回", ascending=False)
            st.dataframe(sorted_reports, width="stretch", hide_index=True)
            latest_report = sorted_reports.iloc[0]
            st.markdown("**直近の検証サマリー**")
            st.write(f"予想番号: {latest_report['予想番号']} / 実際: {latest_report['実際の当選番号']}")
            st.write(f"一致数: {latest_report['本数字一致数']} / ボーナス一致: {latest_report['ボーナス一致']} / 判定: {latest_report['等級判定']}")
            st.write(f"失敗要因: {latest_report['失敗要因']}")
            st.write(f"的中要因: {latest_report.get('的中要因', '-')}")
            st.write(f"足りなかった条件: {latest_report.get('足りなかった条件', '-')}")
            st.write(f"過剰だった条件: {latest_report.get('過剰だった条件', '-')}")
            st.write(f"逆算分析: {latest_report['逆算分析']}")
            st.write(f"改善案: {latest_report['改善案']}")
            st.write(f"次回の仮説: {latest_report['次回の仮説']}")
    with lab_tabs[2]:
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
    with lab_tabs[3]:
        condition_df = build_condition_success_table(reports, number_max=43)
        if condition_df.empty:
            st.info("条件別成功率は、検証レポート作成後に表示されます。")
        else:
            st.dataframe(condition_df, width="stretch", hide_index=True)
    with lab_tabs[4]:
        summary = build_ai_improvement_summary(reports)
        cols = st.columns(3)
        cols[0].metric("一致数", summary["一致数"])
        cols[1].metric("予想", summary["予想"])
        cols[2].metric("結果", summary["結果"])
        st.write(f"的中要因: {summary['的中要因']}")
        st.write(f"外れ要因: {summary['外れ要因']}")
        st.write(f"改善案: {summary['改善案']}")
        st.write(f"次回仮説: {summary['次回仮説']}")
    with lab_tabs[5]:
        video_logs = read_csv(VIDEO_HYPOTHESES_CSV, VIDEO_HYPOTHESIS_COLUMNS)
        with st.form("loto6_video_hypothesis_form"):
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
    with lab_tabs[6]:
        if predictions.empty:
            st.info("次回予想買い目が表示されると predictions.csv に保存されます。")
        else:
            st.dataframe(predictions.sort_values(["開催回", "候補番号"], ascending=[False, True]), width="stretch", hide_index=True)
    with lab_tabs[7]:
        if official_results.empty:
            st.info("当選結果を登録すると results.csv に研究所用の公式結果ログを保存します。")
        else:
            st.dataframe(official_results.sort_values("開催回", ascending=False), width="stretch", hide_index=True)


render_registration_form()

if "registration_message" in st.session_state:
    st.success(st.session_state.pop("registration_message"))

st.info("当選結果を入力する場合は、画面左側のサイドバーにある「新しい当選結果を登録」を使ってください。サイドバーが見えない場合は、左上の矢印で開けます。")

action_cols = st.columns(2)
if action_cols[0].button("最新当選番号を取得してCSV更新"):
    try:
        with st.spinner("最新当選番号を取得し、CSV更新と再分析を実行しています..."):
            message = update_latest_result_from_web()
        st.success(message)
    except Exception as exc:
        st.error(str(exc))

if action_cols[1].button("手元のCSVで再分析"):
    try:
        with st.spinner("分析を更新しています..."):
            run_analysis()
        st.success("分析を更新しました。")
    except Exception as exc:
        st.error(str(exc))


results = merge_official_results(read_csv(RESULTS_CSV, RESULT_COLUMNS))
sets = read_csv(SETS_CSV, SET_COLUMNS)
scores = read_csv(SCORES_CSV)

if results.empty:
    st.info("抽せん結果CSVがまだありません。サイドバーから新しい当選結果を登録してください。")
    st.stop()

latest = results.sort_values("開催回").tail(1).iloc[0]
latest_round = int(latest["開催回"])
if sets.empty:
    latest_set = None
else:
    set_matches = sets[sets["開催回"].map(to_int) == latest_round]
    latest_set = set_matches.tail(1).iloc[0] if not set_matches.empty else None

st.subheader("最新回")
cols = st.columns(4)
cols[0].metric("開催回", int(latest["開催回"]))
cols[1].metric("日付", latest["日付"])
cols[2].metric("球セット", latest_set["球セット"] if latest_set is not None else "-")
cols[3].metric("信頼度", latest_set["信頼度"] if latest_set is not None else "-")

main_numbers = [int(latest[column]) for column in NUMBER_COLUMNS]
bonus = int(latest["BONUS数字"])
st.info(f"本数字: {' - '.join(f'{n:02d}' for n in main_numbers)} / BONUS: {bonus:02d}")

st.subheader("次回候補スコア 上位")
top_count = st.slider("表示件数", 6, 30, 15)
if scores.empty:
    st.info("候補スコアCSVがありません。最新データで再分析してください。")
else:
    st.dataframe(scores.head(top_count), width="stretch", hide_index=True)
    render_prediction_picks(scores, results, latest_round + 1)

render_prediction_lab()

st.subheader("球セット履歴")
history_count = st.slider("履歴表示件数", 10, 80, 30)
if sets.empty:
    st.info("球セット履歴はまだありません。")
else:
    st.dataframe(sets.sort_values("開催回", ascending=False).head(history_count), width="stretch", hide_index=True)

st.subheader("最近の抽せん結果")
st.dataframe(results.sort_values("開催回", ascending=False).head(history_count), width="stretch", hide_index=True)

st.subheader("候補数字メモ")
if scores.empty:
    st.info("候補数字メモを表示するには、候補スコアCSVが必要です。")
else:
    candidate_numbers = scores.head(15)["数字"].astype(int).tolist()
    st.write("上位15個:", " ".join(f"{n:02d}" for n in candidate_numbers))
    st.write("上位6個:", " ".join(f"{n:02d}" for n in candidate_numbers[:6]))
