from datetime import date, datetime
import hashlib
from itertools import combinations
from io import StringIO
from pathlib import Path
import re
import ssl
import subprocess
import sys
import urllib.request

APP_DIR = Path(__file__).resolve().parent
LOTO_LAB_DIR = APP_DIR.parent
CORE_DIR = LOTO_LAB_DIR / "core"
DATA_DIR = LOTO_LAB_DIR / "data"
VERIFICATION_DIR = DATA_DIR / "verification"
AI_IMPROVEMENT_DIR = DATA_DIR / "ai_improvement"

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import pandas as pd
import streamlit as st

from arl_research_engine import (
    ANTI_POPULAR_EXPECTED_VALUE_KEY,
    ANTI_POPULAR_EXPECTED_VALUE_LABEL,
    ARL_MODEL_LABELS,
    BACKTEST_SUMMARY_COLUMNS,
    BALANCE_PREDICTION_COLUMNS,
    BALANCE_VERIFICATION_COLUMNS,
    CHAMINI6_GOD_MODE_KEY,
    CHAMINI6_GOD_MODE_LABEL,
    CONTINUOUS_WIN_RESEARCH_COLUMNS,
    CONTRIBUTION_COLUMNS,
    ENSEMBLE_PREDICTION_COLUMNS,
    MODEL_WEIGHT_HISTORY_COLUMNS,
    POPULARITY_SCORE_COLUMNS,
    PURCHASE_COLUMNS,
    RESEARCH_CYCLE_COLUMNS,
    TICKET_STRATEGY_COLUMNS,
    VIDEO_HYPOTHESIS_COLUMNS,
    add_verification_metrics,
    ai_improvement_weight_rows,
    anti_popular_verification_fields,
    attach_balance_verification_fields,
    apply_ai_improvement_weights,
    apply_prediction_pattern_roles,
    append_winning_condition_history,
    build_ai_improvement_summary,
    build_ai_improvement_weight_summary,
    build_balance_hypothesis_performance,
    build_condition_success_table,
    build_contribution_ranking,
    build_contribution_rows,
    build_continuous_win_research_rows,
    build_enhanced_ai_improvement_report,
    build_ensemble_prediction_rows,
    build_high_prize_backtest_summary,
    build_high_prize_ticket_strategy,
    build_fixed_prediction_overview,
    build_research_cycle_rows,
    build_effective_conditions,
    build_hit_factor_summary,
    build_missing_excess_conditions,
    build_set_ball_analysis,
    build_model_dashboard,
    build_model_scores,
    build_model_support_map,
    build_popularity_score_rows,
    build_purchase_group_summary,
    build_purchase_summary,
    build_research_flow_table,
    build_chamini_sp_performance_summary,
    build_ticket_strategy_rows,
    build_unverified_chamini_sp_predictions,
    build_winning_condition_report,
    balance_prediction_fields_from_pick,
    evaluate_purchase_history,
    extract_video_hypothesis,
    format_contribution_detail,
    generate_anti_popular_expected_value_picks,
    generate_chamini6_god_mode_picks,
    load_winning_condition_history,
    merge_contribution_rows,
    merge_research_cycle_rows,
    parse_json_text,
    purchase_display_df,
    ticket_strategy_display_frame,
    validate_purchase_numbers,
    weighted_model_text,
)
from prl_maintenance import collect_csv_safety_diagnostics, is_light_smoke_mode, is_light_smoke_value


BASE_DIR = LOTO_LAB_DIR
RESULTS_CSV = DATA_DIR / "loto6.csv"
SETS_CSV = DATA_DIR / "loto6_ball_sets.csv"
SCORES_CSV = DATA_DIR / "loto6_next_number_scores.csv"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
PURCHASES_CSV = DATA_DIR / "purchases.csv"
ENSEMBLE_PREDICTIONS_CSV = DATA_DIR / "ensemble_predictions.csv"
TICKET_STRATEGY_HISTORY_CSV = DATA_DIR / "ticket_strategy_history.csv"
POPULARITY_SCORES_CSV = DATA_DIR / "popularity_scores.csv"
CONTINUOUS_WIN_RESEARCH_CSV = DATA_DIR / "continuous_win_research.csv"
BACKTEST_SUMMARY_CSV = DATA_DIR / "backtest_summary.csv"
MODEL_WEIGHT_HISTORY_CSV = AI_IMPROVEMENT_DIR / "model_weight_history.csv"
OFFICIAL_RESULTS_CSV = DATA_DIR / "results.csv"
VERIFICATION_REPORTS_CSV = VERIFICATION_DIR / "verification_reports.csv"
MODEL_SETTINGS_CSV = DATA_DIR / "model_settings.csv"
CONTRIBUTIONS_CSV = VERIFICATION_DIR / "loto6_model_contributions.csv"
RESEARCH_CYCLES_CSV = VERIFICATION_DIR / "loto6_research_cycles.csv"
VIDEO_HYPOTHESES_CSV = VERIFICATION_DIR / "video_hypotheses.csv"
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
PRIZE_COUNT_COLUMNS = ["1等口数", "2等口数", "3等口数", "4等口数", "5等口数"]
PRIZE_AMOUNT_COLUMNS = ["1等賞金", "2等賞金", "3等賞金", "4等賞金", "5等賞金"]
PRIZE_FILL_COLUMNS = [*PRIZE_COUNT_COLUMNS, *PRIZE_AMOUNT_COLUMNS, "キャリーオーバー"]
SET_COLUMNS = ["開催回", "日付", "球セット", "信頼度", "根拠", "動画メタID", "メモ"]
NUMBER_COLUMNS = ["第1数字", "第2数字", "第3数字", "第4数字", "第5数字", "第6数字"]
FILL_ONLY_COLUMNS = ["日付", *NUMBER_COLUMNS, "BONUS数字", *PRIZE_FILL_COLUMNS]

SCORE_COLUMNS = ["順位", "数字", "スコア", "出現回数", "直近30回スコア", "未出現回数", "ボーナス出現回数", "根拠", "更新日時"]
SCORE_NUMERIC_COLUMNS = ["順位", "数字", "スコア", "出現回数", "直近30回スコア", "未出現回数", "ボーナス出現回数"]
SCORE_DISPLAY_COLUMNS = [
    "順位",
    "数字",
    "総合スコア",
    "出現頻度スコア",
    "直近傾向スコア",
    "未出現期間スコア",
    "ボーナス傾向スコア",
    "出現回数",
    "未出現回数",
    "ボーナス出現回数",
    "更新日時",
]
AI_SCORE_DISPLAY_COLUMNS = [
    "AI改善順位",
    "数字",
    "AI改善後スコア",
    "AI改善加点",
    "総合スコア",
    "出現頻度スコア",
    "直近傾向スコア",
    "未出現期間スコア",
    "ボーナス傾向スコア",
    "AI改善理由",
]
PREDICTION_COLUMNS = ["予想ID", "開催回", "予想日", "候補番号", "予想番号", "使用モデル", "予想理由", "保存日時", *BALANCE_PREDICTION_COLUMNS]
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
    "前回数字との重複数",
    "31超え数字の有無",
    "3連続除外チェック",
    "改善メモ",
    "失敗要因",
    "逆算分析",
    "改善案",
    "次回の仮説",
    *BALANCE_VERIFICATION_COLUMNS,
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


def append_research_rows(path, columns, rows):
    if not rows:
        return 0
    existing = read_csv(path, columns)
    incoming = pd.DataFrame(rows).reindex(columns=columns)
    if existing.empty:
        combined = incoming
    else:
        combined = pd.concat([existing.reindex(columns=columns), incoming], ignore_index=True)
    save_csv(combined, path, columns)
    return len(incoming)


def normalize_score_map(score_map):
    values = list(score_map.values())
    if not values:
        return {number: 0.0 for number in range(1, 44)}
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return {number: 0.0 for number in score_map}
    return {number: (value - min_value) / (max_value - min_value) for number, value in score_map.items()}


def normalize_series(values):
    numeric = pd.to_numeric(values, errors="coerce").fillna(0)
    if numeric.empty:
        return numeric
    low = float(numeric.min())
    high = float(numeric.max())
    if high == low:
        return pd.Series([0.0] * len(numeric), index=numeric.index)
    return ((numeric - low) / (high - low) * 100).round(3)


def load_next_score_csv(path=SCORES_CSV, number_max=43):
    warnings = []
    if not path.exists():
        return pd.DataFrame(columns=SCORE_DISPLAY_COLUMNS), [f"{path.name} がありません。手元のCSVで再分析を実行してください。"]
    try:
        score_df = read_csv(path)
    except Exception as exc:
        return pd.DataFrame(columns=SCORE_DISPLAY_COLUMNS), [f"{path.name} を読み込めませんでした: {exc}"]
    if score_df.empty:
        return pd.DataFrame(columns=SCORE_DISPLAY_COLUMNS), [f"{path.name} は空です。手元のCSVで再分析してください。"]
    missing_required = [column for column in ("数字", "スコア") if column not in score_df.columns]
    if missing_required:
        return pd.DataFrame(columns=SCORE_DISPLAY_COLUMNS), [f"{path.name} に必要な列がありません: {', '.join(missing_required)}"]

    score_df = score_df.copy()
    for column in SCORE_NUMERIC_COLUMNS:
        if column not in score_df.columns:
            score_df[column] = 0
            warnings.append(f"{path.name} に {column} 列がないため 0 として扱います。")
        score_df[column] = pd.to_numeric(score_df[column], errors="coerce").fillna(0)

    score_df["数字"] = score_df["数字"].astype(int)
    score_df = score_df[(score_df["数字"] >= 1) & (score_df["数字"] <= number_max)]
    if score_df.empty:
        return pd.DataFrame(columns=SCORE_DISPLAY_COLUMNS), [f"{path.name} に有効な数字がありません。"]

    score_df = score_df.sort_values(["スコア", "出現回数", "数字"], ascending=[False, False, True]).reset_index(drop=True)
    if "順位" not in score_df.columns or not score_df["順位"].any():
        score_df["順位"] = range(1, len(score_df) + 1)
    else:
        score_df["順位"] = pd.to_numeric(score_df["順位"], errors="coerce").fillna(0).astype(int)

    score_df["総合スコア"] = pd.to_numeric(score_df["スコア"], errors="coerce").fillna(0).round(3)
    score_df["出現頻度スコア"] = normalize_series(score_df["出現回数"])
    score_df["直近傾向スコア"] = pd.to_numeric(score_df["直近30回スコア"], errors="coerce").fillna(0).round(3)
    score_df["未出現期間スコア"] = normalize_series(score_df["未出現回数"])
    score_df["ボーナス傾向スコア"] = normalize_series(score_df["ボーナス出現回数"])
    if "更新日時" not in score_df.columns:
        score_df["更新日時"] = ""
        warnings.append(f"{path.name} に 更新日時 列がありません。")
    return score_df, warnings


def score_display_df(score_df):
    if score_df.empty:
        return pd.DataFrame(columns=SCORE_DISPLAY_COLUMNS)
    return score_df.reindex(columns=SCORE_DISPLAY_COLUMNS)


def ai_score_display_df(score_df):
    if score_df.empty:
        return pd.DataFrame(columns=AI_SCORE_DISPLAY_COLUMNS)
    return score_df.reindex(columns=AI_SCORE_DISPLAY_COLUMNS)


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


def safe_prediction_model_id(model_name):
    text = str(model_name)
    if "候補スコア" in text:
        return "next_score_prediction"
    key = model_key_from_name(text) or text
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(key)).strip("_")
    return safe or hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def valid_loto6_prediction_numbers(numbers):
    try:
        parsed = [int(number) for number in numbers]
    except Exception:
        return []
    unique = sorted(set(parsed))
    if len(parsed) != 6 or len(unique) != 6:
        return []
    if any(number < 1 or number > 43 for number in unique):
        return []
    return unique


def best_actionable_model_key(summary_df):
    if summary_df.empty:
        return None
    excluded_models = {
        BACKTEST_MODELS["random_baseline"],
        BACKTEST_MODELS.get(ANTI_POPULAR_EXPECTED_VALUE_KEY),
        BACKTEST_MODELS.get(CHAMINI6_GOD_MODE_KEY),
        CHAMINI6_GOD_MODE_KEY,
    }
    usable_rows = summary_df[~summary_df["モデル"].isin(excluded_models)]
    if usable_rows.empty:
        return None
    return model_key_from_name(str(usable_rows.iloc[0]["モデル"]))


def build_next_number_scores(results):
    if results.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    history = results.copy()
    history["開催回"] = history["開催回"].map(to_int)
    history = history.sort_values("開催回")
    number_rows = [row_main_numbers(row) for _, row in history.iterrows()]
    bonus_rows = [[to_int(row.get("BONUS数字"))] for _, row in history.iterrows()]
    target_round = int(history["開催回"].max()) + 1

    active_setting = read_active_model_setting()
    active_model_key = active_setting["model_key"] if active_setting else "machine_learning"
    active_model_name = active_setting["model_name"] if active_setting else BACKTEST_MODELS.get(active_model_key, active_model_key)
    scores = build_model_scores(number_rows, active_model_key, 43, 6, target_round, bonus_rows)
    recent_scores = build_model_scores(number_rows, "hot_analysis", 43, 6, target_round, bonus_rows)
    normalized_scores = normalize_score_map(scores)
    normalized_recent_scores = normalize_score_map(recent_scores)

    rows = []
    for number in range(1, 44):
        last_seen_index = next(
            (index for index in range(len(number_rows) - 1, -1, -1) if number in number_rows[index]),
            None,
        )
        unseen_count = len(number_rows) if last_seen_index is None else len(number_rows) - last_seen_index - 1
        rows.append(
            {
                "数字": number,
                "スコア": round(normalized_scores.get(number, 0.0) * 100, 3),
                "出現回数": sum(1 for row in number_rows if number in row),
                "直近30回スコア": round(normalized_recent_scores.get(number, 0.0) * 100, 3),
                "未出現回数": unseen_count,
                "ボーナス出現回数": sum(1 for row in bonus_rows if number in row),
                "根拠": f"{active_model_name}を使った第{target_round}回向け候補スコア",
                "更新日時": now_text(),
            }
        )

    score_df = pd.DataFrame(rows)
    score_df = score_df.sort_values(["スコア", "出現回数", "数字"], ascending=[False, False, True]).reset_index(drop=True)
    score_df.insert(0, "順位", range(1, len(score_df) + 1))
    return score_df.reindex(columns=SCORE_COLUMNS)


def run_analysis():
    results = merge_official_results(read_csv(RESULTS_CSV, RESULT_COLUMNS))
    scores = build_next_number_scores(results)
    if scores.empty:
        raise RuntimeError("loto6.csv に抽せん結果がないため、候補スコアを作成できません。")
    save_csv(scores, SCORES_CSV, SCORE_COLUMNS)
    return f"{SCORES_CSV.name} を更新しました。候補数字 {len(scores)}件。"


def run_helper_script(script):
    script_path = BASE_DIR / script
    if not script_path.exists():
        return f"{script}: スキップ（補助スクリプトなし）"
    completed = subprocess.run(
        [sys.executable, str(script_path)],
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


def app_light_smoke_mode():
    if is_light_smoke_mode():
        return True
    try:
        params = st.query_params
        return any(is_light_smoke_value(params.get(key, "")) for key in ("PRL_LIGHT_SMOKE", "light_smoke", "smoke"))
    except Exception:
        return False


def render_prediction_flow_diagnostics(lottery_key="loto6", lottery_label="ロト6"):
    with st.expander(f"{lottery_label} 保存・検証フロー診断（読み取り専用）", expanded=False):
        try:
            diagnostics = collect_csv_safety_diagnostics()
        except Exception as exc:
            st.warning(f"CSV安全診断を読み込めませんでした: {exc}")
            return
        if diagnostics.empty:
            st.info("診断対象のCSV情報がありません。")
            return
        target_col = diagnostics.columns[0]
        filtered = diagnostics[diagnostics[target_col].astype(str).str.startswith(lottery_key)]
        if filtered.empty:
            st.info(f"{lottery_label} の保存・検証診断行はありません。")
            return
        st.caption("この診断は読み取り専用です。表示しても predictions.csv や検証履歴CSVは更新されません。")
        st.dataframe(filtered, width="stretch", hide_index=True)


def render_light_smoke_overview():
    st.success("PRL_LIGHT_SMOKE=1: ロト6軽量スモークモードで起動しています。")
    st.caption(f"重い予測生成、{CHAMINI6_GOD_MODE_LABEL}候補生成、バックテスト、保存ボタン処理は実行しません。")
    results = read_csv(RESULTS_CSV, RESULT_COLUMNS)
    predictions = read_csv(PREDICTIONS_CSV, PREDICTION_COLUMNS)
    reports = read_csv(VERIFICATION_REPORTS_CSV, VERIFICATION_COLUMNS)
    tab_status, tab_flow, tab_csv = st.tabs(["基本構造", "保存・検証フロー", "CSV安全診断"])
    with tab_status:
        latest_round = "-"
        if not results.empty and RESULT_COLUMNS[0] in results.columns:
            values = pd.to_numeric(results[RESULT_COLUMNS[0]], errors="coerce").dropna()
            latest_round = str(int(values.max())) if not values.empty else "-"
        cols = st.columns(4)
        cols[0].metric("最新登録回", latest_round)
        cols[1].metric("予想履歴", len(predictions))
        cols[2].metric("検証履歴", len(reports))
        cols[3].metric("軽量モード", "ON")
        st.info("画面構造と診断だけを確認するモードです。CSVは読み取り専用です。")
    with tab_flow:
        render_prediction_flow_diagnostics("loto6", "ロト6")
    with tab_csv:
        diagnostics = collect_csv_safety_diagnostics()
        st.dataframe(diagnostics, width="stretch", hide_index=True)


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
    table = str.maketrans("０１２３４５６７８９，．－", "0123456789,.-")
    return str(text).translate(table)


def is_blank_value(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = normalize_digits(value).strip()
    return text in {"", "-", "－", "None", "none", "NaN", "nan", "NAN"}


def parse_amount_value(value):
    if is_blank_value(value):
        return None
    text = normalize_digits(value)
    text = re.sub(r"[^\d.-]", "", text)
    if text in {"", "-", "."}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def clean_result_value(column, value):
    if is_blank_value(value):
        return ""
    if column in ["開催回", *NUMBER_COLUMNS, "BONUS数字", *PRIZE_COUNT_COLUMNS, *PRIZE_AMOUNT_COLUMNS, "キャリーオーバー"]:
        parsed = parse_amount_value(value)
        return parsed if parsed is not None else ""
    if column == "日付" and hasattr(value, "strftime"):
        return value.strftime("%Y/%m/%d")
    return str(value).strip()


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


def empty_loto6_result_row():
    return {column: "" for column in RESULT_COLUMNS}


def build_loto6_result_row(round_no, draw_date, main_numbers, bonus_number, extra_values=None):
    sorted_numbers = sorted(main_numbers)
    row = empty_loto6_result_row()
    row.update(
        {
            "開催回": int(round_no),
            "日付": draw_date.strftime("%Y/%m/%d") if hasattr(draw_date, "strftime") else str(draw_date),
            "第1数字": sorted_numbers[0],
            "第2数字": sorted_numbers[1],
            "第3数字": sorted_numbers[2],
            "第4数字": sorted_numbers[3],
            "第5数字": sorted_numbers[4],
            "第6数字": sorted_numbers[5],
            "BONUS数字": int(bonus_number),
        }
    )
    if extra_values:
        for column, value in extra_values.items():
            if column in row:
                row[column] = clean_result_value(column, value)
    return {column: clean_result_value(column, row.get(column, "")) for column in RESULT_COLUMNS}


def extract_prize_values_from_lines(lines):
    values = {}
    for raw_line in lines:
        line = normalize_digits(raw_line)
        carry_match = re.search(r"キャリーオーバー[^0-9-]*([0-9,]+)", line)
        if carry_match:
            values["キャリーオーバー"] = parse_amount_value(carry_match.group(1))

        for grade, count, amount in re.findall(r"([1-5])\s*等[^0-9]*(\d[\d,]*)\s*口[^0-9]*(\d[\d,]*)", line):
            values[f"{grade}等口数"] = parse_amount_value(count)
            values[f"{grade}等賞金"] = parse_amount_value(amount)
    return values


def extract_prize_values_from_table(table, lines):
    values = extract_prize_values_from_lines(lines)
    for _, row in table.iterrows():
        cells = [normalize_digits(value).strip() for value in row.tolist() if not pd.isna(value)]
        if not cells:
            continue
        joined = " ".join(cells)
        grade_match = re.search(r"([1-5])\s*等", joined)
        if not grade_match:
            continue
        grade = grade_match.group(1)
        numeric_values = [parse_amount_value(cell) for cell in cells]
        numeric_values = [value for value in numeric_values if value is not None]
        if len(numeric_values) >= 2:
            values[f"{grade}等口数"] = numeric_values[-2]
            values[f"{grade}等賞金"] = numeric_values[-1]
    return values


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
            prize_values = extract_prize_values_from_table(table, lines)
            return {
                "round_no": int(round_match.group(1)),
                "draw_date": draw_date,
                "main_numbers": sorted(main_numbers),
                "bonus_number": bonus_number,
                "result_row": build_loto6_result_row(
                    int(round_match.group(1)),
                    draw_date,
                    main_numbers,
                    bonus_number,
                    prize_values,
                ),
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


def merge_loto_result_row(results, fetched_row):
    fetched_row = {column: clean_result_value(column, fetched_row.get(column, "")) for column in RESULT_COLUMNS}
    round_no = int(fetched_row["開催回"])
    report = {
        "new_rounds": [],
        "filled_rounds": [],
        "unchanged_rounds": [],
        "missing_fields": [],
        "saved_path": str(RESULTS_CSV),
    }

    if results.empty:
        results = pd.DataFrame([fetched_row])
        report["new_rounds"].append(round_no)
    else:
        results = results.copy()
        for column in RESULT_COLUMNS:
            if column not in results:
                results[column] = ""
        results = results.reindex(columns=RESULT_COLUMNS)
        results = results.astype(object)
        results["開催回"] = results["開催回"].map(to_int)
        same_round = results["開催回"] == round_no
        if same_round.any():
            filled_columns = []
            row_index = results.index[same_round][0]
            for column in FILL_ONLY_COLUMNS:
                fetched_value = fetched_row.get(column, "")
                if is_blank_value(fetched_value):
                    continue
                current_value = results.at[row_index, column]
                if is_blank_value(current_value):
                    results.at[row_index, column] = fetched_value
                    filled_columns.append(column)
            if filled_columns:
                report["filled_rounds"].append(f"第{round_no}回: {', '.join(filled_columns)}")
            else:
                report["unchanged_rounds"].append(round_no)
        else:
            results = pd.concat([results, pd.DataFrame([fetched_row])], ignore_index=True)
            report["new_rounds"].append(round_no)

    results["開催回"] = results["開催回"].map(to_int)
    results = results.sort_values("開催回")
    for column in RESULT_COLUMNS:
        results[column] = results[column].map(lambda value, col=column: clean_result_value(col, value))

    for column in [*PRIZE_COUNT_COLUMNS, *PRIZE_AMOUNT_COLUMNS, "キャリーオーバー"]:
        if is_blank_value(fetched_row.get(column, "")):
            report["missing_fields"].append(column)
    return results, report


def upsert_result_row(round_no, draw_date, main_numbers, bonus_number, extra_values=None):
    results = read_csv(RESULTS_CSV, RESULT_COLUMNS)
    sorted_numbers = sorted(main_numbers)
    result_row = build_loto6_result_row(round_no, draw_date, sorted_numbers, bonus_number, extra_values)
    results, report = merge_loto_result_row(results, result_row)
    save_csv(results, RESULTS_CSV, RESULT_COLUMNS)
    upsert_official_result(round_no, draw_date, sorted_numbers, bonus_number, "web")
    return report


def format_update_report(report):
    lines = [
        f"新規追加した開催回: {', '.join(map(str, report['new_rounds'])) if report['new_rounds'] else 'なし'}",
        f"補完した開催回: {' / '.join(map(str, report['filled_rounds'])) if report['filled_rounds'] else 'なし'}",
        f"変更なしだった開催回: {', '.join(map(str, report['unchanged_rounds'])) if report['unchanged_rounds'] else 'なし'}",
        f"取得できなかった項目: {', '.join(report['missing_fields']) if report['missing_fields'] else 'なし'}",
        f"保存先: {report['saved_path']}",
    ]
    return "\n".join(lines)


def format_manual_prize_report(report):
    lines = [
        f"補完した開催回: 第{report['round_no']}回",
        f"補完した項目: {', '.join(report['filled_columns']) if report['filled_columns'] else 'なし'}",
        f"上書きした項目: {', '.join(report['overwritten_columns']) if report['overwritten_columns'] else 'なし'}",
        f"変更なしだった項目: {', '.join(report['unchanged_columns']) if report['unchanged_columns'] else 'なし'}",
        f"保存先: {report['saved_path']}",
    ]
    return "\n".join(lines)


def fill_loto6_prize_values_manually(round_no, input_values, overwrite_existing=False):
    results = read_csv(RESULTS_CSV, RESULT_COLUMNS)
    if results.empty or "開催回" not in results:
        raise ValueError("loto6.csv に対象データがありません。先に当選結果を登録してください。")

    results = results.copy()
    for column in RESULT_COLUMNS:
        if column not in results:
            results[column] = ""
    results = results.reindex(columns=RESULT_COLUMNS).astype(object)
    results["開催回"] = results["開催回"].map(to_int)

    round_no = int(round_no)
    same_round = results["開催回"] == round_no
    if not same_round.any():
        raise ValueError(f"第{round_no}回は loto6.csv に存在しません。新規追加はしません。")

    row_index = results.index[same_round][0]
    report = {
        "round_no": round_no,
        "filled_columns": [],
        "overwritten_columns": [],
        "unchanged_columns": [],
        "saved_path": str(RESULTS_CSV),
    }

    for column in PRIZE_FILL_COLUMNS:
        input_value = clean_result_value(column, input_values.get(column, ""))
        if is_blank_value(input_value):
            report["unchanged_columns"].append(column)
            continue

        current_value = results.at[row_index, column]
        if is_blank_value(current_value):
            results.at[row_index, column] = input_value
            report["filled_columns"].append(column)
        elif overwrite_existing and clean_result_value(column, current_value) != input_value:
            results.at[row_index, column] = input_value
            report["overwritten_columns"].append(column)
        else:
            report["unchanged_columns"].append(column)

    for column in RESULT_COLUMNS:
        results[column] = results[column].map(lambda value, col=column: clean_result_value(col, value))
    results = results.sort_values("開催回")
    save_csv(results, RESULTS_CSV, RESULT_COLUMNS)
    return report


def update_latest_result_from_web(run_analysis_after=True):
    latest = extract_latest_loto6_result(fetch_web_text(MIZUHO_LOTO6_URL))
    report = upsert_result_row(
        latest["round_no"],
        latest["draw_date"],
        latest["main_numbers"],
        latest["bonus_number"],
        latest.get("result_row", {}),
    )

    helper_messages = []
    verified_count = 0
    if run_analysis_after:
        for script in ("import_dream_backnumber_videos.py", "import_charlie_recent_loto6_sets.py"):
            try:
                run_helper_script(script)
                helper_messages.append(f"{script}: OK")
            except Exception as exc:
                helper_messages.append(f"{script}: {exc}")

        run_analysis()
        verified_count = verify_predictions_for_round(latest["round_no"])

    numbers = " - ".join(f"{number:02d}" for number in latest["main_numbers"])
    return (
        f"第{latest['round_no']}回 {latest['draw_date'].strftime('%Y/%m/%d')} "
        f"{numbers} / BONUS {latest['bonus_number']:02d} を確認しました。"
        + (f" 検証レポート{verified_count}件を更新しました。" if run_analysis_after else "")
        + "\n"
        + format_update_report(report)
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


def history_number_rows(results):
    if results is None or results.empty:
        return []
    history = results.copy()
    history["開催回"] = history["開催回"].map(to_int)
    history = history.sort_values("開催回")
    rows = []
    for _, row in history.iterrows():
        numbers = sorted(to_int(row[column]) for column in NUMBER_COLUMNS)
        if len(numbers) == 6:
            rows.append(numbers)
    return rows


def history_bonus_rows(results):
    if results is None or results.empty or "BONUS数字" not in results:
        return []
    history = results.copy()
    history["開催回"] = history["開催回"].map(to_int)
    history = history.sort_values("開催回")
    rows = []
    for _, row in history.iterrows():
        bonus = to_int(row.get("BONUS数字"))
        if 1 <= bonus <= 43:
            rows.append([bonus])
    return rows


def render_set_ball_analysis_panel(results, sets):
    analysis = build_set_ball_analysis(results, NUMBER_COLUMNS, 43, 6, set_ball_frame=sets, round_column="開催回")
    with st.expander(f"セット球分析（{CHAMINI6_GOD_MODE_LABEL}補助）", expanded=False):
        if analysis.get("available"):
            st.success(analysis.get("message", "セット球分析を反映できます。"))
            if not analysis.get("summary_frame", pd.DataFrame()).empty:
                st.dataframe(analysis["summary_frame"], width="stretch", hide_index=True)
            if not analysis.get("number_frame", pd.DataFrame()).empty:
                st.dataframe(analysis["number_frame"].head(12), width="stretch", hide_index=True)
        else:
            st.info(analysis.get("message", "セット球データなし。"))
    return analysis


def render_chamini6_prediction(results, target_round, ai_weight_summary, set_ball_analysis):
    reports = read_csv(VERIFICATION_REPORTS_CSV, VERIFICATION_COLUMNS)
    _, model_history = load_winning_condition_history(AI_IMPROVEMENT_DIR, "loto6")
    picks = generate_chamini6_god_mode_picks(
        history_number_rows(results),
        history_bonus_rows(results),
        number_max=43,
        draw_size=6,
        target_round=target_round,
        reports=reports,
        model_history=model_history,
        lottery_type="loto6",
        ai_weight_summary=ai_weight_summary,
        set_ball_analysis=set_ball_analysis,
        pick_count=1,
    )
    with st.expander(f"{CHAMINI6_GOD_MODE_LABEL}（統合候補）", expanded=False):
        if not picks:
            st.info(f"{CHAMINI6_GOD_MODE_LABEL}を表示するには、抽せん履歴が必要です。")
            return
        pick = picks[0]
        st.markdown(f"**予想番号: {numbers_to_text(pick.get('numbers', []))}**")
        st.write(f"使用モデル: {pick.get('display_model_name', CHAMINI6_GOD_MODE_LABEL)}")
        st.write(f"重視した要素: {pick.get('emphasized_factors', '-')}")
        st.write(f"選定理由: {pick.get('selection_reason', pick.get('reason', '-'))}")
        if pick.get("balance_score") is not None:
            st.write(f"バランス仮説エンジン: {pick.get('balance_grade')} / score {pick.get('balance_score')}")
            if pick.get("balance_reasons"):
                st.caption("評価理由: " + " / ".join(map(str, pick.get("balance_reasons", []))))
            if pick.get("balance_warnings"):
                st.warning("警告ポイント: " + " / ".join(map(str, pick.get("balance_warnings", []))))
        st.caption(f"Pattern A/B/Cとは別の統合候補です。Pattern Cには低人気期待値要素を、{CHAMINI6_GOD_MODE_LABEL}には複数モデル重みとバランス仮説を反映します。")
        diagnostics = pick.get("anti_popular_diagnostics", {})
        if diagnostics:
            st.caption(
                f"低人気補助: 前回重複{diagnostics.get('previous_overlap_count')}個 / "
                f"31超え数字{'あり' if diagnostics.get('has_over_31') else 'なし'} / "
                f"3連続チェック{'OK' if diagnostics.get('three_consecutive_excluded') else '要注意'}"
            )
        detail = pick.get("chamini6_detail", pd.DataFrame())
        if not detail.empty:
            st.dataframe(detail, width="stretch", hide_index=True)
        if st.button(f"{CHAMINI6_GOD_MODE_LABEL}予測を保存", key="loto6_chamini6_save"):
            result = save_prediction_picks_detailed(picks, target_round, CHAMINI6_GOD_MODE_LABEL)
            render_loto6_prediction_save_result(
                result,
                f"{CHAMINI6_GOD_MODE_LABEL}予測を {{saved}} 件保存しました。",
                f"同じ{CHAMINI6_GOD_MODE_LABEL}予測は保存済みです。",
            )


def build_anti_popular_pattern_pick(results, target_round, anchor_numbers=None):
    return (
        generate_anti_popular_expected_value_picks(
            history_number_rows(results),
            number_max=43,
            draw_size=6,
            target_round=target_round,
            pick_count=1,
            anchor_numbers=anchor_numbers,
            max_anchor_overlap=2,
        )
        or [None]
    )[0]


def render_pattern_pick(pick, title):
    numbers = " - ".join(f"{number:02d}" for number in pick.get("numbers", []))
    pattern = pick.get("pattern_label", title)
    role = pick.get("role_label", "")
    st.markdown(f"**{pattern} {role}: {numbers}**")
    st.write(f"使用モデル: {pick.get('model_name', '既存分析モデル')}")
    st.write(f"重視した要素: {pick.get('emphasized_factors', '-')}")
    st.write(f"選定理由: {pick.get('selection_reason', pick.get('reason', '-'))}")
    st.write(f"他Patternとの重複数: {pick.get('overlap_summary', '-')}")
    diagnostics = pick.get("anti_popular_diagnostics")
    if diagnostics:
        st.caption(
            f"補助モデル確認: 前回重複{diagnostics.get('previous_overlap_count')}個 / "
            f"31超え数字{'あり' if diagnostics.get('has_over_31') else 'なし'} / "
            f"3連続チェック{'OK' if diagnostics.get('three_consecutive_excluded') else '除外対象'}"
        )
        st.caption(pick.get("expected_value_note", ""))
    st.write(f"理由: {pick.get('reason', '-')}")


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
    if active_model_key == ANTI_POPULAR_EXPECTED_VALUE_KEY:
        active_model_name = "標準スコアバランス"
    elif active_model_key in BACKTEST_MODELS and active_model_key != "random_baseline":
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

    anti_pick = build_anti_popular_pattern_pick(
        results,
        target_round or (int(results["開催回"].map(to_int).max()) + 1),
        anchor_numbers=picks[0]["numbers"] if picks else None,
    )
    return apply_prediction_pattern_roles(
        picks,
        draw_size=6,
        number_max=43,
        score_map=score_map,
        base_model_name=active_model_name or "標準スコアバランス",
        anti_popular_pick=anti_pick,
    )


def build_score_number_reasons(numbers, score_df, low_threshold=21):
    rows = []
    row_map = {int(row["数字"]): row for _, row in score_df.iterrows()}
    odd_count = sum(number % 2 for number in numbers)
    low_count = sum(number <= low_threshold for number in numbers)
    for number in numbers:
        row = row_map.get(int(number), {})
        rank = int(row.get("順位", 0) or 0)
        reasons = []
        if rank and rank <= 10:
            reasons.append("総合スコア上位")
        if float(row.get("出現頻度スコア", 0) or 0) >= 70:
            reasons.append("出現頻度スコアが高い")
        if float(row.get("直近傾向スコア", 0) or 0) >= 70:
            reasons.append("直近傾向スコアが高い")
        if float(row.get("未出現期間スコア", 0) or 0) >= 70:
            reasons.append("未出現期間スコアが高い")
        if float(row.get("ボーナス傾向スコア", 0) or 0) >= 70:
            reasons.append("ボーナス傾向で評価")
        if rank > 6 and odd_count in (2, 3, 4):
            reasons.append("奇数偶数バランス調整のため採用")
        if rank > 6 and low_count in (2, 3, 4):
            reasons.append("高低バランス調整のため採用")
        if not reasons:
            reasons.append("候補スコアと全体バランスで採用")
        rows.append(
            {
                "数字": f"{int(number):02d}",
                "順位": rank if rank else "-",
                "総合スコア": round(float(row.get("総合スコア", 0) or 0), 3),
                "主な理由": " / ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def build_ai_weighted_number_reasons(numbers, score_df, low_threshold=21):
    rows = []
    row_map = {int(row["数字"]): row for _, row in score_df.iterrows()}
    odd_count = sum(number % 2 for number in numbers)
    low_count = sum(number <= low_threshold for number in numbers)
    for number in numbers:
        row = row_map.get(int(number), {})
        rank = int(row.get("順位", 0) or 0)
        ai_rank = int(row.get("AI改善順位", 0) or 0)
        reasons = []
        if ai_rank and ai_rank <= 10:
            reasons.append("AI改善反映後の候補上位")
        if rank and rank <= 10:
            reasons.append("候補スコア上位")
        if float(row.get("直近傾向スコア", 0) or 0) >= 70:
            reasons.append("直近傾向スコアが高い")
        if float(row.get("未出現期間スコア", 0) or 0) >= 70:
            reasons.append("未出現期間スコアが高い")
        positive_models = str(row.get("AI改善関連モデル", "") or "").strip()
        negative_models = str(row.get("AI改善抑制モデル", "") or "").strip()
        ai_adjustment = float(row.get("AI改善加点", 0) or 0)
        if positive_models:
            reasons.append(f"AI改善履歴で{positive_models}の重みが上がっているため採用")
        if negative_models and ai_adjustment <= -0.1:
            reasons.append(f"AI改善履歴で{negative_models}は抑制対象だが、全体バランスで残す")
        if ai_adjustment > 0.1:
            reasons.append("直近の当選条件分析で有効モデルに関連")
        if ai_rank > 6 and odd_count in (2, 3, 4):
            reasons.append("奇数偶数バランス調整のため採用")
        if ai_rank > 6 and low_count in (2, 3, 4):
            reasons.append("高低バランス調整のため採用")
        if not reasons:
            reasons.append("候補スコアとAI改善重みの総合評価で採用")
        rows.append(
            {
                "数字": f"{int(number):02d}",
                "候補順位": rank if rank else "-",
                "AI改善順位": ai_rank if ai_rank else "-",
                "AI改善後スコア": round(float(row.get("AI改善後スコア", 0) or 0), 3),
                "AI改善加点": round(ai_adjustment, 3),
                "主な理由": " / ".join(dict.fromkeys(reasons)),
            }
        )
    return pd.DataFrame(rows)


def generate_next_score_prediction_picks(score_df, results, draw_size=6, pick_count=3):
    if score_df.empty or results.empty:
        return []
    score_df = score_df.copy().sort_values(["総合スコア", "出現回数", "数字"], ascending=[False, False, True]).reset_index(drop=True)
    pool = score_df.head(20)
    row_map = {int(row["数字"]): row for _, row in score_df.iterrows()}
    candidate_numbers = [int(number) for number in pool["数字"].tolist()]
    top_10 = set(int(number) for number in score_df.head(10)["数字"].tolist())

    sums = results[NUMBER_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    target_sum = float(sums.mean())
    sum_std = float(sums.std()) if len(sums) > 1 else 25.0
    if pd.isna(sum_std) or sum_std <= 0:
        sum_std = 25.0

    candidates = []
    for combo in combinations(candidate_numbers, draw_size):
        numbers = tuple(sorted(combo))
        odd = sum(number % 2 for number in numbers)
        low = sum(number <= 21 for number in numbers)
        consecutive = count_consecutive_pairs(numbers)
        top_count = len(set(numbers) & top_10)
        if odd not in (2, 3, 4) or low not in (2, 3, 4) or consecutive > 1:
            continue
        if top_count >= draw_size:
            continue
        total = sum(numbers)
        score_total = sum(float(row_map[number].get("総合スコア", 0) or 0) for number in numbers)
        feature_total = sum(
            float(row_map[number].get("出現頻度スコア", 0) or 0) * 0.08
            + float(row_map[number].get("直近傾向スコア", 0) or 0) * 0.08
            + float(row_map[number].get("未出現期間スコア", 0) or 0) * 0.05
            + float(row_map[number].get("ボーナス傾向スコア", 0) or 0) * 0.04
            for number in numbers
        )
        sum_penalty = abs(total - target_sum) / max(sum_std, 1) * 10
        balance_penalty = abs(odd - 3) * 4 + abs(low - 3) * 4 + consecutive * 6
        diversity_bonus = min(draw_size - top_count, 2) * 5
        candidates.append((score_total + feature_total + diversity_bonus - sum_penalty - balance_penalty, numbers))

    if not candidates:
        fallback = tuple(sorted(candidate_numbers[:draw_size]))
        candidates = [(0, fallback)]

    candidates.sort(reverse=True, key=lambda item: item[0])
    picks = []
    used_numbers = set()
    for _, numbers in candidates:
        if picks and len(set(numbers) & used_numbers) > 3:
            continue
        reason_table = build_score_number_reasons(numbers, score_df)
        summary = balance_summary(numbers)
        picks.append(
            {
                "numbers": numbers,
                "reason": (
                    "候補スコアCSVの総合スコアと特徴スコアを使い、"
                    f"奇数{summary['odd']}・偶数{summary['even']}、低数字{summary['low']}・高数字{summary['high']}、"
                    f"合計{summary['sum']}を過去平均{target_sum:.1f}付近に調整。"
                ),
                "number_reasons": reason_table,
            }
        )
        used_numbers.update(numbers)
        if len(picks) == pick_count:
            break
    score_map = {int(row["数字"]): float(row.get("総合スコア", 0) or 0) for _, row in score_df.iterrows()}
    anti_pick = build_anti_popular_pattern_pick(
        results,
        int(results["開催回"].map(to_int).max()) + 1,
        anchor_numbers=picks[0]["numbers"] if picks else None,
    )
    return apply_prediction_pattern_roles(
        picks,
        draw_size=draw_size,
        number_max=43,
        score_map=score_map,
        base_model_name="候補スコア活用予測",
        anti_popular_pick=anti_pick,
    )


def generate_ai_weighted_prediction_picks(score_df, results, weight_summary, draw_size=6, pick_count=3):
    if score_df.empty or results.empty:
        return []
    weighted_df = apply_ai_improvement_weights(score_df, weight_summary)
    if weighted_df.empty or not (weight_summary or {}).get("available"):
        picks = generate_next_score_prediction_picks(score_df, results, draw_size, pick_count)
        for pick in picks:
            pick["reason"] = "AI改善履歴が不足しているため、通常の候補スコア活用予測にフォールバックしています。" + pick["reason"]
        return picks

    weighted_df = weighted_df.copy().sort_values(["AI改善後スコア", "総合スコア", "数字"], ascending=[False, False, True]).reset_index(drop=True)
    pool = weighted_df.head(22)
    row_map = {int(row["数字"]): row for _, row in weighted_df.iterrows()}
    candidate_numbers = [int(number) for number in pool["数字"].tolist()]
    top_10 = set(int(number) for number in weighted_df.head(10)["数字"].tolist())

    sums = results[NUMBER_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    target_sum = float(sums.mean())
    sum_std = float(sums.std()) if len(sums) > 1 else 25.0
    if pd.isna(sum_std) or sum_std <= 0:
        sum_std = 25.0

    candidates = []
    for combo in combinations(candidate_numbers, draw_size):
        numbers = tuple(sorted(combo))
        odd = sum(number % 2 for number in numbers)
        low = sum(number <= 21 for number in numbers)
        consecutive = count_consecutive_pairs(numbers)
        top_count = len(set(numbers) & top_10)
        if odd not in (2, 3, 4) or low not in (2, 3, 4) or consecutive > 1:
            continue
        if top_count >= draw_size:
            continue
        total = sum(numbers)
        score_total = sum(float(row_map[number].get("AI改善後スコア", 0) or 0) for number in numbers)
        feature_total = sum(
            float(row_map[number].get("出現頻度スコア", 0) or 0) * 0.05
            + float(row_map[number].get("直近傾向スコア", 0) or 0) * 0.06
            + float(row_map[number].get("未出現期間スコア", 0) or 0) * 0.05
            + max(float(row_map[number].get("AI改善加点", 0) or 0), 0) * 0.8
            for number in numbers
        )
        sum_penalty = abs(total - target_sum) / max(sum_std, 1) * 10
        balance_penalty = abs(odd - 3) * 4 + abs(low - 3) * 4 + consecutive * 6
        diversity_bonus = min(draw_size - top_count, 2) * 5
        candidates.append((score_total + feature_total + diversity_bonus - sum_penalty - balance_penalty, numbers))

    if not candidates:
        fallback = tuple(sorted(candidate_numbers[:draw_size]))
        candidates = [(0, fallback)]

    candidates.sort(reverse=True, key=lambda item: item[0])
    picks = []
    used_numbers = set()
    for _, numbers in candidates:
        if picks and len(set(numbers) & used_numbers) > 3:
            continue
        reason_table = build_ai_weighted_number_reasons(numbers, weighted_df)
        summary = balance_summary(numbers)
        picks.append(
            {
                "numbers": numbers,
                "reason": (
                    "候補スコアCSVにAI改善履歴のモデル重みを反映し、"
                    f"重み上げ: {weighted_model_text(weight_summary.get('model_weights', {}), True)}、"
                    f"重み下げ: {weighted_model_text(weight_summary.get('model_weights', {}), False)}を加味。"
                    f"奇数{summary['odd']}・偶数{summary['even']}、低数字{summary['low']}・高数字{summary['high']}、"
                    f"合計{summary['sum']}を過去平均{target_sum:.1f}付近に調整。"
                ),
                "number_reasons": reason_table,
            }
        )
        used_numbers.update(numbers)
        if len(picks) == pick_count:
            break
    score_map = {int(row["数字"]): float(row.get("AI改善後スコア", row.get("総合スコア", 0)) or 0) for _, row in weighted_df.iterrows()}
    anti_pick = build_anti_popular_pattern_pick(
        results,
        int(results["開催回"].map(to_int).max()) + 1,
        anchor_numbers=picks[0]["numbers"] if picks else None,
    )
    return apply_prediction_pattern_roles(
        picks,
        draw_size=draw_size,
        number_max=43,
        score_map=score_map,
        base_model_name="AI改善反映予測",
        anti_popular_pick=anti_pick,
    )


def row_main_numbers(row):
    return sorted(to_int(row[column]) for column in NUMBER_COLUMNS)


def high_prize_history_context(result_history):
    if result_history is None or result_history.empty:
        return [], [], 1
    history = result_history.copy()
    round_column = history.columns[0]
    bonus_column = RESULT_COLUMNS[8]
    history[round_column] = history[round_column].map(to_int)
    history = history[history[round_column] > 0].sort_values(round_column)
    number_rows = [row_main_numbers(row) for _, row in history.iterrows()]
    bonus_rows = [[to_int(row.get(bonus_column))] for _, row in history.iterrows()]
    target_round = int(history[round_column].max()) + 1 if not history.empty else 1
    return number_rows, bonus_rows, target_round


def save_high_prize_research_rows(tickets, weight_df, target_round):
    created_at = now_text()
    prediction_date = today_text()
    saved = {}
    saved["ensemble_predictions.csv"] = append_research_rows(
        ENSEMBLE_PREDICTIONS_CSV,
        ENSEMBLE_PREDICTION_COLUMNS,
        build_ensemble_prediction_rows(tickets, "loto6", target_round, prediction_date, created_at),
    )
    saved["ticket_strategy_history.csv"] = append_research_rows(
        TICKET_STRATEGY_HISTORY_CSV,
        TICKET_STRATEGY_COLUMNS,
        build_ticket_strategy_rows(tickets, "loto6", target_round, prediction_date, created_at),
    )
    saved["popularity_scores.csv"] = append_research_rows(
        POPULARITY_SCORES_CSV,
        POPULARITY_SCORE_COLUMNS,
        build_popularity_score_rows(tickets, "loto6", target_round, prediction_date, created_at),
    )
    saved["continuous_win_research.csv"] = append_research_rows(
        CONTINUOUS_WIN_RESEARCH_CSV,
        CONTINUOUS_WIN_RESEARCH_COLUMNS,
        build_continuous_win_research_rows(tickets, "loto6", target_round, prediction_date=prediction_date, created_at=created_at),
    )
    if weight_df is not None and not weight_df.empty:
        weight_rows = weight_df.copy()
        weight_rows["lottery_type"] = "loto6"
        weight_rows["created_at"] = created_at
        saved["ai_improvement/model_weight_history.csv"] = append_research_rows(
            MODEL_WEIGHT_HISTORY_CSV,
            MODEL_WEIGHT_HISTORY_COLUMNS,
            weight_rows.to_dict("records"),
        )
    return saved


def render_high_prize_continuous_mode(result_history, reports):
    st.subheader("高額当選・連続当選強化モード")
    st.caption("当選保証ではなく、予想、結果、検証、改善を継続するための研究モードです。")
    number_rows, bonus_rows, target_round = high_prize_history_context(result_history)
    if not number_rows:
        st.warning("抽選履歴がないため、強化モードを作成できません。ロト6の結果CSVを確認してください。")
        return

    _, model_history = load_winning_condition_history(AI_IMPROVEMENT_DIR, "loto6")
    tickets, weight_df = build_high_prize_ticket_strategy(
        number_rows,
        bonus_rows,
        number_max=43,
        draw_size=6,
        target_round=target_round,
        reports=reports,
        model_history=model_history,
        lottery_type="loto6",
    )
    ticket_df = ticket_strategy_display_frame(tickets)
    mode_tabs = st.tabs(["高額当選・連続当選モード", "アンサンブル予測", "モデル重みランキング", "買い目別スコア", "バックテスト", "AI改善レポート"])

    with mode_tabs[0]:
        cols = st.columns(4)
        cols[0].metric("対象回", target_round)
        cols[1].metric("研究買い目", len(tickets))
        cols[2].metric("対象モデル", len(ARL_MODEL_LABELS))
        cols[3].metric("履歴回数", len(number_rows))
        st.write("3口を本命・安定型、直近トレンド型、高額当選狙い・低人気型に分けて検証します。")
        st.dataframe(build_research_flow_table(), width="stretch", hide_index=True)
        if st.button("強化モードの研究履歴を保存", key="loto6_save_high_prize_mode"):
            try:
                saved = save_high_prize_research_rows(tickets, weight_df, target_round)
                st.success(" / ".join(f"{name}: {count}件" for name, count in saved.items()))
            except Exception as exc:
                st.error(f"研究履歴の保存に失敗しました: {exc}")

    with mode_tabs[1]:
        if ticket_df.empty:
            st.info("アンサンブル予測を作成できませんでした。抽選履歴と検証レポートを確認してください。")
        else:
            st.dataframe(ticket_df, width="stretch", hide_index=True)
        st.markdown("**モデル重み**")
        st.dataframe(weight_df, width="stretch", hide_index=True)

    with mode_tabs[2]:
        st.write("平均一致数、直近成績、長期成績、安定性、期待値、ボーナス一致率を使って重みを作成します。")
        st.dataframe(weight_df.sort_values("applied_weight", ascending=False), width="stretch", hide_index=True)

    with mode_tabs[3]:
        if ticket_df.empty:
            st.info("買い目別スコアはまだありません。")
        else:
            st.dataframe(ticket_df, width="stretch", hide_index=True)
            popularity_rows = build_popularity_score_rows(tickets, "loto6", target_round, today_text(), now_text())
            st.markdown("**低人気スコア内訳**")
            st.dataframe(pd.DataFrame(popularity_rows), width="stretch", hide_index=True)

    with mode_tabs[4]:
        periods = st.multiselect("検証期間", [30, 50, 100, "all"], default=[30], key="loto6_high_prize_backtest_periods")
        if st.button("強化バックテストを実行して保存", key="loto6_run_high_prize_backtest"):
            with st.spinner("強化モードのバックテストを実行しています。"):
                summary_df = build_high_prize_backtest_summary(
                    number_rows,
                    bonus_rows,
                    number_max=43,
                    draw_size=6,
                    lottery_type="loto6",
                    periods=periods,
                    reports=reports,
                    model_history=model_history,
                )
            st.session_state["loto6_high_prize_backtest"] = summary_df
            if not summary_df.empty:
                append_research_rows(BACKTEST_SUMMARY_CSV, BACKTEST_SUMMARY_COLUMNS, summary_df.to_dict("records"))
                st.success(f"backtest_summary.csv に {len(summary_df)} 件保存しました。")
        summary_df = st.session_state.get("loto6_high_prize_backtest", pd.DataFrame())
        if summary_df.empty:
            st.info("検証期間を選んでバックテストを実行してください。")
        else:
            st.dataframe(summary_df, width="stretch", hide_index=True)

    with mode_tabs[5]:
        ai_report = build_enhanced_ai_improvement_report(reports, weight_df, draw_size=6)
        st.dataframe(ai_report, width="stretch", hide_index=True)


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


def run_pre_prediction_research(results, persist_setting=False):
    if results.empty or len(results) <= 30:
        return pd.DataFrame(), pd.DataFrame(), None
    lookback_rounds = min(5, max(3, len(results) - 30))
    summary_df, detail_df = run_backtest(results, lookback_rounds=lookback_rounds)
    best_model_key = best_actionable_model_key(summary_df)
    if best_model_key and persist_setting:
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


def prediction_reason_with_pattern(pick):
    pattern_text = " ".join(
        str(value)
        for value in (pick.get("pattern_label"), pick.get("role_label"))
        if value
    ).strip()
    reason = str(pick.get("reason", pick.get("selection_reason", "")))
    return f"[{pattern_text}] {reason}" if pattern_text else reason


def save_prediction_picks_detailed(picks, target_round, model_name="score_balance_v1"):
    id_col, draw_col, date_col, candidate_col, numbers_col, model_col, reason_col, saved_col = PREDICTION_COLUMNS[:8]
    predictions = read_csv(PREDICTIONS_CSV, PREDICTION_COLUMNS)
    existing_keys = set()
    existing_ids = set()
    if not predictions.empty:
        for _, row in predictions.iterrows():
            existing_keys.add(
                (
                    to_int(row.get(draw_col)),
                    to_int(row.get(candidate_col)),
                    str(row.get(numbers_col)),
                    str(row.get(model_col)),
                )
            )
            existing_ids.add(str(row.get(id_col, "")).strip())

    rows = []
    skipped = 0
    errors = []
    prediction_date = today_text()
    for index, pick in enumerate(picks, start=1):
        numbers = valid_loto6_prediction_numbers(pick.get("numbers", []))
        if not numbers:
            errors.append(f"candidate {index}: invalid loto6 numbers")
            continue
        number_text = numbers_to_text(numbers)
        row_model_name = pick.get("model_name") or model_name
        model_id = safe_prediction_model_id(pick.get("model_key") or row_model_name)
        prediction_id = f"{int(target_round)}-{prediction_date.replace('/', '')}-{model_id}-{index}"
        key = (int(target_round), index, number_text, row_model_name)
        if key in existing_keys or prediction_id in existing_ids:
            skipped += 1
            continue
        row = {
            id_col: prediction_id,
            draw_col: int(target_round),
            date_col: prediction_date,
            candidate_col: index,
            numbers_col: number_text,
            model_col: row_model_name,
            reason_col: prediction_reason_with_pattern(pick),
            saved_col: now_text(),
        }
        row.update(balance_prediction_fields_from_pick(pick))
        rows.append(row)
        existing_keys.add(key)
        existing_ids.add(prediction_id)

    if rows:
        predictions = pd.concat([predictions, pd.DataFrame(rows)], ignore_index=True)
        predictions[draw_col] = predictions[draw_col].map(to_int)
        predictions[candidate_col] = predictions[candidate_col].map(to_int)
        predictions = predictions.sort_values([draw_col, candidate_col, saved_col])
        save_csv(predictions, PREDICTIONS_CSV, PREDICTION_COLUMNS)
    return {
        "saved": len(rows),
        "skipped": skipped,
        "errors": errors,
        "prediction_ids": [row[id_col] for row in rows],
        "target_round": int(target_round),
    }


def render_loto6_prediction_save_result(result, saved_message, duplicate_message):
    if result["saved"]:
        st.success(saved_message.format(saved=result["saved"]))
    else:
        st.info(duplicate_message)
    if result.get("prediction_ids"):
        st.caption(f"保存した予想ID: {', '.join(result['prediction_ids'])}")
    if result.get("skipped"):
        st.warning(f"重複のため {result['skipped']} 件をスキップしました。")
    for error in result.get("errors", []):
        st.warning(error)


def save_prediction_picks(picks, target_round, model_name="score_balance_v1"):
    return save_prediction_picks_detailed(picks, target_round, model_name)["saved"]
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
        row_model_name = pick.get("model_name") or model_name
        model_id = safe_prediction_model_id(pick.get("model_key") or row_model_name)
        key = (int(target_round), index, number_text, row_model_name)
        if key in existing_keys:
            continue
        rows.append(
            {
                "予想ID": f"{int(target_round)}-{prediction_date.replace('/', '')}-{model_id}-{index}",
                "開催回": int(target_round),
                "予想日": prediction_date,
                "候補番号": index,
                "予想番号": number_text,
                "使用モデル": row_model_name,
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

    report = {
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
    return attach_balance_verification_fields(report, prediction_row, draw_size=6)


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

    reports = read_csv(VERIFICATION_REPORTS_CSV, VERIFICATION_COLUMNS)
    if not reports.empty and "予想ID" in reports:
        existing_ids = set(reports["予想ID"].astype(str).str.strip())
        predictions = predictions[~predictions["予想ID"].astype(str).str.strip().isin(existing_ids)]
    if predictions.empty:
        return 0

    report_rows = []
    contribution_rows = []
    winning_condition_rows = []
    model_improvement_rows = []
    for _, prediction_row in predictions.iterrows():
        result_matches = official_results[official_results["開催回"] == to_int(prediction_row["開催回"])]
        if result_matches.empty:
            continue
        official_row = result_matches.tail(1).iloc[0]
        draw_no = to_int(prediction_row["開催回"])
        predicted = parse_number_text(prediction_row["予想番号"])
        actual = parse_number_text(official_row["本数字"])
        bonus = parse_number_text(official_row["ボーナス数字"])
        number_rows, bonus_rows = historical_context_before_round(draw_no)
        support_map = build_model_support_map(
            predicted,
            number_rows,
            number_max=43,
            draw_size=6,
            target_round=draw_no,
            bonus_rows=bonus_rows,
            selected_model=prediction_row.get("使用モデル", ""),
        )
        report_row = build_verification_report_row(prediction_row, official_row, support_map)
        report_row.update(anti_popular_verification_fields(predicted, number_rows[-1] if number_rows else [], 43, 6))
        report_rows.append(report_row)
        winning_row, model_rows = build_winning_condition_report(
            lottery_type="loto6",
            draw_no=draw_no,
            prediction_id=prediction_row["予想ID"],
            prediction_date=prediction_row.get("予想日", ""),
            predicted=predicted,
            actual=actual,
            bonus_numbers=bonus,
            number_rows=number_rows,
            bonus_rows=bonus_rows,
            number_max=43,
            draw_size=6,
            selected_model=prediction_row.get("使用モデル", ""),
            failure_reason=report_row["失敗要因"],
            created_at=now_text(),
        )
        winning_condition_rows.append(winning_row)
        model_improvement_rows.extend(model_rows)
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
    if winning_condition_rows:
        append_winning_condition_history(AI_IMPROVEMENT_DIR, winning_condition_rows, model_improvement_rows)
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


def render_manual_prize_fill_form():
    results = read_csv(RESULTS_CSV, RESULT_COLUMNS)
    existing_rounds = []
    if not results.empty and "開催回" in results:
        existing_rounds = sorted({to_int(value) for value in results["開催回"] if to_int(value) > 0})
    default_round = existing_rounds[-1] if existing_rounds else 1

    with st.expander("口数・賞金を手動補完", expanded=False):
        st.caption("自動取得できない場合に使います。通常は空欄だけを補完し、チェックを入れた時だけ既存値を上書きします。")
        with st.form("manual_loto6_prize_fill_form"):
            round_no = st.number_input("開催回", min_value=1, value=int(default_round), step=1)
            count_cols = st.columns(5)
            amount_cols = st.columns(5)
            input_values = {}
            for index, column in enumerate(PRIZE_COUNT_COLUMNS):
                input_values[column] = count_cols[index].text_input(column, value="", placeholder="例: 2口")
            for index, column in enumerate(PRIZE_AMOUNT_COLUMNS):
                input_values[column] = amount_cols[index].text_input(column, value="", placeholder="例: 124,941,200円")
            input_values["キャリーオーバー"] = st.text_input("キャリーオーバー", value="", placeholder="例: 0円")
            overwrite_existing = st.checkbox("既存値を上書きする", value=False)
            submitted = st.form_submit_button("口数・賞金を補完")

        if not submitted:
            return

        try:
            report = fill_loto6_prize_values_manually(int(round_no), input_values, overwrite_existing=overwrite_existing)
            message = format_manual_prize_report(report)
            st.session_state["loto6_manual_prize_message"] = message
            st.success(message)
        except Exception as exc:
            st.error(str(exc))


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

    save_clicked = st.button("ロト6標準予測を保存", key="loto6_standard_prediction_save")
    saved_count = 0
    save_result = {"saved": 0, "skipped": 0, "errors": [], "prediction_ids": []}
    if save_clicked:
        save_result = save_prediction_picks_detailed(picks, target_round, active_model_name)
        saved_count = save_result["saved"]
        if save_result["prediction_ids"]:
            st.caption(f"保存した予想ID: {', '.join(save_result['prediction_ids'])}")
        if save_result["skipped"]:
            st.warning(f"重複のため {save_result['skipped']} 件をスキップしました。")
        for error in save_result["errors"]:
            st.warning(error)
    if saved_count:
        st.caption(f"予測研究所ログ: 第{target_round}回の予想{saved_count}件を predictions.csv に保存しました。")
    elif save_clicked:
        st.info("同じ抽選回・モデル・予測番号、または同じ予想IDのロト6標準予測は保存済みです。")
    else:
        st.caption(f"第{target_round}回のロト6標準予測を表示しています。保存ボタンを押すまで predictions.csv は更新されません。")

    for index, pick in enumerate(picks, start=1):
        render_pattern_pick(pick, f"第{index}候補")


def render_next_score_prediction(score_df, results, target_round):
    st.subheader("候補スコア活用予測")
    picks = generate_next_score_prediction_picks(score_df, results)
    if not picks:
        st.info("候補スコアCSVと抽せん履歴が揃うと、スコアベース次回予測を表示します。")
        return
    if st.button("候補スコア活用予測を保存", key="loto6_next_score_prediction_save"):
        result = save_prediction_picks_detailed(picks, target_round, "候補スコア活用予測")
        render_loto6_prediction_save_result(
            result,
            "候補スコア活用予測を predictions.csv に {saved}件保存しました。",
            "候補スコア活用予測の同一予想は保存済みです。",
        )
    for index, pick in enumerate(picks, start=1):
        render_pattern_pick(pick, f"候補スコア活用予測 {index}")
        if not pick.get("number_reasons", pd.DataFrame()).empty:
            st.dataframe(pick["number_reasons"], width="stretch", hide_index=True)


def render_ai_improvement_weight_summary(weight_summary):
    st.subheader("AI改善重み")
    for warning in (weight_summary or {}).get("warnings", []):
        st.warning(warning)
    cols = st.columns(4)
    cols[0].metric("AI改善履歴", f"{int((weight_summary or {}).get('history_count', 0))}件")
    cols[1].metric("モデル履歴", f"{int((weight_summary or {}).get('model_history_count', 0))}件")
    cols[2].metric("対象回", str((weight_summary or {}).get("latest_draw_no", "-")))
    cols[3].metric("最終更新", str((weight_summary or {}).get("latest_created_at", "-")))
    st.dataframe(ai_improvement_weight_rows(weight_summary), width="stretch", hide_index=True)
    hypothesis = str((weight_summary or {}).get("latest_hypothesis", "-") or "-")
    st.write(f"直近の改善仮説: {hypothesis}")
    if not (weight_summary or {}).get("available"):
        st.info("AI改善重みが十分でないため、AI改善反映予測は通常の候補スコア活用予測にフォールバックします。")


def render_ai_weighted_prediction(score_df, results, target_round, weight_summary):
    st.subheader("AI改善反映予測")
    weighted_df = apply_ai_improvement_weights(score_df, weight_summary)
    if not weighted_df.empty:
        st.markdown("**AI改善反映後の候補上位**")
        st.dataframe(ai_score_display_df(weighted_df).head(10), width="stretch", hide_index=True)
    picks = generate_ai_weighted_prediction_picks(score_df, results, weight_summary)
    if not picks:
        st.info("候補スコアCSVと抽せん履歴が揃うと、AI改善反映予測を表示します。")
        return
    if st.button("AI改善反映予測を保存", key="loto6_ai_weighted_prediction_save"):
        result = save_prediction_picks_detailed(picks, target_round, "AI改善反映予測")
        render_loto6_prediction_save_result(
            result,
            "AI改善反映予測を predictions.csv に {saved}件保存しました。",
            "AI改善反映予測の同一予想は保存済みです。",
        )
    for index, pick in enumerate(picks, start=1):
        render_pattern_pick(pick, f"AI改善反映予測 {index}")
        if not pick.get("number_reasons", pd.DataFrame()).empty:
            st.dataframe(pick["number_reasons"], width="stretch", hide_index=True)


def render_winning_condition_history():
    history, model_history = load_winning_condition_history(AI_IMPROVEMENT_DIR, "loto6")
    if history.empty:
        st.info("当選条件分析は、予想と実結果を照合すると data/ai_improvement に保存されます。")
        return

    history = history.copy()
    history["draw_no"] = pd.to_numeric(history["draw_no"], errors="coerce").fillna(0)
    latest = history.sort_values(["draw_no", "created_at"], ascending=[False, False]).iloc[0]
    analysis = parse_json_text(latest.get("winning_condition_analysis"), {})
    ensemble = parse_json_text(latest.get("ensemble_analysis"), {})

    cols = st.columns(4)
    cols[0].metric("対象回", int(latest["draw_no"]))
    cols[1].metric("一致数", int(pd.to_numeric(pd.Series([latest["matched_count"]]), errors="coerce").fillna(0).iloc[0]))
    cols[2].metric("追加すべきだった数字", latest.get("should_have_included_numbers", "") or "なし")
    cols[3].metric("除外すべきだった数字", latest.get("should_have_excluded_numbers", "") or "なし")

    st.markdown("**予測と実結果の比較**")
    st.write(f"予測番号: {latest['predicted_numbers']} / 実際の当選番号: {latest['actual_numbers']} / 一致: {latest.get('matched_numbers', '') or 'なし'}")
    st.write(f"なぜ外れたか: {latest.get('failure_reason', '-')}")
    st.write(f"どうなれば近づけたか: {analysis.get('必要だった条件', '-')}")
    st.write(f"今回有効だった分析手法: {latest.get('useful_models', '-')}")
    st.write(f"今回弱かった分析手法: {latest.get('weak_models', '-')}")
    st.write(f"次回上げるべき重み: {latest.get('weight_up_models', '-')}")
    st.write(f"次回下げるべき重み: {latest.get('weight_down_models', '-')}")
    st.write(f"次回の改善仮説: {latest.get('next_hypothesis', '-')}")

    number_detail = analysis.get("数字別特徴", [])
    if number_detail:
        st.markdown("**的中に近づくために必要だった条件分析**")
        st.dataframe(pd.DataFrame(number_detail), width="stretch", hide_index=True)

    if ensemble:
        st.markdown("**後追い最適化: アンサンブル候補**")
        st.dataframe(pd.DataFrame([ensemble]), width="stretch", hide_index=True)

    if not model_history.empty:
        latest_models = model_history[model_history["prediction_id"].astype(str) == str(latest["prediction_id"])]
        if not latest_models.empty:
            show_columns = [
                "model_name",
                "predicted_numbers",
                "matched_count",
                "should_have_included_numbers",
                "should_have_excluded_numbers",
                "needed_conditions",
                "next_hypothesis",
            ]
            st.markdown("**モデル別の改善ポイント**")
            st.dataframe(latest_models.reindex(columns=show_columns), width="stretch", hide_index=True)

    with st.expander("当選条件分析の保存履歴"):
        st.dataframe(history.sort_values(["draw_no", "created_at"], ascending=[False, False]), width="stretch", hide_index=True)


def purchase_method_options():
    options = [
        "AI改善反映予測",
        "候補スコア活用予測",
        *BACKTEST_MODELS.values(),
        "手入力",
        "その他",
    ]
    return list(dict.fromkeys(options))


def render_purchase_summary(purchases, lottery_type):
    summary = build_purchase_summary(purchases, lottery_type)
    cols = st.columns(4)
    cols[0].metric("総購入金額", f"{int(round(summary['総購入金額'])):,}円")
    cols[1].metric("総払戻金", f"{int(round(summary['総払戻金'])):,}円")
    cols[2].metric("累計収支", f"{int(round(summary['累計収支'])):,}円")
    cols[3].metric("回収率", f"{summary['回収率']}%")

    cols = st.columns(3)
    cols[0].metric("的中回数", int(summary["的中回数"]))
    cols[1].metric("最高払戻金", f"{int(round(summary['最高払戻金'])):,}円")
    cols[2].metric("実戦上位方式", summary["実戦成績が良い予測方式"])


def render_purchase_group_tables(game_purchases):
    method_summary = build_purchase_group_summary(game_purchases, "prediction_method", "予測方式")
    model_summary = build_purchase_group_summary(game_purchases, "model_name", "モデル名")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**予測方式別の収支**")
        if method_summary.empty:
            st.info("予測方式別の集計はまだありません。")
        else:
            st.dataframe(method_summary, width="stretch", hide_index=True)
    with col_right:
        st.markdown("**モデル別の収支**")
        if model_summary.empty:
            st.info("モデル別の集計はまだありません。")
        else:
            st.dataframe(model_summary, width="stretch", hide_index=True)


def render_purchase_manager(result_history):
    st.subheader("買い目管理")
    st.caption("実際に購入した買い目、購入額、当選結果、払戻金、収支を記録します。")
    try:
        purchases = read_csv(PURCHASES_CSV, PURCHASE_COLUMNS)
    except Exception as exc:
        st.warning(f"purchases.csv を読み込めませんでした。空の購入履歴として表示します: {exc}")
        purchases = pd.DataFrame(columns=PURCHASE_COLUMNS)
    try:
        evaluated = evaluate_purchase_history(purchases, result_history, "loto6", 6, 43)
    except Exception as exc:
        st.warning(f"購入履歴の照合中に問題がありました。未照合の一覧として表示します: {exc}")
        evaluated = purchases.copy()
        for column in PURCHASE_COLUMNS:
            if column not in evaluated:
                evaluated[column] = ""
        evaluated = evaluated.reindex(columns=PURCHASE_COLUMNS)

    game_purchases = evaluated[evaluated["lottery_type"].astype(str) == "loto6"].copy() if not evaluated.empty else evaluated
    render_purchase_summary(evaluated, "loto6")

    latest_round = 1
    if result_history is not None and not result_history.empty and "開催回" in result_history:
        rounds = pd.to_numeric(result_history["開催回"], errors="coerce").dropna()
        if not rounds.empty:
            latest_round = int(rounds.max()) + 1

    with st.form("loto6_purchase_form"):
        st.markdown("**購入履歴を追加**")
        col_left, col_right = st.columns(2)
        with col_left:
            draw_no = st.number_input("開催回", min_value=1, value=latest_round, step=1)
            purchase_date = st.date_input("購入日", value=date.today())
            numbers_text = st.text_input("購入番号（6個）", placeholder="01-02-03-04-05-06")
            prediction_method = st.selectbox("予測方式", purchase_method_options(), index=0)
        with col_right:
            model_name = st.text_input("モデル名", value=prediction_method)
            ticket_count = st.number_input("口数", min_value=1, value=1, step=1)
            cost = st.number_input("購入金額", min_value=0, value=200, step=100)
            manual_payout = st.number_input("払戻金（賞金未入力時のみ任意）", min_value=0, value=0, step=100)
        notes = st.text_area("メモ", height=80)
        submitted = st.form_submit_button("購入履歴を保存")

    if submitted:
        numbers, errors = validate_purchase_numbers(numbers_text, 6, 43)
        if errors:
            for error in errors:
                st.warning(error)
        else:
            row = {
                "lottery_type": "loto6",
                "draw_no": int(draw_no),
                "purchase_date": purchase_date.strftime("%Y/%m/%d"),
                "numbers": numbers_to_text(numbers),
                "prediction_method": prediction_method,
                "model_name": model_name.strip() or prediction_method,
                "ticket_count": int(ticket_count),
                "cost": int(cost),
                "result_numbers": "",
                "bonus_numbers": "",
                "matched_count": "",
                "bonus_matched_count": "",
                "prize_rank": "",
                "payout": int(manual_payout) if int(manual_payout) > 0 else "",
                "profit_loss": "",
                "status": "未照合",
                "notes": notes,
                "created_at": now_text(),
            }
            try:
                updated = pd.concat([purchases, pd.DataFrame([row])], ignore_index=True)
                updated = evaluate_purchase_history(updated, result_history, "loto6", 6, 43)
                save_csv(updated, PURCHASES_CSV, PURCHASE_COLUMNS)
                st.success("購入履歴を保存しました。")
                rerun_app()
            except Exception as exc:
                st.error(f"購入履歴を保存できませんでした: {exc}")

    st.markdown("**購入履歴一覧**")
    display = purchase_display_df(evaluated, "loto6")
    if display.empty:
        st.info("ロト6の購入履歴はまだありません。")
    else:
        st.dataframe(display, width="stretch", hide_index=True)
        if st.button("照合結果を purchases.csv に反映", key="loto6_purchase_sync"):
            try:
                save_csv(evaluated, PURCHASES_CSV, PURCHASE_COLUMNS)
                st.success("購入履歴の照合結果を保存しました。")
            except Exception as exc:
                st.error(f"照合結果を保存できませんでした: {exc}")

    render_purchase_group_tables(game_purchases)
    ai_rows = game_purchases[
        game_purchases["prediction_method"].astype(str).str.contains("AI改善反映予測", na=False)
        | game_purchases["model_name"].astype(str).str.contains("AI改善反映予測", na=False)
    ] if not game_purchases.empty else pd.DataFrame(columns=PURCHASE_COLUMNS)
    st.markdown("**AI改善反映予測の実戦成績**")
    ai_summary = build_purchase_group_summary(ai_rows, "prediction_method", "予測方式")
    if ai_summary.empty:
        st.info("AI改善反映予測の購入履歴はまだありません。")
    else:
        st.dataframe(ai_summary, width="stretch", hide_index=True)


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

    fixed_predictions, target_round, fixed_status = build_fixed_prediction_overview(
        predictions,
        official_results,
        draw_size=6,
        number_max=43,
    )
    if fixed_predictions.empty:
        st.info(fixed_status)
    else:
        st.markdown(f"**第{target_round}回 固定予測（次回検証対象）**")
        st.caption(f"状態: {fixed_status}。抽選結果登録後、この{len(fixed_predictions)}件を検証レポートへ照合します。")
        st.dataframe(fixed_predictions, width="stretch", hide_index=True)

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

    lab_tabs = st.tabs(["バックテスト", "検証レポート", "モデル貢献度", "条件別成功率", "AI改善レポート", "当選条件分析", "動画仮説研究", "予想履歴", "公式結果", "買い目管理", "高額当選・連続当選モード"])
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
        _, model_history = load_winning_condition_history(AI_IMPROVEMENT_DIR, "loto6")
        dashboard = build_model_dashboard(
            reports,
            draw_size=6,
            model_history=model_history,
            include_target_models=True,
        )
        if ranking.empty and dashboard.empty:
            st.info("モデル貢献度は、予想と結果を照合すると作成されます。")
        else:
            st.markdown("**モデル成績ランキング**")
            st.dataframe(dashboard, width="stretch", hide_index=True)
            with st.expander("ChaminiSP / バランス仮説 研究成績", expanded=False):
                st.markdown("**ChaminiSP God Mode 総合成績**")
                st.dataframe(build_chamini_sp_performance_summary(reports, draw_size=6), width="stretch", hide_index=True)
                balance_stats = build_balance_hypothesis_performance(reports, draw_size=6)
                st.markdown("**balance hypothesis 研究成績**")
                st.dataframe(balance_stats["overview"], width="stretch", hide_index=True)
                if not balance_stats["grade"].empty:
                    st.markdown("**grade別成績**")
                    st.dataframe(balance_stats["grade"], width="stretch", hide_index=True)
                if not balance_stats["score_groups"].empty:
                    st.markdown("**高スコア・低スコア群比較**")
                    st.dataframe(balance_stats["score_groups"], width="stretch", hide_index=True)
                if not balance_stats["subscores"].empty:
                    st.markdown("**サブスコア別研究成績**")
                    st.dataframe(balance_stats["subscores"], width="stretch", hide_index=True)
                unverified = build_unverified_chamini_sp_predictions(predictions, official_results, reports, draw_size=6, number_max=43)
                st.markdown("**未検証のChaminiSP保存済み予想**")
                if unverified.empty:
                    st.info("未検証のChaminiSP予想はありません。")
                else:
                    st.dataframe(unverified, width="stretch", hide_index=True)
            st.markdown("**モデル貢献度ランキング**")
            if ranking.empty:
                st.info("貢献度ランキングは、的中数字の要因分析後に表示されます。")
            else:
                st.dataframe(ranking, width="stretch", hide_index=True)
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
        render_winning_condition_history()
    with lab_tabs[6]:
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
    with lab_tabs[7]:
        if predictions.empty:
            st.info("次回予想買い目が表示されると predictions.csv に保存されます。")
        else:
            st.dataframe(predictions.sort_values(["開催回", "候補番号"], ascending=[False, True]), width="stretch", hide_index=True)
    with lab_tabs[8]:
        if official_results.empty:
            st.info("当選結果を登録すると results.csv に研究所用の公式結果ログを保存します。")
        else:
            st.dataframe(official_results.sort_values("開催回", ascending=False), width="stretch", hide_index=True)
    with lab_tabs[9]:
        render_purchase_manager(results)
    with lab_tabs[10]:
        render_high_prize_continuous_mode(results, reports)


if app_light_smoke_mode():
    render_light_smoke_overview()
    st.stop()


render_registration_form()

if "registration_message" in st.session_state:
    st.success(st.session_state.pop("registration_message"))

st.info("当選結果を入力する場合は、画面左側のサイドバーにある「新しい当選結果を登録」を使ってください。サイドバーが見えない場合は、左上の矢印で開けます。")

action_cols = st.columns(4)
if action_cols[0].button("ロト6最新結果を自動更新"):
    try:
        with st.spinner("最新結果を取得し、CSV更新と再分析を実行しています..."):
            message = update_latest_result_from_web()
        st.session_state["loto6_update_message"] = message
        st.success(message)
    except Exception as exc:
        st.error(str(exc))

if action_cols[1].button("ロト6の口数・賞金を再取得"):
    try:
        with st.spinner("口数・賞金を再取得しています..."):
            message = update_latest_result_from_web(run_analysis_after=False)
        st.session_state["loto6_update_message"] = message
        st.success(message)
    except Exception as exc:
        st.error(f"取得に失敗しました。時間をおいて再実行してください。詳細: {exc}")

if action_cols[2].button("空欄の賞金情報を補完"):
    try:
        with st.spinner("空欄の賞金情報を補完しています..."):
            message = update_latest_result_from_web(run_analysis_after=False)
        st.session_state["loto6_update_message"] = message
        st.success(message)
    except Exception as exc:
        st.error(f"補完に失敗しました。既存データは変更されていません。詳細: {exc}")

if action_cols[3].button("手元のCSVで再分析"):
    try:
        with st.spinner("分析を更新しています..."):
            run_analysis()
        st.success("分析を更新しました。")
    except Exception as exc:
        st.error(str(exc))

if st.session_state.get("loto6_update_message"):
    st.info(st.session_state["loto6_update_message"])

render_manual_prize_fill_form()

if st.session_state.get("loto6_manual_prize_message"):
    st.info(st.session_state["loto6_manual_prize_message"])


results = merge_official_results(read_csv(RESULTS_CSV, RESULT_COLUMNS))
sets = read_csv(SETS_CSV, SET_COLUMNS)
scores, score_warnings = load_next_score_csv(SCORES_CSV, 43)
ai_weight_summary = build_ai_improvement_weight_summary(AI_IMPROVEMENT_DIR, "loto6")

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
top_count = st.selectbox("表示件数", [10, 20, 30, 43], index=1)
for warning in score_warnings:
    st.warning(warning)
render_ai_improvement_weight_summary(ai_weight_summary)
set_ball_analysis = render_set_ball_analysis_panel(results, sets)
if scores.empty:
    st.info("候補スコアCSVがありません。上の「手元のCSVで再分析」から作成できます。")
else:
    st.dataframe(score_display_df(scores).head(top_count), width="stretch", hide_index=True)
    render_prediction_picks(scores, results, latest_round + 1)
    render_next_score_prediction(scores, results, latest_round + 1)
    render_ai_weighted_prediction(scores, results, latest_round + 1, ai_weight_summary)

render_chamini6_prediction(results, latest_round + 1, ai_weight_summary, set_ball_analysis)
render_prediction_flow_diagnostics("loto6", "ロト6")
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
