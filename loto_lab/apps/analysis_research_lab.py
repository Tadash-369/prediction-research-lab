from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
LOTO_LAB_DIR = APP_DIR.parent
PROJECT_ROOT = LOTO_LAB_DIR.parent
CORE_DIR = LOTO_LAB_DIR / "core"
DATA_DIR = LOTO_LAB_DIR / "data"
VERIFICATION_DIR = DATA_DIR / "verification"
AI_IMPROVEMENT_DIR = DATA_DIR / "ai_improvement"
DOCS_DIR = LOTO_LAB_DIR / "docs"
BACKUP_DIR = DATA_DIR / "backups"

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import pandas as pd
import streamlit as st

from arl_research_engine import (
    CONTRIBUTION_COLUMNS,
    FUTURE_LOTO_MODEL_LABELS,
    LOTO_MODEL_LABELS,
    PURCHASE_COLUMNS,
    PROJECT_ENGLISH_NAME,
    PROJECT_JAPANESE_NAME,
    PROJECT_SHORT_NAME,
    RESEARCH_CYCLE_COLUMNS,
    VIDEO_HYPOTHESIS_COLUMNS,
    add_verification_metrics,
    build_ai_improvement_weight_summary,
    build_ai_improvement_summary,
    build_condition_success_table,
    build_contribution_ranking,
    build_fixed_prediction_overview,
    build_model_dashboard,
    build_purchase_summary,
    build_research_flow_table,
    evaluate_purchase_history,
    load_winning_condition_history,
    parse_json_text,
    weighted_model_text,
)
from prl_maintenance import run_maintenance


BASE_DIR = LOTO_LAB_DIR
PURCHASES_CSV = DATA_DIR / "purchases.csv"


def read_csv(path, columns=None):
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def save_csv(df, path, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns:
        df = df.reindex(columns=columns)
    df.to_csv(path, index=False, encoding="utf-8-sig")


DISPLAY_TEXT_COLUMNS = [
    "予想保存数",
    "予測保存数",
    "検証数",
    "平均一致数",
    "平均期待値",
    "平均的中率",
    "平均勝率",
    "状態",
]


def dataframe_for_display(df, text_columns=None):
    display_df = df.copy()
    for column in (text_columns or DISPLAY_TEXT_COLUMNS):
        if column in display_df.columns:
            display_df[column] = display_df[column].astype(str)
    for column in display_df.columns:
        values = display_df[column].dropna()
        if values.empty:
            continue
        has_placeholder = values.astype(str).str.strip().isin(["-", "－", ""]).any()
        has_numeric = pd.to_numeric(values, errors="coerce").notna().any()
        if has_placeholder and has_numeric:
            display_df[column] = display_df[column].astype(str)
    return display_df


def display_dataframe(df, **kwargs):
    st.dataframe(dataframe_for_display(df), **kwargs)


def numeric_column(df, column_name, default=0):
    if column_name not in df:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[column_name], errors="coerce").fillna(default)


def latest_round_text(df):
    if df.empty or "開催回" not in df:
        return "-"
    rounds = pd.to_numeric(df["開催回"], errors="coerce").dropna()
    return str(int(rounds.max())) if not rounds.empty else "-"


def average_metric_text(df, column, suffix=""):
    if df.empty or column not in df:
        return "-"
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return "-"
    return f"{round(float(values.mean()), 3)}{suffix}"


def score_csv_overview(label, path, number_max, top_n=10):
    if not path.exists():
        return {
            "対象": label,
            "CSV": "なし",
            "上位候補数字": "-",
            "最終更新": "-",
            "状態": f"{path.name} がありません",
        }
    try:
        score_df = read_csv(path)
    except Exception as exc:
        return {
            "対象": label,
            "CSV": "読込不可",
            "上位候補数字": "-",
            "最終更新": "-",
            "状態": f"読み込みエラー: {exc}",
        }
    if score_df.empty:
        return {"対象": label, "CSV": "あり", "上位候補数字": "-", "最終更新": "-", "状態": "CSVが空です"}
    missing = [column for column in ("数字", "スコア") if column not in score_df.columns]
    if missing:
        return {
            "対象": label,
            "CSV": "あり",
            "上位候補数字": "-",
            "最終更新": "-",
            "状態": f"必要な列がありません: {', '.join(missing)}",
        }
    score_df = score_df.copy()
    score_df["数字"] = pd.to_numeric(score_df["数字"], errors="coerce")
    score_df["スコア"] = pd.to_numeric(score_df["スコア"], errors="coerce")
    score_df = score_df.dropna(subset=["数字", "スコア"])
    score_df = score_df[(score_df["数字"] >= 1) & (score_df["数字"] <= number_max)]
    if score_df.empty:
        return {"対象": label, "CSV": "あり", "上位候補数字": "-", "最終更新": "-", "状態": "有効な数字がありません"}
    score_df = score_df.sort_values("スコア", ascending=False)
    top_numbers = " ".join(f"{int(number):02d}" for number in score_df.head(top_n)["数字"].tolist())
    updated = "-"
    if "更新日時" in score_df.columns:
        updated_values = score_df["更新日時"].dropna().astype(str)
        updated_values = updated_values[updated_values.str.strip() != ""]
        if not updated_values.empty:
            updated = updated_values.iloc[0]
    return {"対象": label, "CSV": "あり", "上位候補数字": top_numbers, "最終更新": updated, "状態": "OK"}


def ai_weight_overview(label, lottery_type):
    summary = build_ai_improvement_weight_summary(AI_IMPROVEMENT_DIR, lottery_type)
    warnings = summary.get("warnings", [])
    status = "OK" if summary.get("available") else "候補スコア予測へフォールバック"
    if warnings:
        status = warnings[0]
    hypothesis = str(summary.get("latest_hypothesis", "-") or "-")
    if len(hypothesis) > 80:
        hypothesis = hypothesis[:77] + "..."
    return {
        "対象": label,
        "AI改善履歴": f"{int(summary.get('history_count', 0))}件",
        "重み上げモデル": weighted_model_text(summary.get("model_weights", {}), True),
        "重み下げモデル": weighted_model_text(summary.get("model_weights", {}), False),
        "直近仮説": hypothesis,
        "最終更新": summary.get("latest_created_at", "-"),
        "状態": status,
    }


def purchase_summary_row(label, summary):
    return {
        "対象": label,
        "総購入金額": int(round(summary["総購入金額"])),
        "総払戻金": int(round(summary["総払戻金"])),
        "累計収支": int(round(summary["累計収支"])),
        "回収率": f"{summary['回収率']}%",
        "的中回数": int(summary["的中回数"]),
        "最高払戻金": int(round(summary["最高払戻金"])),
        "実戦成績が良い予測方式": summary["実戦成績が良い予測方式"],
    }


def load_purchase_tracking_overview():
    warnings = []
    try:
        purchases = read_csv(PURCHASES_CSV, PURCHASE_COLUMNS)
    except Exception as exc:
        warnings.append(f"purchases.csv を読み込めませんでした: {exc}")
        return pd.DataFrame(columns=PURCHASE_COLUMNS), warnings
    if purchases.empty:
        return purchases, warnings
    try:
        purchases = evaluate_purchase_history(purchases, read_csv(DATA_DIR / "loto6.csv"), "loto6", 6, 43)
    except Exception as exc:
        warnings.append(f"ロト6購入履歴の照合をスキップしました: {exc}")
    try:
        purchases = evaluate_purchase_history(purchases, read_csv(DATA_DIR / "loto7.csv"), "loto7", 7, 37)
    except Exception as exc:
        warnings.append(f"ロト7購入履歴の照合をスキップしました: {exc}")
    return purchases, warnings


def render_home():
    loto6_predictions = read_csv(DATA_DIR / "predictions.csv")
    loto6_official = read_csv(DATA_DIR / "results.csv")
    loto6_reports = add_verification_metrics(read_csv(VERIFICATION_DIR / "verification_reports.csv"), draw_size=6)
    loto7_predictions = read_csv(DATA_DIR / "loto7_predictions.csv")
    loto7_official = read_csv(DATA_DIR / "loto7_results.csv")
    loto7_reports = add_verification_metrics(read_csv(VERIFICATION_DIR / "loto7_verification_reports.csv"), draw_size=7)
    purchases, purchase_warnings = load_purchase_tracking_overview()
    docs = list(DOCS_DIR.glob("*.md")) if DOCS_DIR.exists() else []
    backups = list(BACKUP_DIR.glob("*.csv")) if BACKUP_DIR.exists() else []

    cols = st.columns(4)
    cols[0].metric("ロト6 最新検証回", latest_round_text(loto6_reports))
    cols[1].metric("ロト7 最新検証回", latest_round_text(loto7_reports))
    cols[2].metric("ドキュメント", len(docs))
    cols[3].metric("バックアップ", len(backups))

    summary = pd.DataFrame(
        [
            {
                "部門": "ロト6",
                "予想保存数": len(loto6_predictions),
                "検証数": len(loto6_reports),
                "平均一致数": average_metric_text(loto6_reports, "本数字一致数"),
                "平均期待値": average_metric_text(loto6_reports, "期待値"),
                "状態": "運用中" if len(loto6_reports) else "検証待ち",
            },
            {
                "部門": "ロト7",
                "予想保存数": len(loto7_predictions),
                "検証数": len(loto7_reports),
                "平均一致数": average_metric_text(loto7_reports, "本数字一致数"),
                "平均期待値": average_metric_text(loto7_reports, "期待値"),
                "状態": "運用中" if len(loto7_reports) else "検証待ち",
            },
            {"部門": "AI改善", "予想保存数": "-", "検証数": len(loto6_reports) + len(loto7_reports), "平均一致数": "-", "平均期待値": "-", "状態": "運用中"},
            {"部門": "保守", "予想保存数": "-", "検証数": "-", "平均一致数": "-", "平均期待値": "-", "状態": "同期・補完対応"},
        ]
    )
    for column in ["予想保存数", "検証数", "平均一致数", "平均期待値"]:
        summary[column] = summary[column].astype(str)
    st.markdown("**研究所サマリー**")
    display_dataframe(summary, width="stretch", hide_index=True)

    st.markdown("**次回検証対象の固定予測**")
    prediction_tabs = st.tabs(["ロト6", "ロト7"])
    for tab, label, predictions, official, draw_size, number_max in [
        (prediction_tabs[0], "ロト6", loto6_predictions, loto6_official, 6, 43),
        (prediction_tabs[1], "ロト7", loto7_predictions, loto7_official, 7, 37),
    ]:
        with tab:
            fixed_predictions, target_round, fixed_status = build_fixed_prediction_overview(
                predictions,
                official,
                draw_size=draw_size,
                number_max=number_max,
            )
            if fixed_predictions.empty:
                st.info(f"{label}: {fixed_status}")
            else:
                st.caption(f"{label} 第{target_round}回 / 状態: {fixed_status}")
                display_dataframe(fixed_predictions, width="stretch", hide_index=True)

    score_overview = pd.DataFrame(
        [
            score_csv_overview("ロト6", DATA_DIR / "loto6_next_number_scores.csv", 43),
            score_csv_overview("ロト7", DATA_DIR / "loto7_next_number_scores.csv", 37),
        ]
    )
    st.markdown("**候補スコア概要**")
    st.dataframe(score_overview, width="stretch", hide_index=True)

    ai_overview = pd.DataFrame(
        [
            ai_weight_overview("ロト6", "loto6"),
            ai_weight_overview("ロト7", "loto7"),
        ]
    )
    st.markdown("**AI改善重み概要**")
    st.dataframe(ai_overview, width="stretch", hide_index=True)

    for warning in purchase_warnings:
        st.warning(warning)
    purchase_overview = pd.DataFrame(
        [
            purchase_summary_row("ロト6", build_purchase_summary(purchases, "loto6")),
            purchase_summary_row("ロト7", build_purchase_summary(purchases, "loto7")),
            purchase_summary_row("全体", build_purchase_summary(purchases)),
        ]
    )
    st.markdown("**購入・払戻概要**")
    st.dataframe(purchase_overview, width="stretch", hide_index=True)

    next_actions = pd.DataFrame(
        [
            {"優先": 1, "次の作業": "ロト6またはロト7で最新当選結果を登録し、検証レポートを更新する"},
            {"優先": 2, "次の作業": "保守部門でVer1.0補完とフォルダ同期を実行する"},
            {"優先": 3, "次の作業": "AI改善部門で成功要因、失敗要因、改善案、次回仮説を確認する"},
            {"優先": 4, "次の作業": "モデル別成績と条件別成功率を見て、次回予想の仮説を決める"},
        ]
    )
    st.markdown("**次にやること**")
    display_dataframe(next_actions, width="stretch", hide_index=True)


def render_model_catalog():
    model_df = pd.DataFrame(
        [{"番号": index, "区分": "現行", "モデル": label} for index, label in enumerate(LOTO_MODEL_LABELS.values(), start=1)]
        + [
            {"番号": index, "区分": "将来追加予定", "モデル": label}
            for index, label in enumerate(FUTURE_LOTO_MODEL_LABELS.values(), start=len(LOTO_MODEL_LABELS) + 1)
        ]
    )
    display_dataframe(model_df, width="stretch", hide_index=True)


def render_winning_condition_panel(lottery_type, name):
    history, model_history = load_winning_condition_history(AI_IMPROVEMENT_DIR, lottery_type)
    if history.empty:
        st.info(f"{name}の当選条件分析は、予想と実結果を照合すると data/ai_improvement に保存されます。")
        return

    history = history.copy()
    history["draw_no"] = pd.to_numeric(history["draw_no"], errors="coerce").fillna(0)
    latest = history.sort_values(["draw_no", "created_at"], ascending=[False, False]).iloc[0]
    analysis = parse_json_text(latest.get("winning_condition_analysis"), {})
    ensemble = parse_json_text(latest.get("ensemble_analysis"), {})

    cols = st.columns(4)
    cols[0].metric("対象回", int(latest["draw_no"]))
    cols[1].metric("一致数", latest.get("matched_count", "-"))
    cols[2].metric("追加候補", latest.get("should_have_included_numbers", "") or "なし")
    cols[3].metric("除外候補", latest.get("should_have_excluded_numbers", "") or "なし")

    st.write(f"予測番号: {latest['predicted_numbers']} / 実際: {latest['actual_numbers']}")
    st.write(f"外れた理由: {latest.get('failure_reason', '-')}")
    st.write(f"的中に近づくために必要だった条件: {analysis.get('必要だった条件', '-')}")
    st.write(f"今回有効だった分析手法: {latest.get('useful_models', '-')}")
    st.write(f"今回弱かった分析手法: {latest.get('weak_models', '-')}")
    st.write(f"次回上げるべき重み: {latest.get('weight_up_models', '-')}")
    st.write(f"次回下げるべき重み: {latest.get('weight_down_models', '-')}")
    st.write(f"次回の改善仮説: {latest.get('next_hypothesis', '-')}")

    number_detail = analysis.get("数字別特徴", [])
    if number_detail:
        st.markdown("**数字別特徴**")
        st.dataframe(pd.DataFrame(number_detail), width="stretch", hide_index=True)

    if ensemble:
        st.markdown("**後追い最適化: アンサンブル候補**")
        st.dataframe(pd.DataFrame([ensemble]), width="stretch", hide_index=True)

    if not model_history.empty:
        latest_models = model_history[model_history["prediction_id"].astype(str) == str(latest["prediction_id"])]
        if not latest_models.empty:
            st.markdown("**モデル別改善ポイント**")
            st.dataframe(
                latest_models.reindex(
                    columns=[
                        "model_name",
                        "predicted_numbers",
                        "matched_count",
                        "should_have_included_numbers",
                        "should_have_excluded_numbers",
                        "needed_conditions",
                        "next_hypothesis",
                    ]
                ),
                width="stretch",
                hide_index=True,
            )


def render_loto_lab(name, prediction_file, result_file, report_file, contribution_file, cycle_file, number_max, draw_size):
    predictions = read_csv(DATA_DIR / prediction_file)
    official = read_csv(DATA_DIR / result_file)
    reports = add_verification_metrics(read_csv(VERIFICATION_DIR / report_file), draw_size=draw_size)
    contributions = read_csv(VERIFICATION_DIR / contribution_file, CONTRIBUTION_COLUMNS)
    cycles = read_csv(VERIFICATION_DIR / cycle_file, RESEARCH_CYCLE_COLUMNS)

    cols = st.columns(4)
    cols[0].metric("予想履歴", len(predictions))
    cols[1].metric("公式結果ログ", len(official))
    cols[2].metric("検証レポート", len(reports))
    cols[3].metric("貢献度ログ", len(contributions))

    fixed_predictions, target_round, fixed_status = build_fixed_prediction_overview(
        predictions,
        official,
        draw_size=draw_size,
        number_max=number_max,
    )
    if fixed_predictions.empty:
        st.info(f"{name}: {fixed_status}")
    else:
        st.markdown(f"**{name} 第{target_round}回 固定予測（次回検証対象）**")
        st.caption(f"状態: {fixed_status}")
        display_dataframe(fixed_predictions, width="stretch", hide_index=True)

    st.markdown("**研究フロー**")
    display_dataframe(build_research_flow_table(), width="stretch", hide_index=True)

    lottery_type = "loto6" if "ロト6" in name else "loto7"
    _, model_history = load_winning_condition_history(AI_IMPROVEMENT_DIR, lottery_type)
    tabs = st.tabs(["研究サイクル", "モデル別成績", "モデル貢献度", "条件別成功率", "AI改善レポート", "当選条件分析", "動画仮説"])
    with tabs[0]:
        if cycles.empty:
            st.info(f"{name}の研究サイクル履歴は、検証レポート作成後に保存されます。")
        else:
            display_dataframe(cycles.sort_values(["開催回", "予想ID"], ascending=[False, True]), width="stretch", hide_index=True)
    with tabs[1]:
        dashboard = build_model_dashboard(
            reports,
            draw_size=draw_size,
            model_history=model_history,
            include_target_models=True,
        )
        if dashboard.empty:
            st.info(f"{name}のモデル別成績は、予想と結果の照合後に表示されます。")
        else:
            display_dataframe(dashboard, width="stretch", hide_index=True)
    with tabs[2]:
        ranking = build_contribution_ranking(contributions)
        if ranking.empty:
            st.info(f"{name}の貢献度ランキングは、的中数字の要因分析後に表示されます。")
        else:
            display_dataframe(ranking, width="stretch", hide_index=True)
    with tabs[3]:
        condition_df = build_condition_success_table(reports, number_max=number_max)
        if condition_df.empty:
            st.info(f"{name}の条件別成功率は、検証レポート作成後に表示されます。")
        else:
            display_dataframe(condition_df, width="stretch", hide_index=True)
    with tabs[4]:
        summary = build_ai_improvement_summary(reports)
        metric_cols = st.columns(3)
        metric_cols[0].metric("一致数", summary["一致数"])
        metric_cols[1].metric("予想", summary["予想"])
        metric_cols[2].metric("結果", summary["結果"])
        st.write(f"的中要因: {summary['的中要因']}")
        st.write(f"外れ要因: {summary['外れ要因']}")
        st.write(f"改善案: {summary['改善案']}")
        st.write(f"次回仮説: {summary['次回仮説']}")
    with tabs[5]:
        render_winning_condition_panel(lottery_type, name)
    with tabs[6]:
        video_logs = read_csv(VERIFICATION_DIR / "video_hypotheses.csv", VIDEO_HYPOTHESIS_COLUMNS)
        if video_logs.empty:
            st.info("動画仮説ログはまだありません。")
        else:
            display_dataframe(video_logs.sort_values("保存日時", ascending=False), width="stretch", hide_index=True)

def render_ai_department():
    loto6_reports = add_verification_metrics(read_csv(VERIFICATION_DIR / "verification_reports.csv"), draw_size=6)
    loto7_reports = add_verification_metrics(read_csv(VERIFICATION_DIR / "loto7_verification_reports.csv"), draw_size=7)
    st.info("AI改善部門は、外れても失敗扱いせず、成功要因・失敗要因・改善案・次回仮説を研究資産として保存します。")
    tabs = st.tabs(["ロト6", "ロト7"])
    for tab, name, reports in [(tabs[0], "ロト6", loto6_reports), (tabs[1], "ロト7", loto7_reports)]:
        with tab:
            summary = build_ai_improvement_summary(reports)
            cols = st.columns(3)
            cols[0].metric("一致数", summary["一致数"])
            cols[1].metric("予想", summary["予想"])
            cols[2].metric("結果", summary["結果"])
            st.write(f"{name} 成功要因: {summary['的中要因']}")
            st.write(f"{name} 失敗要因: {summary['外れ要因']}")
            st.write(f"{name} 改善案: {summary['改善案']}")
            st.write(f"{name} 次回仮説: {summary['次回仮説']}")
            lottery_type = "loto6" if name == "ロト6" else "loto7"
            with st.expander(f"{name} 当選条件分析"):
                render_winning_condition_panel(lottery_type, name)


def render_verification_department():
    rows = []
    for name, report_file, draw_size in [("ロト6", "verification_reports.csv", 6), ("ロト7", "loto7_verification_reports.csv", 7)]:
        reports = add_verification_metrics(read_csv(VERIFICATION_DIR / report_file), draw_size=draw_size)
        if reports.empty:
            rows.append({"対象": name, "予測保存数": 0, "平均一致数": "-", "平均的中率": "-", "平均勝率": "-", "平均期待値": "-"})
            continue
        reports["本数字一致数"] = numeric_column(reports, "本数字一致数")
        reports["的中率"] = numeric_column(reports, "的中率")
        reports["勝率"] = numeric_column(reports, "勝率")
        reports["期待値"] = numeric_column(reports, "期待値")
        rows.append(
            {
                "対象": name,
                "予測保存数": len(reports),
                "平均一致数": round(float(reports["本数字一致数"].mean()), 3),
                "平均的中率": f"{round(float(reports['的中率'].mean()), 1)}%",
                "平均勝率": f"{round(float(reports['勝率'].mean()), 1)}%",
                "平均期待値": round(float(reports["期待値"].mean()), 3),
            }
        )
    display_dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

def render_system_department():
    apps = pd.DataFrame(
        [
            {"アプリ": "ロト研究所アプリ", "機能": "CSV自動読込 / 当選履歴管理 / 分析実行 / 予想生成 / 結果保存 / モデル別成績表 / AI改善レポート", "状態": "運用中"},
        ]
    )
    display_dataframe(apps, width="stretch", hide_index=True)


def render_maintenance_department():
    st.info("既存CSVをVer1.0形式へ補完します。実行時は自動でバックアップを作成します。")
    cols = st.columns(2)
    if cols[0].button("補完内容を確認"):
        summary = run_maintenance(apply_changes=False)
        display_dataframe(summary, width="stretch", hide_index=True)
    if cols[1].button("Ver1.0補完を実行"):
        summary = run_maintenance(apply_changes=True)
        st.success("補完を実行しました。")
        display_dataframe(summary, width="stretch", hide_index=True)


def render_launch_guide():
    st.markdown("**専用画面を同時に使う場合は、画面ごとに別ポートで起動します。**")
    st.caption("トップ画面を開いたまま、別のターミナルでロト6・ロト7専用画面を起動できます。")
    guide = pd.DataFrame(
        [
            {"画面": "分析研究所トップ", "起動コマンド": ".\\.venv\\Scripts\\streamlit.exe run analysis_research_lab.py --server.port 8501"},
            {"画面": "ロト6専用画面", "起動コマンド": ".\\.venv\\Scripts\\streamlit.exe run loto6_streamlit_app.py --server.port 8502"},
            {"画面": "ロト7専用画面", "起動コマンド": ".\\.venv\\Scripts\\streamlit.exe run loto7_streamlit_app.py --server.port 8503"},

            {"画面": "分析研究所トップ", "起動コマンド": "python -m streamlit run loto_lab/apps/analysis_research_lab.py --server.port 8501", "URL": "http://localhost:8501"},
            {"画面": "ロト6分析", "起動コマンド": "python -m streamlit run loto_lab/apps/loto6_streamlit_app.py --server.port 8502", "URL": "http://localhost:8502"},
            {"画面": "ロト7分析", "起動コマンド": "python -m streamlit run loto_lab/apps/loto7_streamlit_app.py --server.port 8503", "URL": "http://localhost:8503"},
        ]
    )
    display_dataframe(guide, width="stretch", hide_index=True)
    st.markdown("**ブラウザで開くURL**")
    st.code(
        "トップ画面: http://localhost:8501\n"
        "ロト6専用画面: http://localhost:8502\n"
        "ロト7専用画面: http://localhost:8503",
        language="text",
    )


def render_roadmap():
    roadmap = pd.DataFrame(
        [
            {"Phase": "Phase1", "内容": "出現頻度分析 / ホット分析 / コールド分析 / ペア分析"},
            {"Phase": "Phase2", "内容": "予想保存 / 結果保存 / 成績管理"},
            {"Phase": "Phase3", "内容": "マルコフ / ベイズ / モンテカルロ"},
            {"Phase": "Phase4", "内容": "機械学習"},
            {"Phase": "Phase5", "内容": "AI研究員エージェント"},
            {"Phase": "Phase6", "内容": "完全自動研究所"},
        ]
    )
    display_dataframe(roadmap, width="stretch", hide_index=True)


st.set_page_config(page_title=f"{PROJECT_JAPANESE_NAME} | {PROJECT_SHORT_NAME}", layout="wide")
st.title(PROJECT_JAPANESE_NAME)
st.caption(f"{PROJECT_ENGLISH_NAME}（{PROJECT_SHORT_NAME}）")
st.info("目的は当選や利益そのものではなく、予測手法の研究・検証・改善を継続し、予測精度を向上させることです。")

lab = st.radio("部門", ["ホーム", "ロト6", "ロト7", "AI改善", "検証", "保守", "システム開発"], horizontal=True)

with st.expander("ロト分析モデル一覧", expanded=False):
    render_model_catalog()

with st.expander("起動ガイド", expanded=False):
    render_launch_guide()

with st.expander("開発ロードマップ", expanded=False):
    render_roadmap()

if lab == "ホーム":
    render_home()
elif lab == "ロト6":
    render_loto_lab(
        "ロト6",
        prediction_file="predictions.csv",
        result_file="results.csv",
        report_file="verification_reports.csv",
        contribution_file="loto6_model_contributions.csv",
        cycle_file="loto6_research_cycles.csv",
        number_max=43,
        draw_size=6,
    )
elif lab == "ロト7":
    render_loto_lab(
        "ロト7",
        prediction_file="loto7_predictions.csv",
        result_file="loto7_results.csv",
        report_file="loto7_verification_reports.csv",
        contribution_file="loto7_model_contributions.csv",
        cycle_file="loto7_research_cycles.csv",
        number_max=37,
        draw_size=7,
    )
elif lab == "AI改善":
    render_ai_department()
elif lab == "検証":
    render_verification_department()
elif lab == "保守":
    render_maintenance_department()
elif lab == "システム開発":
    render_system_department()
