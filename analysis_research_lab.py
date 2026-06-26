from pathlib import Path

import pandas as pd
import streamlit as st

from arl_research_engine import (
    CONTRIBUTION_COLUMNS,
    FUTURE_LOTO_MODEL_LABELS,
    LOTO_MODEL_LABELS,
    PROJECT_ENGLISH_NAME,
    PROJECT_JAPANESE_NAME,
    PROJECT_SHORT_NAME,
    RESEARCH_CYCLE_COLUMNS,
    VIDEO_HYPOTHESIS_COLUMNS,
    add_verification_metrics,
    build_ai_improvement_summary,
    build_condition_success_table,
    build_contribution_ranking,
    build_model_dashboard,
    build_research_flow_table,
)
from prl_maintenance import run_maintenance


BASE_DIR = Path(__file__).resolve().parent


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


def render_home():
    loto6_predictions = read_csv(BASE_DIR / "predictions.csv")
    loto6_reports = add_verification_metrics(read_csv(BASE_DIR / "verification_reports.csv"), draw_size=6)
    loto7_predictions = read_csv(BASE_DIR / "loto7_predictions.csv")
    loto7_reports = add_verification_metrics(read_csv(BASE_DIR / "loto7_verification_reports.csv"), draw_size=7)
    docs = list((BASE_DIR / "分析研究所" / "docs").glob("*.md")) if (BASE_DIR / "分析研究所" / "docs").exists() else []
    backups = list((BASE_DIR / "分析研究所" / "data" / "backups").glob("*.csv")) if (BASE_DIR / "分析研究所" / "data" / "backups").exists() else []

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
            {"部門": "投資", "予想保存数": "-", "検証数": "-", "平均一致数": "-", "平均期待値": "-", "状態": "基礎機能実装中"},
            {"部門": "AI改善", "予想保存数": "-", "検証数": len(loto6_reports) + len(loto7_reports), "平均一致数": "-", "平均期待値": "-", "状態": "運用中"},
            {"部門": "保守", "予想保存数": "-", "検証数": "-", "平均一致数": "-", "平均期待値": "-", "状態": "同期・補完対応"},
        ]
    )
    for column in ["予想保存数", "検証数", "平均一致数", "平均期待値"]:
        summary[column] = summary[column].astype(str)
    st.markdown("**研究所サマリー**")
    st.dataframe(summary, width="stretch", hide_index=True)

    next_actions = pd.DataFrame(
        [
            {"優先": 1, "次の作業": "ロト6またはロト7で最新当選結果を登録し、検証レポートを更新する"},
            {"優先": 2, "次の作業": "保守部門でVer1.0補完とフォルダ同期を実行する"},
            {"優先": 3, "次の作業": "AI改善部門で成功要因、失敗要因、改善案、次回仮説を確認する"},
            {"優先": 4, "次の作業": "モデル別成績と条件別成功率を見て、次回予想の仮説を決める"},
        ]
    )
    st.markdown("**次にやること**")
    st.dataframe(next_actions, width="stretch", hide_index=True)


def render_model_catalog():
    model_df = pd.DataFrame(
        [{"番号": index, "区分": "現行", "モデル": label} for index, label in enumerate(LOTO_MODEL_LABELS.values(), start=1)]
        + [
            {"番号": index, "区分": "将来追加予定", "モデル": label}
            for index, label in enumerate(FUTURE_LOTO_MODEL_LABELS.values(), start=len(LOTO_MODEL_LABELS) + 1)
        ]
    )
    st.dataframe(model_df, width="stretch", hide_index=True)


def render_loto_lab(name, prediction_file, result_file, report_file, contribution_file, cycle_file, number_max, draw_size):
    predictions = read_csv(BASE_DIR / prediction_file)
    official = read_csv(BASE_DIR / result_file)
    reports = add_verification_metrics(read_csv(BASE_DIR / report_file), draw_size=draw_size)
    contributions = read_csv(BASE_DIR / contribution_file, CONTRIBUTION_COLUMNS)
    cycles = read_csv(BASE_DIR / cycle_file, RESEARCH_CYCLE_COLUMNS)

    cols = st.columns(4)
    cols[0].metric("予想履歴", len(predictions))
    cols[1].metric("公式結果ログ", len(official))
    cols[2].metric("検証レポート", len(reports))
    cols[3].metric("貢献度ログ", len(contributions))

    st.markdown("**研究フロー**")
    st.dataframe(build_research_flow_table(), width="stretch", hide_index=True)

    tabs = st.tabs(["研究サイクル", "モデル別成績", "モデル貢献度", "条件別成功率", "AI改善レポート", "動画仮説"])
    with tabs[0]:
        if cycles.empty:
            st.info(f"{name}の研究サイクル履歴は、検証レポート作成後に保存されます。")
        else:
            st.dataframe(cycles.sort_values(["開催回", "予想ID"], ascending=[False, True]), width="stretch", hide_index=True)
    with tabs[1]:
        dashboard = build_model_dashboard(reports, draw_size=draw_size)
        if dashboard.empty:
            st.info(f"{name}のモデル別成績は、予想と結果の照合後に表示されます。")
        else:
            st.dataframe(dashboard, width="stretch", hide_index=True)
    with tabs[2]:
        ranking = build_contribution_ranking(contributions)
        if ranking.empty:
            st.info(f"{name}の貢献度ランキングは、的中数字の要因分析後に表示されます。")
        else:
            st.dataframe(ranking, width="stretch", hide_index=True)
    with tabs[3]:
        condition_df = build_condition_success_table(reports, number_max=number_max)
        if condition_df.empty:
            st.info(f"{name}の条件別成功率は、検証レポート作成後に表示されます。")
        else:
            st.dataframe(condition_df, width="stretch", hide_index=True)
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
        video_logs = read_csv(BASE_DIR / "video_hypotheses.csv", VIDEO_HYPOTHESIS_COLUMNS)
        if video_logs.empty:
            st.info("動画仮説ログはまだありません。")
        else:
            st.dataframe(video_logs.sort_values("保存日時", ascending=False), width="stretch", hide_index=True)


def render_investment_lab():
    investment_dir = BASE_DIR / "分析研究所" / "投資"
    portfolio_columns = ["銘柄コード", "銘柄名", "分類", "保有数", "取得単価", "現在値", "配当単価", "メモ"]
    dividend_columns = ["日付", "銘柄コード", "銘柄名", "受取配当", "税引後", "メモ"]
    watch_columns = ["対象", "分類", "現在値", "移動平均", "RSI", "MACD", "配当利回り", "PER", "PBR", "EPS成長率", "判定"]
    portfolio_path = investment_dir / "portfolio.csv"
    dividend_path = investment_dir / "dividends.csv"
    watch_path = investment_dir / "watchlist.csv"

    portfolio = read_csv(portfolio_path, portfolio_columns)
    dividends = read_csv(dividend_path, dividend_columns)
    watchlist = read_csv(watch_path, watch_columns)
    if portfolio.empty:
        portfolio = pd.DataFrame(
            [
                {"銘柄コード": "1321", "銘柄名": "日経225 ETF", "分類": "ETF", "保有数": 0, "取得単価": 0, "現在値": 0, "配当単価": 0, "メモ": ""},
                {"銘柄コード": "SPY", "銘柄名": "S&P500 ETF", "分類": "米国ETF", "保有数": 0, "取得単価": 0, "現在値": 0, "配当単価": 0, "メモ": ""},
            ],
            columns=portfolio_columns,
        )
    if watchlist.empty:
        watchlist = pd.DataFrame(
            [
                {"対象": "日経平均", "分類": "指数", "現在値": 0, "移動平均": 0, "RSI": 0, "MACD": 0, "配当利回り": 0, "PER": 0, "PBR": 0, "EPS成長率": 0, "判定": "観察"},
                {"対象": "ドル円", "分類": "為替", "現在値": 0, "移動平均": 0, "RSI": 0, "MACD": 0, "配当利回り": 0, "PER": 0, "PBR": 0, "EPS成長率": 0, "判定": "観察"},
            ],
            columns=watch_columns,
        )

    portfolio_calc = portfolio.copy()
    for column in ["保有数", "取得単価", "現在値", "配当単価"]:
        portfolio_calc[column] = pd.to_numeric(portfolio_calc[column], errors="coerce").fillna(0)
    portfolio_calc["評価額"] = portfolio_calc["保有数"] * portfolio_calc["現在値"]
    portfolio_calc["取得額"] = portfolio_calc["保有数"] * portfolio_calc["取得単価"]
    portfolio_calc["損益"] = portfolio_calc["評価額"] - portfolio_calc["取得額"]
    portfolio_calc["年間配当予想"] = portfolio_calc["保有数"] * portfolio_calc["配当単価"]
    total_value = float(portfolio_calc["評価額"].sum())
    total_profit = float(portfolio_calc["損益"].sum())
    total_dividend = float(portfolio_calc["年間配当予想"].sum())
    yield_rate = round(total_dividend / total_value * 100, 2) if total_value else 0.0

    cols = st.columns(4)
    cols[0].metric("評価額", f"{round(total_value):,}")
    cols[1].metric("損益", f"{round(total_profit):,}")
    cols[2].metric("年間配当予想", f"{round(total_dividend):,}")
    cols[3].metric("配当利回り", f"{yield_rate}%")

    targets = pd.DataFrame(
        [
            {"対象": "日経平均", "分析項目": "移動平均 / RSI / MACD / ボリンジャーバンド / 出来高"},
            {"対象": "TOPIX", "分析項目": "移動平均 / RSI / MACD / ボリンジャーバンド / 出来高"},
            {"対象": "S&P500", "分析項目": "移動平均 / RSI / MACD / ボリンジャーバンド / 出来高"},
            {"対象": "NASDAQ", "分析項目": "移動平均 / RSI / MACD / ボリンジャーバンド / 出来高"},
            {"対象": "ドル円", "分析項目": "移動平均 / RSI / MACD / ボリンジャーバンド / 出来高"},
            {"対象": "高配当株", "分析項目": "配当利回り / PER / PBR / EPS成長率 / 出来高"},
            {"対象": "ETF", "分析項目": "移動平均 / RSI / MACD / 配当利回り / 出来高"},
            {"対象": "米国ETF", "分析項目": "移動平均 / RSI / MACD / 配当利回り / 出来高"},
            {"対象": "優待株", "分析項目": "配当利回り / PER / PBR / EPS成長率"},
        ]
    )
    tabs = st.tabs(["ポートフォリオ", "配当管理", "売買候補", "分析対象"])
    with tabs[0]:
        edited = st.data_editor(portfolio, num_rows="dynamic", width="stretch", hide_index=True, key="portfolio_editor")
        if st.button("ポートフォリオを保存"):
            save_csv(edited, portfolio_path, portfolio_columns)
            st.success("ポートフォリオを保存しました。")
        st.markdown("**集計**")
        st.dataframe(portfolio_calc, width="stretch", hide_index=True)
    with tabs[1]:
        edited_dividends = st.data_editor(dividends, num_rows="dynamic", width="stretch", hide_index=True, key="dividend_editor")
        if st.button("配当履歴を保存"):
            save_csv(edited_dividends, dividend_path, dividend_columns)
            st.success("配当履歴を保存しました。")
    with tabs[2]:
        edited_watchlist = st.data_editor(watchlist, num_rows="dynamic", width="stretch", hide_index=True, key="watchlist_editor")
        if st.button("売買候補を保存"):
            save_csv(edited_watchlist, watch_path, watch_columns)
            st.success("売買候補を保存しました。")
    with tabs[3]:
        st.dataframe(targets, width="stretch", hide_index=True)


def render_ai_department():
    loto6_reports = add_verification_metrics(read_csv(BASE_DIR / "verification_reports.csv"), draw_size=6)
    loto7_reports = add_verification_metrics(read_csv(BASE_DIR / "loto7_verification_reports.csv"), draw_size=7)
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


def render_verification_department():
    rows = []
    for name, report_file, draw_size in [("ロト6", "verification_reports.csv", 6), ("ロト7", "loto7_verification_reports.csv", 7)]:
        reports = add_verification_metrics(read_csv(BASE_DIR / report_file), draw_size=draw_size)
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
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_hospital_system():
    hospital_dir = BASE_DIR / "分析研究所" / "病院業務改善"
    staff_columns = ["職員ID", "氏名", "職種", "常勤換算", "夜勤可否", "委員会", "メモ"]
    shift_columns = ["日付", "勤務帯", "必要人数", "配置人数", "不足人数", "メモ"]
    committee_columns = ["委員会名", "担当者", "開催頻度", "負担度", "改善メモ"]
    workload_columns = ["業務", "担当部署", "件数", "所要分", "負担度", "改善案"]
    staff_path = hospital_dir / "staff.csv"
    shift_path = hospital_dir / "shift_requirements.csv"
    committee_path = hospital_dir / "committees.csv"
    workload_path = hospital_dir / "workload.csv"

    staff = read_csv(staff_path, staff_columns)
    shifts = read_csv(shift_path, shift_columns)
    committees = read_csv(committee_path, committee_columns)
    workload = read_csv(workload_path, workload_columns)
    if staff.empty:
        staff = pd.DataFrame([{"職員ID": "S001", "氏名": "", "職種": "看護師", "常勤換算": 1.0, "夜勤可否": "可", "委員会": "", "メモ": ""}], columns=staff_columns)
    if shifts.empty:
        shifts = pd.DataFrame([{"日付": "", "勤務帯": "日勤", "必要人数": 0, "配置人数": 0, "不足人数": 0, "メモ": ""}], columns=shift_columns)
    if committees.empty:
        committees = pd.DataFrame([{"委員会名": "", "担当者": "", "開催頻度": "月1回", "負担度": 0, "改善メモ": ""}], columns=committee_columns)
    if workload.empty:
        workload = pd.DataFrame([{"業務": "", "担当部署": "", "件数": 0, "所要分": 0, "負担度": 0, "改善案": ""}], columns=workload_columns)

    staff_count = len(staff)
    shortage_total = numeric_column(shifts, "不足人数").sum() if not shifts.empty else 0
    workload_score = numeric_column(workload, "負担度").mean() if not workload.empty else 0
    cols = st.columns(3)
    cols[0].metric("登録職員", staff_count)
    cols[1].metric("不足人数合計", int(shortage_total))
    cols[2].metric("平均負担度", round(float(workload_score), 2))

    tabs = st.tabs(["職員", "勤務表", "委員会", "業務負担", "機能一覧"])
    with tabs[0]:
        edited_staff = st.data_editor(staff, num_rows="dynamic", width="stretch", hide_index=True, key="hospital_staff_editor")
        if st.button("職員データを保存"):
            save_csv(edited_staff, staff_path, staff_columns)
            st.success("職員データを保存しました。")
    with tabs[1]:
        edited_shifts = st.data_editor(shifts, num_rows="dynamic", width="stretch", hide_index=True, key="hospital_shift_editor")
        if st.button("勤務表条件を保存"):
            save_csv(edited_shifts, shift_path, shift_columns)
            st.success("勤務表条件を保存しました。")
    with tabs[2]:
        edited_committees = st.data_editor(committees, num_rows="dynamic", width="stretch", hide_index=True, key="hospital_committee_editor")
        if st.button("委員会データを保存"):
            save_csv(edited_committees, committee_path, committee_columns)
            st.success("委員会データを保存しました。")
    with tabs[3]:
        edited_workload = st.data_editor(workload, num_rows="dynamic", width="stretch", hide_index=True, key="hospital_workload_editor")
        if st.button("業務負担データを保存"):
            save_csv(edited_workload, workload_path, workload_columns)
            st.success("業務負担データを保存しました。")
    with tabs[4]:
        functions = pd.DataFrame(
            [
                {"機能": "勤務表自動作成", "状態": "基礎データ管理まで実装"},
                {"機能": "人員配置最適化", "状態": "不足人数集計まで実装"},
                {"機能": "委員会管理", "状態": "基礎データ管理まで実装"},
                {"機能": "業務負担分析", "状態": "負担度集計まで実装"},
            ]
        )
        st.dataframe(functions, width="stretch", hide_index=True)

def render_system_department():
    apps = pd.DataFrame(
        [
            {"アプリ": "ロト研究所アプリ", "機能": "CSV自動読込 / 当選履歴管理 / 分析実行 / 予想生成 / 結果保存 / モデル別成績表 / AI改善レポート", "状態": "運用中"},
            {"アプリ": "投資研究所アプリ", "機能": "ポートフォリオ管理 / 配当管理 / ETF分析 / 売買候補抽出", "状態": "基礎機能実装"},
            {"アプリ": "病院業務改善システム", "機能": "勤務表条件 / 人員配置集計 / 委員会管理 / 業務負担分析", "状態": "基礎機能実装"},
        ]
    )
    st.dataframe(apps, width="stretch", hide_index=True)


def render_maintenance_department():
    st.info("既存CSVをVer1.0形式へ補完します。実行時は自動でバックアップを作成します。")
    cols = st.columns(2)
    if cols[0].button("補完内容を確認"):
        summary = run_maintenance(apply_changes=False)
        st.dataframe(summary, width="stretch", hide_index=True)
    if cols[1].button("Ver1.0補完を実行"):
        summary = run_maintenance(apply_changes=True)
        st.success("補完を実行しました。")
        st.dataframe(summary, width="stretch", hide_index=True)


def render_launch_guide():
    guide = pd.DataFrame(
        [
            {"画面": "分析研究所トップ", "起動コマンド": ".\\.venv\\Scripts\\streamlit.exe run analysis_research_lab.py --server.port 8501"},
            {"画面": "ロト6分析", "起動コマンド": ".\\.venv\\Scripts\\streamlit.exe run loto6_streamlit_app.py --server.port 8501"},
            {"画面": "ロト7分析", "起動コマンド": ".\\.venv\\Scripts\\streamlit.exe run loto7_streamlit_app.py --server.port 8501"},
        ]
    )
    st.dataframe(guide, width="stretch", hide_index=True)


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
    st.dataframe(roadmap, width="stretch", hide_index=True)


st.set_page_config(page_title=f"{PROJECT_JAPANESE_NAME} | {PROJECT_SHORT_NAME}", layout="wide")
st.title(PROJECT_JAPANESE_NAME)
st.caption(f"{PROJECT_ENGLISH_NAME}（{PROJECT_SHORT_NAME}）")
st.info("目的は当選や利益そのものではなく、予測手法の研究・検証・改善を継続し、予測精度を向上させることです。")

lab = st.radio("部門", ["ホーム", "ロト6", "ロト7", "投資", "AI改善", "検証", "保守", "システム開発", "病院業務改善"], horizontal=True)

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
elif lab == "投資":
    render_investment_lab()
elif lab == "AI改善":
    render_ai_department()
elif lab == "検証":
    render_verification_department()
elif lab == "保守":
    render_maintenance_department()
elif lab == "システム開発":
    render_system_department()
else:
    render_hospital_system()
