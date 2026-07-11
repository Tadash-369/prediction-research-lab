from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from balance_weight_research import (
    append_review_history,
    build_balance_weight_research,
    build_manual_review_row,
    build_review_export_package,
    read_review_history,
    review_export_csv_bytes,
    review_export_json_bytes,
)


def _show_table(title, df):
    st.markdown(f"**{title}**")
    if df is None or df.empty:
        st.info("表示できる研究データはまだありません。")
        return
    st.dataframe(df, width="stretch", hide_index=True)


def _candidate_options(weight_research):
    candidates = weight_research.get("review_candidates", pd.DataFrame())
    if candidates is None or candidates.empty or "candidate_type" not in candidates:
        return []
    return [str(value) for value in candidates["candidate_type"].dropna().unique()]


def _export_filename(game, candidate_type, suffix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_game = str(game or "loto").replace(" ", "_")
    safe_candidate = str(candidate_type or "candidate").replace(" ", "_")
    return f"{safe_game}_{safe_candidate}_balance_weight_review_{timestamp}.{suffix}"


def render_balance_weight_research_ui(
    reports,
    draw_size=6,
    game="",
    review_history_path=None,
    key_prefix="balance_weight",
    allow_review=True,
    weight_research=None,
    show_core_tables=True,
):
    if weight_research is None:
        weight_research = build_balance_weight_research(reports, draw_size=draw_size)
    with st.expander("バランス仮説 重み改善研究・手動レビュー（読み取り専用）", expanded=False):
        st.caption("この領域は研究用の比較表示です。候補重みを本番予測へ自動反映する処理はありません。")
        st.caption("同点1位は tie_count / is_tied_top で区別し、単独1位としては扱いません。")
        if show_core_tables:
            _show_table("現行研究用ウェイト", weight_research.get("current_weights", pd.DataFrame()))
            _show_table("サブスコア貢献度研究", weight_research.get("subscore_research", pd.DataFrame()))
            _show_table("改善候補ウェイト概要", weight_research.get("candidate_summary", pd.DataFrame()))
            _show_table("Conservative / Balanced / Experimental 候補ウェイト", weight_research.get("candidate_weights", pd.DataFrame()))
            _show_table("候補ウェイトシミュレーション", weight_research.get("simulation", pd.DataFrame()))
        _show_table("回別トップ候補ランキング評価", weight_research.get("per_draw_ranking_summary", pd.DataFrame()))
        _show_table("ランキング安定性", weight_research.get("ranking_stability", pd.DataFrame()))
        _show_table("ローリング評価（過去N回で学習、次回で評価）", weight_research.get("rolling_evaluation", pd.DataFrame()))
        _show_table("本番採用前チェックリスト", weight_research.get("production_checklist", pd.DataFrame()))
        if show_core_tables:
            _show_table("AI改善・失敗分析", weight_research.get("failure_research", pd.DataFrame()))

        candidate_types = _candidate_options(weight_research)
        if not candidate_types:
            st.info("手動レビュー対象の候補重みはまだありません。")
            return weight_research

        selected = st.selectbox(
            "レビュー・エクスポート対象",
            candidate_types,
            key=f"{key_prefix}_candidate_type",
        )
        package = build_review_export_package(weight_research, game=game, candidate_type=selected)
        cols = st.columns(2)
        cols[0].download_button(
            "レビュー用JSONをダウンロード",
            data=review_export_json_bytes(package),
            file_name=_export_filename(game, selected, "json"),
            mime="application/json",
            key=f"{key_prefix}_download_json",
        )
        cols[1].download_button(
            "レビュー用CSVをダウンロード",
            data=review_export_csv_bytes(package),
            file_name=_export_filename(game, selected, "csv"),
            mime="text/csv",
            key=f"{key_prefix}_download_csv",
        )

        if review_history_path is None:
            st.info("この画面ではレビュー履歴の保存は行いません。ロト6/ロト7専用画面で手動保存できます。")
            return weight_research

        history_path = Path(review_history_path)
        history = read_review_history(history_path)
        with st.expander("保存済みレビュー履歴", expanded=False):
            _show_table("直近レビュー履歴", history.tail(20))

        if not allow_review:
            st.info("軽量スモークモードではレビュー履歴を保存しません。")
            return weight_research

        with st.form(f"{key_prefix}_manual_review_form"):
            decision = st.selectbox(
                "手動判断",
                ["hold", "candidate_for_adoption", "rejected", "unreviewed"],
                key=f"{key_prefix}_decision",
            )
            comment = st.text_area(
                "レビューコメント",
                value="",
                key=f"{key_prefix}_comment",
                help="判断理由や次回確認したい観点を記録します。本番重みは変更されません。",
            )
            reviewer = st.text_input(
                "reviewer（任意）",
                value="",
                key=f"{key_prefix}_reviewer",
            )
            submitted = st.form_submit_button("レビュー履歴へ保存")

        if submitted:
            review_row = build_manual_review_row(
                weight_research,
                game=game,
                candidate_type=selected,
                decision=decision,
                review_comment=comment,
                reviewer=reviewer,
            )
            ok, message = append_review_history(history_path, review_row)
            if ok:
                st.success(message)
            else:
                st.error(message)
    return weight_research
