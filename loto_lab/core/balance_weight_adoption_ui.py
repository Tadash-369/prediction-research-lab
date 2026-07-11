from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from balance_weight_adoption import (
    adoption_overview,
    adoption_package_json_bytes,
    append_approval_history,
    build_adoption_package,
    build_approval_row,
    candidate_from_review_row,
    compare_candidate_to_current,
    configuration_preview,
    dry_run_report_csv_bytes,
    latest_review_decisions,
    load_review_history,
    patch_preview_text,
    read_approval_history,
    readiness_score,
    rollback_plan,
    run_adoption_dry_run,
    selectable_reviews,
    validate_candidate,
    weight_diff_summary,
)


def _show_table(title, df):
    st.markdown(f"**{title}**")
    if df is None or df.empty:
        st.info("表示できるデータはまだありません。")
        return
    st.dataframe(df, width="stretch", hide_index=True)


def _filename(game, candidate_type, suffix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{game}_{candidate_type}_adoption_dry_run_{timestamp}.{suffix}"


def render_adoption_overview(review_history_path, approval_history_path=None, games=None):
    review_history = load_review_history(review_history_path)
    approval_history = read_approval_history(approval_history_path)
    rows = []
    for game in games or ["loto6", "loto7"]:
        overview = adoption_overview(review_history, approval_history, game=game)
        if not overview.empty:
            rows.extend(overview.to_dict(orient="records"))
    with st.expander("採用準備dry-run 概要（読み取り専用）", expanded=False):
        st.caption("ここでは本番重みの変更操作は行いません。レビュー履歴と承認履歴を読み取り専用で集計します。")
        _show_table("ロト別の採用準備状況", pd.DataFrame(rows))


def render_adoption_dry_run_ui(
    reports,
    game,
    review_history_path,
    approval_history_path,
    key_prefix,
    allow_write=True,
    light_smoke=False,
):
    with st.expander("採用準備dry-run（本番反映なし）", expanded=False):
        st.caption("Ver1.11のdry-runは、本番重み・予測CSV・検証CSV・レビュー履歴を変更しない読み取り専用評価です。")
        st.caption("承認履歴は下部の明示フォーム送信時のみ保存されます。承認しても本番重みは変更されません。")
        if light_smoke:
            st.info("軽量スモークモードでは採用準備dry-runの重い計算と履歴保存を省略します。")
            return

        review_history = load_review_history(review_history_path)
        approval_history = read_approval_history(approval_history_path)
        latest = latest_review_decisions(review_history, game=game)
        selectable = selectable_reviews(review_history, game=game)
        _show_table("最新レビュー判定", latest[latest["is_latest"]].tail(20) if not latest.empty else latest)

        if selectable.empty:
            st.info("dry-run対象のレビュー済み候補はまだありません。先にVer1.10の手動レビューを保存してください。")
            return

        labels = []
        rows = []
        for _, row in selectable.iterrows():
            label = f"{row.get('decision_label', row.get('decision', ''))} | {row.get('candidate_type', '')} | {row.get('reviewed_at', '')} | {str(row.get('candidate_weights_hash', ''))[:12]}"
            labels.append(label)
            rows.append(row)
        selected_label = st.selectbox("dry-run対象候補", labels, key=f"{key_prefix}_candidate")
        selected_row = dict(rows[labels.index(selected_label)])
        candidate = candidate_from_review_row(selected_row)

        if candidate.get("decision") == "rejected":
            st.warning("却下候補が選択されています。研究比較はできますが、採用準備としては注意扱いです。")
        elif candidate.get("decision") == "candidate_for_adoption":
            st.success("採用候補としてレビュー済みです。ただし本番反映ではありません。")
        else:
            st.info("採用候補以外のレビュー判定です。dry-runは参考比較として扱います。")

        diff_df = compare_candidate_to_current(candidate.get("candidate_weights"))
        diff_summary = weight_diff_summary(diff_df)
        _show_table("Current / Candidate 重み差分", diff_df)
        _show_table("重み差分サマリー", diff_summary)

        safety_status, safety_checks = validate_candidate(candidate, game=game)
        _show_table(f"dry-run安全検証: {safety_status}", safety_checks)
        if safety_status == "blocked":
            st.error("安全検証がblockedのため、dry-run計算は実行しません。")
            return

        dry_run = run_adoption_dry_run(reports, candidate, game=game)
        _show_table("dry-run比較サマリー", dry_run.get("summary", pd.DataFrame()))
        _show_table("回別dry-run比較", dry_run.get("per_draw", pd.DataFrame()))
        _show_table("ローリングdry-run比較", dry_run.get("rolling", pd.DataFrame()))
        _show_table("レビュー後ドリフト確認", dry_run.get("drift", pd.DataFrame()))

        candidate_approvals = approval_history[
            (approval_history["game"].astype(str) == str(game))
            & (approval_history["candidate_weights_hash"].astype(str) == str(candidate.get("candidate_weights_hash", "")))
        ] if not approval_history.empty else pd.DataFrame()
        readiness = readiness_score(
            candidate,
            dry_run.get("safety_checks", pd.DataFrame()),
            dry_run.get("summary", pd.DataFrame()),
            dry_run.get("rolling", pd.DataFrame()),
            dry_run.get("drift", pd.DataFrame()),
            approval_history=candidate_approvals,
        )
        _show_table("採用準備スコア", readiness)

        config_preview = configuration_preview(candidate)
        _show_table("本番設定変更プレビュー（実ファイル非変更）", config_preview)
        patch_text = patch_preview_text(candidate)
        st.markdown("**dry-runパッチプレビュー（適用しません）**")
        st.code(patch_text or "差分を生成できませんでした。", language="diff")
        rollback = rollback_plan(candidate)
        st.markdown("**ロールバック案（説明のみ、自動実行なし）**")
        st.json(rollback)

        package = build_adoption_package(
            game,
            candidate,
            dry_run,
            diff_df,
            diff_summary,
            readiness,
            config_preview,
            rollback,
            approvals=candidate_approvals,
        )
        cols = st.columns(2)
        cols[0].download_button(
            "採用準備JSONをダウンロード",
            data=adoption_package_json_bytes(package),
            file_name=_filename(game, candidate.get("candidate_type", "candidate"), "json"),
            mime="application/json",
            key=f"{key_prefix}_package_json",
        )
        cols[1].download_button(
            "dry-run CSVレポートをダウンロード",
            data=dry_run_report_csv_bytes(package),
            file_name=_filename(game, candidate.get("candidate_type", "candidate"), "csv"),
            mime="text/csv",
            key=f"{key_prefix}_package_csv",
        )

        with st.expander("承認履歴", expanded=False):
            _show_table("対象候補の承認履歴", candidate_approvals.tail(20))

        if not allow_write:
            st.info("このモードでは承認履歴を保存しません。")
            return

        with st.form(f"{key_prefix}_approval_form"):
            approval_status = st.selectbox(
                "承認状態",
                ["unapproved", "needs_rework", "dry_run_approved", "approved_for_manual_preparation"],
                key=f"{key_prefix}_approval_status",
            )
            approval_comment = st.text_area("承認コメント", value="", key=f"{key_prefix}_approval_comment")
            approver = st.text_input("approver（任意）", value="", key=f"{key_prefix}_approver")
            submitted = st.form_submit_button("承認履歴へ保存")
        if submitted:
            row = build_approval_row(
                game,
                candidate,
                approval_status,
                approval_comment,
                readiness,
                dry_run.get("summary", pd.DataFrame()),
                dry_run.get("drift", pd.DataFrame()),
                dry_run.get("safety_checks", pd.DataFrame()),
                approver=approver,
            )
            ok, message = append_approval_history(approval_history_path, row)
            if ok:
                st.success(message)
            else:
                st.error(message)
