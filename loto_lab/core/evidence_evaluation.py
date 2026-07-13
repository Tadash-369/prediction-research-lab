from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from prospective_evidence import (
    GAME_CONFIGS,
    PREDICTION_FINGERPRINT_COLUMN,
    calculate_prediction_fingerprint,
    get_git_commit,
    read_csv_safely,
    sha256_file,
    verify_batch,
)


EVALUATION_SCHEMA_VERSION = "1.0"
RECORD_EVALUATION_COLUMNS = [
    "game", "開催回", "抽せん日", "evidence_id", "元バッチpath", "使用モデル", "正規化モデル名", "候補番号", "予測番号",
    "本数字", "ボーナス数字", "本数字一致数", "一致した本数字", "ボーナス一致数", "一致したボーナス数字", "ボーナス込み一致数",
    "等級判定可能性", "予測生成日時", "保存日時", "入力最終開催回", "入力SHA-256", "commit hash", PREDICTION_FINGERPRINT_COLUMN,
    "Evidence Record SHA-256", "評価日時", "評価状態", "除外理由", "評価record SHA-256",
]
MODEL_SUMMARY_COLUMNS = [
    "game", "開催回", "使用モデル", "正規化モデル名", "評価口数", "本数字一致数合計", "最大一致数", "平均一致数", "ボーナス一致数",
    "3個以上一致口数", "4個以上一致口数", "開催回順位", "同率順位", "ランダムモデルとの差", "独自的中数字", "他モデルとの重複的中",
    "口数効率", "評価期間区分", "モデル集計SHA-256",
]
EVALUATION_INTEGER_COLUMNS = {"開催回", "候補番号", "本数字一致数", "ボーナス一致数", "ボーナス込み一致数", "入力最終開催回"}
SUMMARY_INTEGER_COLUMNS = {"開催回", "評価口数", "本数字一致数合計", "最大一致数", "ボーナス一致数", "3個以上一致口数", "4個以上一致口数", "開催回順位", "同率順位"}
SUMMARY_FLOAT_COLUMNS = {"平均一致数", "ランダムモデルとの差", "口数効率"}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_value(value: Any, kind: str = "string") -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if kind == "integer":
        return int(float(value))
    if kind == "float":
        return float(value)
    return str(value).strip()


def _parse_numbers(value: Any) -> list[int]:
    text = str(value or "")
    for separator in (",", " ", "/", "|"):
        text = text.replace(separator, "-")
    values = []
    for part in text.split("-"):
        try:
            values.append(int(float(part.strip())))
        except (TypeError, ValueError):
            continue
    return values


def _numbers_text(values: list[int]) -> str:
    return "-".join(f"{value:02d}" for value in sorted(values))


def result_row_sha256(game: str, row: dict[str, Any]) -> str:
    payload = {
        "game": str(game), "開催回": int(float(row.get("開催回"))), "抽せん日": str(row.get("抽せん日", "")).strip(),
        "本数字": _numbers_text(_parse_numbers(row.get("本数字"))), "ボーナス数字": _numbers_text(_parse_numbers(row.get("ボーナス数字"))),
    }
    return _sha(payload)


def _valid_result(game: str, row: dict[str, Any]) -> list[str]:
    config = GAME_CONFIGS[game]
    main = _parse_numbers(row.get("本数字"))
    bonus = _parse_numbers(row.get("ボーナス数字"))
    reasons = []
    if len(main) != config["draw_size"] or len(set(main)) != len(main) or any(value < 1 or value > config["number_max"] for value in main):
        reasons.append("invalid_main_numbers")
    expected_bonus = 1 if game == "loto6" else 2
    if len(bonus) != expected_bonus or len(set(bonus)) != len(bonus) or any(value < 1 or value > config["number_max"] for value in bonus):
        reasons.append("invalid_bonus_numbers")
    return reasons


def list_evaluation_manifests(root: Path) -> list[Path]:
    return sorted(root.rglob("evaluation_manifest.json")) if root.exists() else []


def _existing_evaluations(root: Path, batch_sha: str, evidence_ids: set[str]) -> list[dict[str, Any]]:
    matches = []
    for path in list_evaluation_manifests(root):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("source_batch_sha256") == batch_sha or evidence_ids & set(manifest.get("evidence_ids", [])):
            matches.append({"path": str(path.parent), "manifest": manifest})
    return matches


def diagnose_evaluation(batch_dir: Path, result_row: dict[str, Any] | None = None, result_game: str | None = None, evaluation_root: Path | None = None) -> dict[str, Any]:
    checked = verify_batch(batch_dir)
    manifest = checked.get("manifest", {})
    game = str(manifest.get("game", ""))
    draw_no = manifest.get("draw_no")
    diagnosis = {
        "status": "unknown", "evaluation_ready": False, "reasons": [], "game": game, "draw_no": draw_no,
        "draw_date": manifest.get("draw_date", ""), "integrity_status": checked.get("status", "unknown"),
        "source_batch_sha256": manifest.get("batch_sha256", ""), "source_fingerprint_set_sha256": manifest.get("fingerprint_set_sha256", ""),
        "existing_evaluations": 0, "batch_path": str(batch_dir), "result_row": result_row,
    }
    if checked.get("status") != "valid":
        diagnosis.update(status="invalid_integrity", reasons=checked.get("reasons", ["evidence_invalid"]))
        return diagnosis
    records = checked["records"]
    evidence_ids = set(records.get("evidence_id", []))
    existing = _existing_evaluations(evaluation_root or Path("__missing__"), diagnosis["source_batch_sha256"], evidence_ids)
    diagnosis["existing_evaluations"] = len(existing)
    diagnosis["existing_paths"] = [item["path"] for item in existing]
    if len(existing) > 1:
        diagnosis.update(status="duplicate_evaluation", reasons=["multiple_existing_evaluations"])
        return diagnosis
    if len(existing) == 1:
        diagnosis.update(status="evaluated", reasons=["evaluation_already_saved"])
        return diagnosis
    if result_row is None:
        diagnosis.update(status="awaiting_result", reasons=["result_not_available"])
        return diagnosis
    diagnosis["result_status"] = "result_available"
    if result_game != game:
        diagnosis.update(status="result_mismatch", reasons=["game_mismatch"])
        return diagnosis
    try:
        result_draw = int(float(result_row.get("開催回")))
    except (TypeError, ValueError):
        diagnosis.update(status="result_mismatch", reasons=["draw_mismatch"])
        return diagnosis
    if result_draw != int(draw_no):
        diagnosis.update(status="result_mismatch", reasons=["draw_mismatch"])
        return diagnosis
    if str(result_row.get("抽せん日", "")).strip() != str(manifest.get("draw_date", "")).strip():
        diagnosis.update(status="result_mismatch", reasons=["draw_date_mismatch"])
        return diagnosis
    result_reasons = _valid_result(game, result_row)
    if result_reasons:
        diagnosis.update(status="excluded", reasons=result_reasons)
        return diagnosis
    diagnosis.update(status="evaluation_ready", evaluation_ready=True, reasons=[], result_row_sha256=result_row_sha256(game, result_row))
    return diagnosis


def find_result_for_batch(batch_dir: Path, results: pd.DataFrame, result_game: str, evaluation_root: Path | None = None) -> dict[str, Any]:
    checked = verify_batch(batch_dir)
    manifest = checked.get("manifest", {})
    draw_no = manifest.get("draw_no")
    result_row = None
    if results is not None and not results.empty and "開催回" in results:
        draws = pd.to_numeric(results["開催回"], errors="coerce")
        matched = results[draws == int(draw_no)] if draw_no is not None else pd.DataFrame()
        if not matched.empty:
            result_row = matched.iloc[-1].to_dict()
    return diagnose_evaluation(batch_dir, result_row, result_game, evaluation_root)


def _evaluation_record_hash(row: dict[str, Any]) -> str:
    return _sha({column: _canonical_value(row.get(column, ""), "integer" if column in EVALUATION_INTEGER_COLUMNS else "string") for column in RECORD_EVALUATION_COLUMNS if column != "評価record SHA-256"})


def _summary_hash(row: dict[str, Any]) -> str:
    return _sha({column: _canonical_value(row.get(column, ""), "integer" if column in SUMMARY_INTEGER_COLUMNS else "float" if column in SUMMARY_FLOAT_COLUMNS else "string") for column in MODEL_SUMMARY_COLUMNS if column != "モデル集計SHA-256"})


def build_evaluation_preview(batch_dir: Path, diagnosis: dict[str, Any], evaluated_at: datetime | None = None) -> dict[str, Any]:
    if not diagnosis.get("evaluation_ready"):
        return {"records": pd.DataFrame(columns=RECORD_EVALUATION_COLUMNS), "model_summary": pd.DataFrame(columns=MODEL_SUMMARY_COLUMNS), "errors": diagnosis.get("reasons", [])}
    checked = verify_batch(batch_dir)
    evidence = checked["records"]
    result = diagnosis["result_row"]
    actual, bonus = set(_parse_numbers(result["本数字"])), set(_parse_numbers(result["ボーナス数字"]))
    evaluated_at = evaluated_at or datetime.now()
    rows = []
    for _, source in evidence.iterrows():
        predicted = set(_parse_numbers(source["予測番号"]))
        hits, bonus_hits = sorted(predicted & actual), sorted(predicted & bonus)
        row = {
            "game": source["game"], "開催回": source["開催回"], "抽せん日": result["抽せん日"], "evidence_id": source["evidence_id"],
            "元バッチpath": str(batch_dir), "使用モデル": source["使用モデル"], "正規化モデル名": source["正規化モデル名"], "候補番号": source["候補番号"],
            "予測番号": source["予測番号"], "本数字": _numbers_text(sorted(actual)), "ボーナス数字": _numbers_text(sorted(bonus)),
            "本数字一致数": len(hits), "一致した本数字": _numbers_text(hits), "ボーナス一致数": len(bonus_hits), "一致したボーナス数字": _numbers_text(bonus_hits),
            "ボーナス込み一致数": len(hits) + len(bonus_hits), "等級判定可能性": "判定不能", "予測生成日時": source["予測生成日時"], "保存日時": source["保存日時"],
            "入力最終開催回": source["入力データ最終開催回"], "入力SHA-256": source["入力データSHA-256"], "commit hash": source["実行コードcommit hash"],
            PREDICTION_FINGERPRINT_COLUMN: source[PREDICTION_FINGERPRINT_COLUMN], "Evidence Record SHA-256": source["レコードSHA-256"],
            "評価日時": evaluated_at.isoformat(timespec="seconds"), "評価状態": "evaluated", "除外理由": "", "評価record SHA-256": "",
        }
        row["評価record SHA-256"] = _evaluation_record_hash(row)
        rows.append(row)
    records = pd.DataFrame(rows, columns=RECORD_EVALUATION_COLUMNS)
    hit_sets = {row["正規化モデル名"]: set(_parse_numbers(row["一致した本数字"])) for row in rows}
    random_average = float(records.loc[records["正規化モデル名"] == "random_baseline", "本数字一致数"].mean()) if "random_baseline" in set(records["正規化モデル名"]) else 0.0
    summaries = []
    for (model_key, model_name), group in records.groupby(["正規化モデル名", "使用モデル"]):
        hit_union = set().union(*(hit_sets.get(model_key, set()) for _ in [0]))
        others = set().union(*(values for key, values in hit_sets.items() if key != model_key)) if len(hit_sets) > 1 else set()
        summary = {
            "game": diagnosis["game"], "開催回": diagnosis["draw_no"], "使用モデル": model_name, "正規化モデル名": model_key,
            "評価口数": len(group), "本数字一致数合計": int(group["本数字一致数"].sum()), "最大一致数": int(group["本数字一致数"].max()),
            "平均一致数": float(group["本数字一致数"].mean()), "ボーナス一致数": int(group["ボーナス一致数"].sum()),
            "3個以上一致口数": int((group["本数字一致数"] >= 3).sum()), "4個以上一致口数": int((group["本数字一致数"] >= 4).sum()),
            "開催回順位": 0, "同率順位": 0, "ランダムモデルとの差": float(group["本数字一致数"].mean()) - random_average,
            "独自的中数字": _numbers_text(sorted(hit_union - others)), "他モデルとの重複的中": _numbers_text(sorted(hit_union & others)),
            "口数効率": float(group["本数字一致数"].sum()) / len(group), "評価期間区分": "単回結果", "モデル集計SHA-256": "",
        }
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    summary_df["開催回順位"] = summary_df["平均一致数"].rank(method="dense", ascending=False).astype(int)
    counts = summary_df["平均一致数"].value_counts()
    summary_df["同率順位"] = summary_df.apply(lambda row: row["開催回順位"] if counts[row["平均一致数"]] > 1 else 0, axis=1)
    for index, row in summary_df.iterrows():
        summary_df.at[index, "モデル集計SHA-256"] = _summary_hash(row.to_dict())
    return {"records": records, "model_summary": summary_df.reindex(columns=MODEL_SUMMARY_COLUMNS), "errors": []}


def to_accuracy_ticket_metrics(records: pd.DataFrame) -> pd.DataFrame:
    if records is None or records.empty:
        return pd.DataFrame(columns=["game", "prediction_id", "draw_no", "draw_date", "model", "candidate_no", "numbers", "main_matches", "bonus_matches", "matches_with_bonus"])
    return pd.DataFrame({
        "game": records["game"], "prediction_id": records["evidence_id"], "draw_no": records["開催回"], "draw_date": records["抽せん日"],
        "model": records["使用モデル"], "candidate_no": records["候補番号"], "numbers": records["予測番号"],
        "main_matches": records["本数字一致数"], "bonus_matches": records["ボーナス一致数"], "matches_with_bonus": records["ボーナス込み一致数"],
    })


def _evaluation_batch_hash(records: pd.DataFrame, summaries: pd.DataFrame) -> str:
    values = sorted(records["評価record SHA-256"].astype(str).tolist()) + sorted(summaries["モデル集計SHA-256"].astype(str).tolist())
    return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()


def planned_evaluation_path(root: Path, game: str, draw_no: int, when: datetime | None = None) -> Path:
    return root / game / str(int(draw_no)) / (when or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")


def save_evaluation(preview: dict[str, Any], diagnosis: dict[str, Any], root: Path, result_file: Path, repo_root: Path, evaluated_at: datetime | None = None, fail_stage: str = "") -> dict[str, Any]:
    if not diagnosis.get("evaluation_ready"):
        raise ValueError("evaluation_readyではありません")
    records, summaries = preview["records"].copy(), preview["model_summary"].copy()
    if records.empty or summaries.empty:
        raise ValueError("評価プレビューが空です")
    existing = _existing_evaluations(root, diagnosis["source_batch_sha256"], set(records["evidence_id"]))
    result_hash = diagnosis["result_row_sha256"]
    commit = get_git_commit(repo_root)
    duplicate = [item for item in existing if item["manifest"].get("result_row_sha256") == result_hash and item["manifest"].get("evaluation_commit") == commit and item["manifest"].get("schema_version") == EVALUATION_SCHEMA_VERSION]
    if duplicate:
        raise FileExistsError("完全同一評価は保存済みです")
    evaluated_at = evaluated_at or datetime.now()
    final = planned_evaluation_path(root, diagnosis["game"], diagnosis["draw_no"], evaluated_at)
    temp = final.parent / f".{final.name}.tmp-{uuid.uuid4().hex}"
    temp.mkdir(parents=True, exist_ok=False)
    try:
        record_path, summary_path = temp / "record_evaluations.csv", temp / "model_summary.csv"
        records.to_csv(record_path, index=False, encoding="utf-8-sig")
        summaries.to_csv(summary_path, index=False, encoding="utf-8-sig")
        batch_hash = _evaluation_batch_hash(records, summaries)
        kind = "initial_evaluation" if not existing else "corrected_result_evaluation" if any(item["manifest"].get("result_row_sha256") != result_hash for item in existing) else "reevaluation"
        source_manifest = verify_batch(Path(diagnosis["batch_path"]))["manifest"]
        manifest = {
            "schema_version": EVALUATION_SCHEMA_VERSION, "status": "evaluated", "evaluation_kind": kind, "reevaluation_reason": "" if not existing else "input_or_code_changed",
            "game": diagnosis["game"], "draw_no": diagnosis["draw_no"], "draw_date": diagnosis["draw_date"], "evaluated_at": evaluated_at.isoformat(timespec="seconds"),
            "source_batch_path": diagnosis["batch_path"], "source_batch_sha256": diagnosis["source_batch_sha256"], "source_fingerprint_set_sha256": diagnosis["source_fingerprint_set_sha256"],
            "result_file": result_file.name, "result_file_sha256": sha256_file(result_file), "result_row_sha256": result_hash, "result_row": {key: diagnosis["result_row"].get(key, "") for key in ("開催回", "抽せん日", "本数字", "ボーナス数字")},
            "evaluation_commit": commit, "record_count": len(records), "model_count": len(summaries), "evidence_ids": records["evidence_id"].tolist(),
            "evaluation_batch_sha256": batch_hash, "record_csv_sha256": sha256_file(record_path), "model_summary_csv_sha256": sha256_file(summary_path),
        }
        manifest["manifest_sha256"] = _sha(manifest)
        (temp / "evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if fail_stage == "after_write":
            raise RuntimeError("injected failure")
        checked = verify_evaluation(temp)
        if checked["status"] != "valid":
            raise ValueError("評価バッチ検証失敗: " + " / ".join(checked["reasons"]))
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, final)
        return {"status": "saved", "path": final, "manifest": manifest}
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def verify_evaluation(path: Path) -> dict[str, Any]:
    record_path, summary_path, manifest_path = path / "record_evaluations.csv", path / "model_summary.csv", path / "evaluation_manifest.json"
    if not all(item.exists() for item in (record_path, summary_path, manifest_path)):
        return {"status": "invalid_schema", "reasons": ["required_files_missing"]}
    try:
        records = pd.read_csv(record_path, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_filter=False)
        summaries = pd.read_csv(summary_path, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_filter=False)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "corrupted", "reasons": [str(exc)]}
    reasons = []
    if list(records.columns) != RECORD_EVALUATION_COLUMNS or list(summaries.columns) != MODEL_SUMMARY_COLUMNS:
        reasons.append("schema_mismatch")
    if len(records) != manifest.get("record_count") or len(summaries) != manifest.get("model_count"):
        reasons.append("count_mismatch")
    if any(_evaluation_record_hash(row) != str(row["評価record SHA-256"]) for row in records.to_dict("records")):
        reasons.append("record_hash_mismatch")
    if any(_summary_hash(row) != str(row["モデル集計SHA-256"]) for row in summaries.to_dict("records")):
        reasons.append("summary_hash_mismatch")
    if _evaluation_batch_hash(records, summaries) != manifest.get("evaluation_batch_sha256"):
        reasons.append("batch_hash_mismatch")
    if sha256_file(record_path) != manifest.get("record_csv_sha256") or sha256_file(summary_path) != manifest.get("model_summary_csv_sha256"):
        reasons.append("file_hash_mismatch")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if _sha(payload) != manifest.get("manifest_sha256"):
        reasons.append("manifest_hash_mismatch")
    return {"status": "invalid_hash" if reasons else "valid", "reasons": reasons, "records": records, "model_summary": summaries, "manifest": manifest}
