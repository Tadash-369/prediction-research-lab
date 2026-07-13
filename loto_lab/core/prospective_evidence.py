from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from arl_research_engine import LOTO_MODEL_LABELS, build_model_scores, model_candidate_numbers


EVIDENCE_COLUMNS = [
    "evidence_id", "game", "開催回", "抽せん日", "予測生成日時", "保存日時", "使用モデル", "正規化モデル名",
    "モデルバージョン", "重みバージョン", "使用seed", "再現性レベル", "候補番号", "予測番号", "予測口数",
    "入力データ最終開催回", "入力データ最終抽せん日", "入力データ件数", "入力データSHA-256", "実行コードcommit hash",
    "実行環境識別情報", "予測生成経路", "予測生成オプション", "prospective判定", "抽せん前保存判定", "保存状態",
    "レコードSHA-256", "バッチSHA-256", "備考",
]
RECORD_HASH_EXCLUDED = {"レコードSHA-256", "バッチSHA-256"}
GAME_CONFIGS = {
    "loto6": {"draw_size": 6, "number_max": 43, "main_columns": [f"第{i}数字" for i in range(1, 7)], "bonus_columns": ["BONUS数字"]},
    "loto7": {"draw_size": 7, "number_max": 37, "main_columns": [f"第{i}数字" for i in range(1, 8)], "bonus_columns": ["BONUS数字1", "BONUS数字2"]},
}
EXCLUDED_MODEL_KEYS = {"anti_popular_expected_value", "balance_hypothesis_engine", "chamini_sp_god_mode"}
MODEL_VERSION = "arl_research_engine.build_model_scores:v1"


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        if not path.exists():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def get_git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        bundled = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "native" / "git" / "cmd" / "git.exe"
        try:
            result = subprocess.run([str(bundled), "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            return ""
    value = result.stdout.strip().lower()
    return value if len(value) == 40 and all(character in "0123456789abcdef" for character in value) else ""


def _int(value: Any) -> int | None:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def _date(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(str(value).strip(), errors="coerce")
    return None if pd.isna(parsed) else parsed


def _numbers_text(numbers: list[int]) -> str:
    return "-".join(f"{number:02d}" for number in sorted(numbers))


def _parse_numbers(value: Any) -> list[int]:
    text = str(value or "")
    for separator in (",", " ", "/", "|"):
        text = text.replace(separator, "-")
    return [number for part in text.split("-") if (number := _int(part)) is not None]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _record_hash(record: dict[str, Any]) -> str:
    payload = {key: record.get(key, "") for key in EVIDENCE_COLUMNS if key not in RECORD_HASH_EXCLUDED}
    return sha256_bytes(_canonical_json(payload))


def _batch_hash(records: list[dict[str, Any]]) -> str:
    return sha256_bytes("\n".join(sorted(str(record["レコードSHA-256"]) for record in records)).encode("ascii"))


def model_catalog() -> pd.DataFrame:
    rows = []
    for key, label in LOTO_MODEL_LABELS.items():
        excluded = key in EXCLUDED_MODEL_KEYS
        rows.append({
            "model_key": key, "model_name": label, "normalized_model_name": key,
            "status": "研究実行対象外" if excluded else "有効", "reason": "専用の本番補助生成経路が必要" if excluded else "共有スコア生成入口を利用可能",
            "seed_supported": key == "random_baseline", "reproducibility": "seeded" if key == "random_baseline" else ("partial" if excluded else "exact"),
        })
    return pd.DataFrame(rows)


def _history_rows(history: pd.DataFrame, official: pd.DataFrame, game: str) -> pd.DataFrame:
    config = GAME_CONFIGS[game]
    rows: dict[int, dict[str, Any]] = {}
    if history is not None and not history.empty:
        for _, source in history.iterrows():
            draw = _int(source.get("開催回"))
            date = source.get("日付")
            numbers = [_int(source.get(column)) for column in config["main_columns"]]
            bonus = [_int(source.get(column)) for column in config["bonus_columns"]]
            if draw is not None and all(number is not None for number in numbers):
                rows[draw] = {"開催回": draw, "抽せん日": date, "numbers": numbers, "bonus": [number for number in bonus if number is not None]}
    if official is not None and not official.empty:
        for _, source in official.iterrows():
            draw = _int(source.get("開催回"))
            numbers = _parse_numbers(source.get("本数字"))
            if draw is not None and len(numbers) == config["draw_size"]:
                rows[draw] = {"開催回": draw, "抽せん日": source.get("抽せん日"), "numbers": numbers, "bonus": _parse_numbers(source.get("ボーナス数字"))}
    return pd.DataFrame([rows[key] for key in sorted(rows)])


def resolve_next_draw(history: pd.DataFrame, official: pd.DataFrame, game: str) -> dict[str, Any]:
    merged = _history_rows(history, official, game)
    if merged.empty:
        return {"game": game, "target_draw": None, "draw_date": None, "status": "unknown", "reason": "入力履歴が空です", "history": merged}
    latest = merged.iloc[-1]
    latest_date = _date(latest["抽せん日"])
    if latest_date is None:
        return {"game": game, "target_draw": int(latest["開催回"]) + 1, "draw_date": None, "status": "unknown", "reason": "最新抽せん日を解決できません", "history": merged}
    if game == "loto7":
        next_date = latest_date + timedelta(days=7)
    else:
        days = 3 if latest_date.weekday() == 0 else 4 if latest_date.weekday() == 3 else None
        if days is None:
            return {"game": game, "target_draw": int(latest["開催回"]) + 1, "draw_date": None, "status": "unknown", "reason": "ロト6の最新抽せん曜日が月曜・木曜ではありません", "history": merged}
        next_date = latest_date + timedelta(days=days)
    return {"game": game, "target_draw": int(latest["開催回"]) + 1, "draw_date": next_date.strftime("%Y/%m/%d"), "status": "resolved", "reason": "", "history": merged}


def diagnose_prospective(context: dict[str, Any], now: datetime, commit_hash: str, input_hash: str, model_name: str = "catalog") -> dict[str, Any]:
    reasons = []
    history = context.get("history", pd.DataFrame())
    target = context.get("target_draw")
    draw_date = _date(context.get("draw_date"))
    latest_draw = int(history["開催回"].max()) if not history.empty else None
    latest_date = str(history.iloc[-1]["抽せん日"]) if not history.empty else ""
    if context.get("status") != "resolved" or target is None or draw_date is None:
        reasons.append(context.get("reason") or "開催情報を解決できません")
    if target is not None and latest_draw is not None and latest_draw >= int(target):
        reasons.append("入力履歴に対象回以降のデータがあります")
    if draw_date is not None and draw_date.normalize() <= pd.Timestamp(now).normalize():
        reasons.append("抽せん日当日または過去日のため保存できません")
    if not commit_hash:
        reasons.append("commit hashを取得できません")
    if not input_hash:
        reasons.append("入力データSHA-256を取得できません")
    if not model_name:
        reasons.append("モデル名が不明です")
    elif any(marker in model_name.lower() for marker in ("test", "demo", "sample", "テスト", "デモ", "サンプル")):
        reasons.append("test、demo、sample用途は正式証拠にできません")
    return {
        "prospective": not reasons, "reasons": reasons, "game": context.get("game"), "target_draw": target,
        "draw_date": context.get("draw_date"), "current_time": now.isoformat(timespec="seconds"),
        "input_latest_draw": latest_draw, "input_latest_date": latest_date, "input_count": len(history),
        "input_hash": input_hash, "commit_hash": commit_hash,
    }


def generate_preview(context: dict[str, Any], diagnosis: dict[str, Any], selected_models: list[str] | None = None, generated_at: datetime | None = None) -> dict[str, Any]:
    if not diagnosis.get("prospective"):
        return {"records": pd.DataFrame(columns=EVIDENCE_COLUMNS), "errors": diagnosis.get("reasons", []), "model_status": model_catalog()}
    generated_at = generated_at or datetime.now()
    config = GAME_CONFIGS[context["game"]]
    history = context["history"]
    number_rows = [list(value) for value in history["numbers"]]
    bonus_rows = [list(value) for value in history["bonus"]]
    catalog = model_catalog()
    if selected_models is not None:
        catalog = catalog[catalog["model_key"].isin(selected_models)]
    records, errors = [], []
    environment = f"Python {platform.python_version()} | {platform.system()} {platform.release()}"
    for _, model in catalog.iterrows():
        if model["status"] != "有効":
            continue
        key, label = str(model["model_key"]), str(model["model_name"])
        seed = int(context["target_draw"]) + config["number_max"] if key == "random_baseline" else "not_supported"
        try:
            scores = build_model_scores(number_rows, key, config["number_max"], config["draw_size"], context["target_draw"], bonus_rows)
            numbers = model_candidate_numbers(scores, config["draw_size"])
            if len(numbers) != config["draw_size"] or len(set(numbers)) != len(numbers) or any(number < 1 or number > config["number_max"] for number in numbers):
                raise ValueError("予測番号が不正です")
        except Exception as exc:
            errors.append({"model_key": key, "model_name": label, "status": "実行エラー", "reason": str(exc)})
            continue
        identity = {"game": context["game"], "draw": context["target_draw"], "model": key, "candidate": 1, "numbers": numbers, "input_hash": diagnosis["input_hash"], "commit": diagnosis["commit_hash"], "seed": seed}
        record = {
            "evidence_id": sha256_bytes(_canonical_json(identity)), "game": context["game"], "開催回": context["target_draw"], "抽せん日": context["draw_date"],
            "予測生成日時": generated_at.isoformat(timespec="seconds"), "保存日時": "pending", "使用モデル": label, "正規化モデル名": key,
            "モデルバージョン": MODEL_VERSION, "重みバージョン": "not_applicable", "使用seed": seed, "再現性レベル": model["reproducibility"],
            "候補番号": 1, "予測番号": _numbers_text(numbers), "予測口数": 1, "入力データ最終開催回": diagnosis["input_latest_draw"],
            "入力データ最終抽せん日": diagnosis["input_latest_date"], "入力データ件数": diagnosis["input_count"], "入力データSHA-256": diagnosis["input_hash"],
            "実行コードcommit hash": diagnosis["commit_hash"], "実行環境識別情報": environment, "予測生成経路": "arl_research_engine.build_model_scores",
            "予測生成オプション": json.dumps({"draw_size": config["draw_size"], "number_max": config["number_max"], "ticket_count": 1}, ensure_ascii=False, sort_keys=True),
            "prospective判定": True, "抽せん前保存判定": True, "保存状態": "preview", "レコードSHA-256": "", "バッチSHA-256": "", "備考": "本番買い目・正式予測履歴とは分離",
        }
        record["レコードSHA-256"] = _record_hash(record)
        records.append(record)
    return {"records": pd.DataFrame(records, columns=EVIDENCE_COLUMNS), "errors": errors, "model_status": catalog}


def _existing_evidence_ids(root: Path, game: str) -> set[str]:
    values = set()
    game_root = root / game / "batches"
    if not game_root.exists():
        return values
    for path in game_root.rglob("predictions.csv"):
        frame = read_csv_safely(path)
        if "evidence_id" in frame:
            values.update(frame["evidence_id"].astype(str))
    return values


def planned_batch_path(root: Path, game: str, draw_no: int, when: datetime | None = None) -> Path:
    when = when or datetime.now()
    return root / game / "batches" / str(int(draw_no)) / when.strftime("%Y%m%d_%H%M%S_%f")


def verify_batch(batch_dir: Path) -> dict[str, Any]:
    csv_path, manifest_path = batch_dir / "predictions.csv", batch_dir / "manifest.json"
    if not csv_path.exists() or not manifest_path.exists():
        return {"status": "invalid_schema", "reasons": ["predictions.csvまたはmanifest.jsonがありません"], "records": pd.DataFrame()}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame = read_csv_safely(csv_path)
    except Exception as exc:
        return {"status": "corrupted", "reasons": [str(exc)], "records": pd.DataFrame()}
    reasons = []
    if list(frame.columns) != EVIDENCE_COLUMNS:
        reasons.append("CSV schema mismatch")
    if len(frame) != int(manifest.get("record_count", -1)):
        reasons.append("record count mismatch")
    if sha256_file(csv_path) != manifest.get("file_sha256"):
        reasons.append("CSV file hash mismatch")
    manifest_payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if sha256_bytes(_canonical_json(manifest_payload)) != manifest.get("manifest_sha256"):
        reasons.append("manifest hash mismatch")
    records = frame.to_dict("records") if not frame.empty else []
    for record in records:
        if _record_hash(record) != str(record.get("レコードSHA-256", "")):
            reasons.append(f"record hash mismatch: {record.get('evidence_id', '')}")
        numbers = _parse_numbers(record.get("予測番号"))
        config = GAME_CONFIGS.get(str(record.get("game")))
        if config is None or len(numbers) != config["draw_size"] or len(numbers) != len(set(numbers)):
            reasons.append(f"invalid numbers: {record.get('evidence_id', '')}")
        if str(record.get("game")) != str(manifest.get("game")) or _int(record.get("開催回")) != _int(manifest.get("draw_no")):
            reasons.append(f"game or draw mismatch: {record.get('evidence_id', '')}")
        if str(record.get("prospective判定")).lower() not in ("true", "1") or str(record.get("抽せん前保存判定")).lower() not in ("true", "1"):
            reasons.append(f"not prospective: {record.get('evidence_id', '')}")
        if not str(record.get("使用モデル", "")).strip() or not str(record.get("保存日時", "")).strip():
            reasons.append(f"required value missing: {record.get('evidence_id', '')}")
    if records and _batch_hash(records) != manifest.get("batch_sha256"):
        reasons.append("batch hash mismatch")
    return {"status": "invalid_hash" if reasons else "valid", "reasons": reasons, "records": frame, "manifest": manifest}


def save_batch(preview: pd.DataFrame, diagnosis: dict[str, Any], root: Path, saved_at: datetime | None = None, fail_stage: str = "") -> dict[str, Any]:
    if not diagnosis.get("prospective"):
        raise ValueError("prospective条件を満たしていません")
    if preview is None or preview.empty:
        raise ValueError("保存対象がありません")
    game = str(diagnosis["game"])
    if preview["evidence_id"].astype(str).duplicated().any():
        raise ValueError("バッチ内でevidence_idが重複しています")
    existing = _existing_evidence_ids(root, game)
    duplicate = set(preview["evidence_id"].astype(str)) & existing
    if duplicate:
        raise FileExistsError(f"完全同一証拠は保存済みです: {sorted(duplicate)[0]}")
    saved_at = saved_at or datetime.now()
    if _date(diagnosis["draw_date"]).normalize() <= pd.Timestamp(saved_at).normalize():
        raise ValueError("保存時点で抽せん日当日または過去日です")
    final_dir = planned_batch_path(root, game, int(diagnosis["target_draw"]), saved_at)
    temp_dir = final_dir.parent / f".{final_dir.name}.tmp-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        records = preview.copy()
        records["保存日時"] = saved_at.isoformat(timespec="seconds")
        records["保存状態"] = "saved"
        for index, record in records.iterrows():
            records.at[index, "レコードSHA-256"] = _record_hash(record.to_dict())
        payload = records.to_dict("records")
        batch_hash = _batch_hash(payload)
        records["バッチSHA-256"] = batch_hash
        csv_path = temp_dir / "predictions.csv"
        records.to_csv(csv_path, index=False, encoding="utf-8-sig")
        manifest = {"schema_version": "1.0", "game": game, "draw_no": int(diagnosis["target_draw"]), "draw_date": diagnosis["draw_date"], "created_at": saved_at.isoformat(timespec="seconds"), "record_count": len(records), "batch_sha256": batch_hash, "file_sha256": sha256_file(csv_path), "commit_hash": diagnosis["commit_hash"], "input_data_sha256": diagnosis["input_hash"]}
        manifest["manifest_sha256"] = sha256_bytes(_canonical_json(manifest))
        (temp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if fail_stage == "after_write":
            raise RuntimeError("injected failure")
        checked = verify_batch(temp_dir)
        if checked["status"] != "valid":
            raise ValueError("一時バッチ検証失敗: " + " / ".join(checked["reasons"]))
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            raise FileExistsError(f"同一バッチ保存先が存在します: {final_dir}")
        os.replace(temp_dir, final_dir)
        return {"status": "saved", "path": final_dir, "record_count": len(records), "batch_sha256": batch_hash}
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def scan_evidence(root: Path, results_by_game: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    rows = []
    results_by_game = results_by_game or {}
    for game in GAME_CONFIGS:
        paths = list((root / game / "batches").rglob("manifest.json")) if (root / game / "batches").exists() else []
        result_draws = {_int(value) for value in results_by_game.get(game, pd.DataFrame()).get("開催回", [])}
        seen = set()
        for manifest_path in paths:
            checked = verify_batch(manifest_path.parent)
            frame = checked.get("records", pd.DataFrame())
            for _, record in frame.iterrows():
                evidence_id = str(record.get("evidence_id", ""))
                duplicate = evidence_id in seen
                seen.add(evidence_id)
                integrity = "duplicate" if duplicate else checked["status"]
                result_available = _int(record.get("開催回")) in result_draws
                evaluation = "evaluation_ready" if integrity == "valid" and result_available else "awaiting_result" if integrity == "valid" else "excluded"
                rows.append({"game": game, "evidence_id": evidence_id, "開催回": record.get("開催回", ""), "使用モデル": record.get("使用モデル", ""), "integrity_status": integrity, "result_status": "result_available" if result_available else "awaiting_result", "evaluation_status": evaluation, "batch_path": str(manifest_path.parent)})
    return pd.DataFrame(rows, columns=["game", "evidence_id", "開催回", "使用モデル", "integrity_status", "result_status", "evaluation_status", "batch_path"])


def load_game_context(data_dir: Path, repo_root: Path, game: str, now: datetime | None = None) -> dict[str, Any]:
    history_path = data_dir / ("loto6.csv" if game == "loto6" else "loto7.csv")
    official_path = data_dir / ("results.csv" if game == "loto6" else "loto7_results.csv")
    history, official = read_csv_safely(history_path), read_csv_safely(official_path)
    context = resolve_next_draw(history, official, game)
    commit = get_git_commit(repo_root)
    input_hash = source_files_sha256([history_path, official_path])
    diagnosis = diagnose_prospective(context, now or datetime.now(), commit, input_hash)
    return {"context": context, "diagnosis": diagnosis, "history_path": history_path, "official_path": official_path}
