from datetime import datetime
from fnmatch import fnmatchcase
from pathlib import Path

import pandas as pd


RUNTIME_STATUS_OK = "正常"
RUNTIME_STATUS_MISSING = "未作成"
RUNTIME_STATUS_CREATED = "自動生成済み"
RUNTIME_STATUS_BROKEN = "破損"
RUNTIME_STATUS_FALLBACK = "フォールバック中"
RUNTIME_SETTING_COLUMNS = ["設定名", "モデルキー", "モデル名", "根拠", "更新日時"]
RUNTIME_SETTING_HISTORY_COLUMNS = [
    "履歴ID",
    "ゲーム",
    "設定名",
    "旧モデルキー",
    "旧モデル名",
    "新モデルキー",
    "新モデル名",
    "変更理由",
    "変更元",
    "変更日時",
    "保存結果",
    "エラー内容",
]
RUNTIME_HISTORY_GAMES = {"loto6", "loto7"}
RUNTIME_HISTORY_CHANGE_SOURCES = {
    "pre_prediction_research",
    "backtest",
    "manual_ui",
    "initialization",
    "migration",
    "unknown",
}
RUNTIME_HISTORY_SAVE_RESULTS = {"success", "rejected", "failed", "no_change"}
RUNTIME_DIAGNOSIS_STATUS_LABELS = {
    "healthy": "正常",
    "warning": "注意",
    "fallback": "フォールバック中",
    "missing": "未作成",
    "corrupted": "破損",
    "error": "読込エラー",
}
RUNTIME_HISTORY_STATUS_LABELS = {
    "healthy": "正常",
    "warning": "注意",
    "missing": "履歴なし",
    "corrupted": "破損",
    "error": "読込エラー",
}


def runtime_columns(columns):
    return list(dict.fromkeys(list(columns or []) + RUNTIME_SETTING_COLUMNS))


def read_runtime_csv(path, columns):
    columns = runtime_columns(columns)
    if not Path(path).exists():
        return pd.DataFrame(columns=columns)
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _empty_diagnostic(path, setting_name):
    return {
        "path": str(path),
        "status": RUNTIME_STATUS_MISSING,
        "setting_name": setting_name,
        "model_key": "",
        "model_name": "",
        "reason": "",
        "git_managed": "いいえ",
        "runtime_dir_exists": Path(path).parent.exists(),
        "runtime_file_exists": Path(path).exists(),
        "created": False,
        "errors": [],
    }


def _missing_columns(df, columns):
    return [column for column in columns if column not in df.columns]


def _has_corrupted_text(value):
    if value is None:
        return False
    try:
        if pd.isna(value) == True:
            return False
    except Exception:
        pass
    return "?" in str(value)


def _setting_row(setting_name, model_key, model_name, reason, updated_at):
    return {
        "設定名": setting_name,
        "モデルキー": model_key,
        "モデル名": model_name,
        "根拠": reason,
        "更新日時": updated_at,
    }


def ensure_runtime_csv(runtime_path, template_path, columns):
    columns = runtime_columns(columns)
    runtime_path = Path(runtime_path)
    template_path = Path(template_path) if template_path else None
    diagnostic = _empty_diagnostic(runtime_path, "")
    try:
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic["runtime_dir_exists"] = True
        if runtime_path.exists():
            diagnostic["runtime_file_exists"] = True
            diagnostic["status"] = RUNTIME_STATUS_OK
            return diagnostic

        if template_path and template_path.exists():
            source_df = read_runtime_csv(template_path, columns)
            source_df = source_df.reindex(columns=columns)
            write_runtime_csv_atomic(source_df, runtime_path, columns)
            diagnostic["status"] = RUNTIME_STATUS_CREATED
        else:
            write_runtime_csv_atomic(pd.DataFrame(columns=columns), runtime_path, columns)
            diagnostic["status"] = RUNTIME_STATUS_CREATED
        diagnostic["runtime_file_exists"] = runtime_path.exists()
        diagnostic["created"] = True
    except Exception as exc:
        diagnostic["status"] = RUNTIME_STATUS_FALLBACK
        diagnostic["errors"].append(str(exc))
    return diagnostic


def validate_runtime_setting_row(row, setting_name, available_models):
    errors = []
    if str(row.get("設定名", "")).strip() != setting_name:
        errors.append("設定名が期待値と一致しません。")
    model_key = str(row.get("モデルキー", "")).strip()
    if not model_key:
        errors.append("モデルキーが空です。")
    elif model_key not in available_models:
        errors.append(f"利用できないモデルキーです: {model_key}")
    model_name = str(row.get("モデル名", "")).strip()
    if _has_corrupted_text(model_name):
        errors.append("モデル名に破損疑い文字 '?' が含まれています。")
    reason = row.get("根拠", "")
    if not isinstance(reason, str):
        errors.append("根拠が文字列ではありません。")
    return errors


def read_runtime_setting(runtime_path, template_path, columns, setting_name, available_models):
    columns = runtime_columns(columns)
    diagnostic = ensure_runtime_csv(runtime_path, template_path, columns)
    diagnostic["setting_name"] = setting_name
    runtime_path = Path(runtime_path)
    if not runtime_path.exists():
        diagnostic["status"] = RUNTIME_STATUS_FALLBACK
        return None, diagnostic
    try:
        settings = read_runtime_csv(runtime_path, columns)
    except Exception as exc:
        diagnostic["status"] = RUNTIME_STATUS_FALLBACK
        diagnostic["errors"].append(str(exc))
        return None, diagnostic

    missing = _missing_columns(settings, columns)
    if missing:
        diagnostic["status"] = RUNTIME_STATUS_BROKEN
        diagnostic["errors"].append("不足列: " + ", ".join(missing))
        return None, diagnostic
    if settings.empty:
        diagnostic["status"] = RUNTIME_STATUS_FALLBACK
        diagnostic["errors"].append("実行時設定が空です。")
        return None, diagnostic

    active = settings[settings["設定名"].astype(str) == setting_name]
    if active.empty:
        diagnostic["status"] = RUNTIME_STATUS_FALLBACK
        diagnostic["errors"].append("対象設定名の行がありません。")
        return None, diagnostic

    row = active.tail(1).iloc[0].to_dict()
    errors = validate_runtime_setting_row(row, setting_name, available_models)
    if errors:
        diagnostic["status"] = RUNTIME_STATUS_FALLBACK
        diagnostic["errors"].extend(errors)
        return None, diagnostic

    diagnostic["status"] = RUNTIME_STATUS_OK
    diagnostic["model_key"] = str(row["モデルキー"])
    diagnostic["model_name"] = str(row["モデル名"])
    diagnostic["reason"] = str(row.get("根拠", ""))
    return {
        "model_key": diagnostic["model_key"],
        "model_name": diagnostic["model_name"],
        "reason": diagnostic["reason"],
    }, diagnostic


def write_runtime_csv_atomic(df, path, columns):
    columns = runtime_columns(columns)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    output = df.reindex(columns=columns)
    output.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    check = read_runtime_csv(tmp_path, columns)
    missing = _missing_columns(check, columns)
    if missing:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise ValueError("実行時設定の一時保存検証に失敗しました: " + ", ".join(missing))
    tmp_path.replace(path)


def save_runtime_setting(
    runtime_path,
    template_path,
    columns,
    setting_name,
    model_key,
    model_name,
    reason,
    available_models,
    updated_at,
    history_csv=None,
    game=None,
    change_source="unknown",
):
    columns = runtime_columns(columns)
    runtime_path = Path(runtime_path)
    row = _setting_row(
        _clean_runtime_value(setting_name),
        _clean_runtime_value(model_key),
        _clean_runtime_value(model_name),
        _clean_runtime_value(reason),
        _clean_runtime_value(updated_at),
    )
    old_setting = _current_runtime_setting_snapshot(runtime_path, columns, row["設定名"])
    result = _runtime_save_result(runtime_path, row)
    validation_errors = _validate_runtime_save_row(row, row["設定名"], available_models)
    if validation_errors:
        result.update(
            success=False,
            status="rejected",
            message="設定内容が不正なため保存しませんでした。",
            errors=validation_errors,
        )
        _attach_history_result(
            result,
            history_csv,
            game,
            row,
            old_setting,
            "rejected",
            change_source,
            validation_errors,
        )
        return result

    ensure_result = ensure_runtime_csv(runtime_path, template_path, columns)
    if ensure_result.get("errors") and not runtime_path.exists():
        result.update(
            success=False,
            status="failed",
            message="現在設定ファイルを準備できませんでした。",
            errors=list(ensure_result["errors"]),
        )
        _attach_history_result(
            result,
            history_csv,
            game,
            row,
            old_setting,
            "failed",
            change_source,
            result["errors"],
        )
        return result

    try:
        settings = read_runtime_csv(runtime_path, columns)
        missing = _missing_columns(settings, columns)
        if missing:
            raise ValueError("現在設定CSVの不足列: " + ", ".join(missing))
        active = settings[settings["設定名"].map(_clean_runtime_value) == row["設定名"]]
        old_setting = active.tail(1).iloc[0].to_dict() if not active.empty else old_setting
        if _same_runtime_setting(old_setting, row):
            result.update(
                success=True,
                status="no_change",
                message="現在設定と同一のため書き換えませんでした。",
                changed=False,
            )
            _attach_history_result(
                result,
                history_csv,
                game,
                row,
                old_setting,
                "no_change",
                change_source,
                [],
            )
            return result

        remaining = settings[settings["設定名"].map(_clean_runtime_value) != row["設定名"]]
        updated = pd.concat([remaining.reindex(columns=columns), pd.DataFrame([row])], ignore_index=True)
        write_runtime_csv_atomic(updated, runtime_path, columns)
    except Exception as exc:
        result.update(
            success=False,
            status="failed",
            message="現在設定の保存に失敗しました。",
            errors=[str(exc)],
        )
        _attach_history_result(
            result,
            history_csv,
            game,
            row,
            old_setting,
            "failed",
            change_source,
            result["errors"],
        )
        return result

    result.update(
        success=True,
        status="success",
        message="現在設定を保存しました。",
        current_setting_saved=True,
        changed=True,
        runtime_file_exists=runtime_path.exists(),
    )
    effective_source = "initialization" if ensure_result.get("created") else change_source
    _attach_history_result(
        result,
        history_csv,
        game,
        row,
        old_setting,
        "success",
        effective_source,
        [],
    )
    return result


def _clean_runtime_value(value):
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _validate_runtime_save_row(row, setting_name, available_models):
    errors = list(validate_runtime_setting_row(row, setting_name, available_models))
    model_key = _clean_runtime_value(row.get("モデルキー"))
    model_name = _clean_runtime_value(row.get("モデル名"))
    reason = _clean_runtime_value(row.get("根拠"))
    updated_at = _clean_runtime_value(row.get("更新日時"))
    if not model_name:
        errors.append("モデル名が空です。")
    expected_name = _available_model_names(available_models).get(model_key, "")
    if expected_name and model_name and model_name != expected_name:
        errors.append("モデル名がモデルキーの定義と一致しません。")
    if not reason:
        errors.append("変更理由が空です。")
    if not updated_at:
        errors.append("更新日時が空です。")
    return list(dict.fromkeys(errors))


def _current_runtime_setting_snapshot(runtime_path, columns, setting_name):
    empty = _setting_row(setting_name, "", "", "", "")
    runtime_path = Path(runtime_path)
    if not runtime_path.is_file():
        return empty
    try:
        settings = read_runtime_csv(runtime_path, columns)
    except Exception:
        return empty
    if _missing_columns(settings, columns) or settings.empty:
        return empty
    active = settings[settings["設定名"].map(_clean_runtime_value) == setting_name]
    return active.tail(1).iloc[0].to_dict() if not active.empty else empty


def _same_runtime_setting(old_setting, new_setting):
    return all(
        _clean_runtime_value(old_setting.get(column)) == _clean_runtime_value(new_setting.get(column))
        for column in ("設定名", "モデルキー", "モデル名", "根拠")
    )


def _runtime_save_result(runtime_path, row):
    return {
        "success": False,
        "status": "failed",
        "message": "",
        "current_setting_saved": False,
        "history_saved": False,
        "history_id": "",
        "changed": False,
        "runtime_file_exists": Path(runtime_path).is_file(),
        "model_key": row["モデルキー"],
        "model_name": row["モデル名"],
        "reason": row["根拠"],
        "warnings": [],
        "errors": [],
    }


def _runtime_history_id(game):
    prefix = {"loto6": "L6", "loto7": "L7"}.get(_clean_runtime_value(game), "RX")
    return f"{prefix}-RUNTIME-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"


def _normalized_change_source(change_source):
    source = _clean_runtime_value(change_source)
    return source if source in RUNTIME_HISTORY_CHANGE_SOURCES else "unknown"


def _runtime_history_row(game, new_setting, old_setting, save_result, change_source, errors=None):
    return {
        "履歴ID": _runtime_history_id(game),
        "ゲーム": _clean_runtime_value(game),
        "設定名": _clean_runtime_value(new_setting.get("設定名")),
        "旧モデルキー": _clean_runtime_value(old_setting.get("モデルキー")),
        "旧モデル名": _clean_runtime_value(old_setting.get("モデル名")),
        "新モデルキー": _clean_runtime_value(new_setting.get("モデルキー")),
        "新モデル名": _clean_runtime_value(new_setting.get("モデル名")),
        "変更理由": _clean_runtime_value(new_setting.get("根拠")),
        "変更元": _normalized_change_source(change_source),
        "変更日時": _clean_runtime_value(new_setting.get("更新日時")) or datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        "保存結果": save_result,
        "エラー内容": " / ".join(_clean_runtime_value(error) for error in (errors or []) if _clean_runtime_value(error)),
    }


def _read_runtime_history_csv(path):
    path = Path(path)
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def read_runtime_setting_history(history_csv, game=None, limit=None):
    path = Path(history_csv)
    if not path.is_file():
        return pd.DataFrame(columns=RUNTIME_SETTING_HISTORY_COLUMNS)
    try:
        history = _read_runtime_history_csv(path)
    except Exception:
        return pd.DataFrame(columns=RUNTIME_SETTING_HISTORY_COLUMNS)
    if _missing_columns(history, RUNTIME_SETTING_HISTORY_COLUMNS):
        return pd.DataFrame(columns=RUNTIME_SETTING_HISTORY_COLUMNS)

    history = history.reindex(columns=RUNTIME_SETTING_HISTORY_COLUMNS).copy()
    if game is not None:
        history = history[history["ゲーム"].map(_clean_runtime_value) == _clean_runtime_value(game)]
    history["_changed_at"] = pd.to_datetime(history["変更日時"], errors="coerce")
    history["_row_order"] = range(len(history))
    history = history.sort_values(["_changed_at", "_row_order"], ascending=[False, False], na_position="last")
    history = history.drop(columns=["_changed_at", "_row_order"])
    if limit is not None:
        try:
            history = history.head(max(0, int(limit)))
        except (TypeError, ValueError):
            pass
    return history.reset_index(drop=True)


def _write_runtime_history_atomic(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    output = history.reindex(columns=RUNTIME_SETTING_HISTORY_COLUMNS)
    try:
        output.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        check = _read_runtime_history_csv(tmp_path)
        missing = _missing_columns(check, RUNTIME_SETTING_HISTORY_COLUMNS)
        if missing:
            raise ValueError("履歴CSVの一時保存検証に失敗しました: " + ", ".join(missing))
        ids = check["履歴ID"].map(_clean_runtime_value)
        if not ids.empty and (ids.eq("").any() or ids.duplicated().any()):
            raise ValueError("履歴IDが空、または重複しています。")
        if len(check) != len(output):
            raise ValueError("履歴CSVの行数検証に失敗しました。")
        tmp_path.replace(path)
    except Exception:
        try:
            if tmp_path.is_file():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def append_runtime_setting_history(history_csv, history_row):
    path = Path(history_csv)
    row = {column: _clean_runtime_value(history_row.get(column)) for column in RUNTIME_SETTING_HISTORY_COLUMNS}
    result = {"success": False, "saved": False, "skipped": False, "history_id": row["履歴ID"], "errors": []}
    try:
        if path.exists():
            existing = _read_runtime_history_csv(path)
            missing = _missing_columns(existing, RUNTIME_SETTING_HISTORY_COLUMNS)
            if missing:
                raise ValueError("既存履歴CSVの不足列: " + ", ".join(missing))
            existing = existing.reindex(columns=RUNTIME_SETTING_HISTORY_COLUMNS)
        else:
            existing = pd.DataFrame(columns=RUNTIME_SETTING_HISTORY_COLUMNS)

        if not existing.empty and row["保存結果"] == "no_change":
            latest = existing.tail(1).iloc[0]
            comparable = [
                "ゲーム",
                "設定名",
                "旧モデルキー",
                "旧モデル名",
                "新モデルキー",
                "新モデル名",
                "変更理由",
                "変更元",
                "保存結果",
            ]
            if all(_clean_runtime_value(latest.get(column)) == row[column] for column in comparable):
                result.update(success=True, skipped=True, history_id=_clean_runtime_value(latest.get("履歴ID")))
                return result

        existing_ids = existing["履歴ID"].map(_clean_runtime_value) if "履歴ID" in existing else pd.Series(dtype=str)
        if not row["履歴ID"] or row["履歴ID"] in set(existing_ids):
            raise ValueError("履歴IDが空、または既存履歴と重複しています。")
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        _write_runtime_history_atomic(combined, path)
    except Exception as exc:
        result["errors"].append(str(exc))
        return result
    result.update(success=True, saved=True)
    return result


def _attach_history_result(result, history_csv, game, new_setting, old_setting, save_result, change_source, errors):
    if history_csv is None:
        return
    history_row = _runtime_history_row(game, new_setting, old_setting, save_result, change_source, errors)
    history_result = append_runtime_setting_history(history_csv, history_row)
    result["history_saved"] = bool(history_result.get("saved"))
    result["history_id"] = history_result.get("history_id", "")
    if not history_result.get("success"):
        result["warnings"].append("設定履歴を保存できませんでした: " + " / ".join(history_result.get("errors") or []))


def diagnose_runtime_setting_history(history_csv):
    path = Path(history_csv)
    diagnostic = {
        "path": str(path),
        "display_path": _display_path(path),
        "status": "missing",
        "status_label": RUNTIME_HISTORY_STATUS_LABELS["missing"],
        "file_exists": path.exists(),
        "readable": False,
        "required_columns_ok": False,
        "history_count": 0,
        "duplicate_history_ids": 0,
        "invalid_games": 0,
        "invalid_save_results": 0,
        "missing_datetime_count": 0,
        "corrupted_model_name_count": 0,
        "error_count": 0,
        "rejected_count": 0,
        "failed_count": 0,
        "latest_changed_at": "",
        "gitignored": _gitignored_state(path),
        "warnings": [],
        "errors": [],
    }
    if not path.exists():
        return diagnostic
    try:
        history = _read_runtime_history_csv(path)
        diagnostic["readable"] = True
    except Exception as exc:
        diagnostic["status"] = "error"
        diagnostic["status_label"] = RUNTIME_HISTORY_STATUS_LABELS["error"]
        diagnostic["errors"].append(str(exc))
        return diagnostic

    missing = _missing_columns(history, RUNTIME_SETTING_HISTORY_COLUMNS)
    if missing:
        diagnostic["status"] = "corrupted"
        diagnostic["status_label"] = RUNTIME_HISTORY_STATUS_LABELS["corrupted"]
        diagnostic["errors"].append("不足列: " + ", ".join(missing))
        return diagnostic

    diagnostic["required_columns_ok"] = True
    history = history.reindex(columns=RUNTIME_SETTING_HISTORY_COLUMNS)
    diagnostic["history_count"] = int(len(history))
    ids = history["履歴ID"].map(_clean_runtime_value)
    diagnostic["duplicate_history_ids"] = int(ids[ids.duplicated(keep=False) & ids.ne("")].nunique())
    games = history["ゲーム"].map(_clean_runtime_value)
    diagnostic["invalid_games"] = int((~games.isin(RUNTIME_HISTORY_GAMES)).sum())
    save_results = history["保存結果"].map(_clean_runtime_value)
    diagnostic["invalid_save_results"] = int((~save_results.isin(RUNTIME_HISTORY_SAVE_RESULTS)).sum())
    changed_at = history["変更日時"].map(_clean_runtime_value)
    diagnostic["missing_datetime_count"] = int(changed_at.eq("").sum())
    old_names = history["旧モデル名"].map(_clean_runtime_value)
    new_names = history["新モデル名"].map(_clean_runtime_value)
    diagnostic["corrupted_model_name_count"] = int((old_names.str.contains("?", regex=False) | new_names.str.contains("?", regex=False)).sum())
    diagnostic["error_count"] = int(history["エラー内容"].map(_clean_runtime_value).ne("").sum())
    diagnostic["rejected_count"] = int(save_results.eq("rejected").sum())
    diagnostic["failed_count"] = int(save_results.eq("failed").sum())
    latest = read_runtime_setting_history(path, limit=1)
    if not latest.empty:
        diagnostic["latest_changed_at"] = _clean_runtime_value(latest.iloc[0].get("変更日時"))

    corrupted = any(
        diagnostic[key]
        for key in ("duplicate_history_ids", "invalid_games", "invalid_save_results", "corrupted_model_name_count")
    )
    warning = any(
        diagnostic[key]
        for key in ("missing_datetime_count", "error_count", "rejected_count", "failed_count")
    )
    if corrupted:
        diagnostic["status"] = "corrupted"
    elif warning:
        diagnostic["status"] = "warning"
    else:
        diagnostic["status"] = "healthy"
    if history.empty:
        diagnostic["status_label"] = "履歴なし"
    else:
        diagnostic["status_label"] = RUNTIME_HISTORY_STATUS_LABELS[diagnostic["status"]]
    if diagnostic["duplicate_history_ids"]:
        diagnostic["warnings"].append("履歴IDが重複しています。")
    if diagnostic["invalid_games"]:
        diagnostic["warnings"].append("不正なゲーム名があります。")
    if diagnostic["invalid_save_results"]:
        diagnostic["warnings"].append("不正な保存結果があります。")
    if diagnostic["missing_datetime_count"]:
        diagnostic["warnings"].append("変更日時が空欄の履歴があります。")
    if diagnostic["corrupted_model_name_count"]:
        diagnostic["warnings"].append("モデル名に破損疑い文字 '?' を含む履歴があります。")
    return diagnostic


def _available_model_names(available_models):
    if isinstance(available_models, dict):
        return {str(key): _clean_runtime_value(value) for key, value in available_models.items()}
    return {str(key): "" for key in (available_models or [])}


def _find_project_root(path):
    path = Path(path)
    for parent in (path.parent, *path.parents):
        if (parent / ".gitignore").is_file():
            return parent
    return None


def _display_path(path):
    path = Path(path)
    root = _find_project_root(path)
    if root is None:
        return path.name
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _gitignored_state(path):
    root = _find_project_root(path)
    if root is None:
        return None
    gitignore = root / ".gitignore"
    try:
        try:
            lines = gitignore.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeDecodeError:
            lines = gitignore.read_text(encoding="cp932").splitlines()
        target = Path(path).relative_to(root).as_posix()
    except Exception:
        return None

    ignored = False
    for raw_line in lines:
        pattern = raw_line.strip().replace("\\", "/")
        if not pattern or pattern.startswith("#") or pattern.startswith("!"):
            continue
        pattern = pattern.lstrip("/").rstrip("/")
        if fnmatchcase(target, pattern) or fnmatchcase(target, f"*/{pattern}"):
            ignored = True
    return ignored


def _diagnose_runtime_file(path, columns, setting_name, available_models, allow_blank_metadata=False):
    path = Path(path)
    model_names = _available_model_names(available_models)
    result = {
        "exists": path.exists(),
        "readable": False,
        "required_columns_ok": False,
        "setting_name_ok": False,
        "model_key_ok": False,
        "model_name_ok": False,
        "model_key": "",
        "model_name": "",
        "reason": "",
        "updated_at": "",
        "row_count": 0,
        "matching_row_count": 0,
        "warnings": [],
        "errors": [],
        "status": "missing",
    }
    if not path.exists():
        return result

    try:
        settings = read_runtime_csv(path, columns)
        result["readable"] = True
    except Exception as exc:
        result["status"] = "error"
        result["errors"].append(str(exc))
        return result

    result["row_count"] = int(len(settings))
    missing = _missing_columns(settings, columns)
    if missing:
        result["status"] = "corrupted"
        result["errors"].append("不足列: " + ", ".join(missing))
        return result
    result["required_columns_ok"] = True
    if settings.empty:
        result["status"] = "fallback"
        result["errors"].append("実行時設定が空です。")
        return result

    names = settings["設定名"].map(_clean_runtime_value)
    active = settings[names == setting_name]
    result["matching_row_count"] = int(len(active))
    if active.empty:
        result["status"] = "fallback"
        result["errors"].append("対象設定名の行がありません。")
        return result

    result["setting_name_ok"] = True
    if len(active) > 1:
        result["warnings"].append(f"対象設定名の行が{len(active)}件あります。最新行を表示しています。")
    row = active.tail(1).iloc[0].to_dict()
    result["model_key"] = _clean_runtime_value(row.get("モデルキー"))
    result["model_name"] = _clean_runtime_value(row.get("モデル名"))
    result["reason"] = _clean_runtime_value(row.get("根拠"))
    result["updated_at"] = _clean_runtime_value(row.get("更新日時"))

    if result["model_key"] and result["model_key"] in model_names:
        result["model_key_ok"] = True
    else:
        result["errors"].append("有効なモデルキーではありません。")

    if result["model_name"] and "?" not in result["model_name"]:
        expected_name = model_names.get(result["model_key"], "")
        if expected_name and result["model_name"] != expected_name:
            result["errors"].append("モデル名がモデルキーの定義と一致しません。")
        else:
            result["model_name_ok"] = True
    else:
        result["errors"].append("モデル名が空、または破損疑い文字 '?' を含みます。")

    if not allow_blank_metadata and not result["reason"]:
        result["warnings"].append("根拠が空欄です。")
    if not allow_blank_metadata and not result["updated_at"]:
        result["warnings"].append("更新日時が空欄です。")
    if result["errors"]:
        result["status"] = "corrupted"
    elif result["warnings"]:
        result["status"] = "warning"
    else:
        result["status"] = "healthy"
    return result


def diagnose_runtime_setting(
    runtime_csv,
    template_csv,
    columns=None,
    setting_name=None,
    available_models=None,
    create=False,
    *,
    expected_setting_name=None,
    valid_model_keys=None,
):
    """Return a read-only health report for one runtime settings CSV.

    The original Ver1.13 six-argument form remains supported. The shorter
    Ver1.14 form is also accepted as ``(runtime, template, setting, models)``.
    ``create`` is retained for call-site compatibility but is intentionally
    ignored because diagnostics must never initialize or repair files.
    """
    del create
    if expected_setting_name is not None:
        setting_name = expected_setting_name
        available_models = valid_model_keys
    elif isinstance(columns, str) and setting_name is not None and available_models is None:
        available_models = setting_name
        setting_name = columns
        columns = RUNTIME_SETTING_COLUMNS

    columns = runtime_columns(columns)
    runtime_path = Path(runtime_csv)
    template_path = Path(template_csv) if template_csv else runtime_path.with_name("template.csv")
    setting_name = _clean_runtime_value(setting_name)
    model_names = _available_model_names(available_models)
    runtime = _diagnose_runtime_file(runtime_path, columns, setting_name, model_names)
    template = _diagnose_runtime_file(
        template_path,
        columns,
        setting_name,
        model_names,
        allow_blank_metadata=True,
    )

    diagnostic = {
        "path": str(runtime_path),
        "display_path": _display_path(runtime_path),
        "status": runtime["status"],
        "status_label": RUNTIME_DIAGNOSIS_STATUS_LABELS.get(runtime["status"], "不明"),
        "setting_name": setting_name,
        "model_key": runtime["model_key"],
        "model_name": runtime["model_name"],
        "reason": runtime["reason"],
        "updated_at": runtime["updated_at"],
        "runtime_dir_exists": runtime_path.parent.exists(),
        "runtime_file_exists": runtime["exists"],
        "template_file_exists": template["exists"],
        "readable": runtime["readable"],
        "required_columns_ok": runtime["required_columns_ok"],
        "setting_name_ok": runtime["setting_name_ok"],
        "model_key_ok": runtime["model_key_ok"],
        "model_name_ok": runtime["model_name_ok"],
        "row_count": runtime["row_count"],
        "matching_row_count": runtime["matching_row_count"],
        "fallback_active": runtime["status"] in {"fallback", "missing", "corrupted", "error"},
        "fallback_model_key": "machine_learning" if "machine_learning" in model_names else "",
        "gitignored": _gitignored_state(runtime_path),
        "warnings": list(runtime["warnings"]),
        "errors": list(runtime["errors"]),
        "template_status": template["status"],
        "template_status_label": RUNTIME_DIAGNOSIS_STATUS_LABELS.get(template["status"], "不明"),
        "template_readable": template["readable"],
        "template_required_columns_ok": template["required_columns_ok"],
        "template_setting_name_ok": template["setting_name_ok"],
        "template_model_key_ok": template["model_key_ok"],
        "template_model_name_ok": template["model_name_ok"],
        "template_warnings": list(template["warnings"]),
        "template_errors": list(template["errors"]),
    }
    if diagnostic["gitignored"] is True:
        diagnostic["git_managed"] = "対象外"
    elif diagnostic["gitignored"] is False:
        diagnostic["git_managed"] = "対象"
        diagnostic["warnings"].append("runtime設定が.gitignore対象として確認できません。")
    else:
        diagnostic["git_managed"] = "判定不能"
        diagnostic["warnings"].append(".gitignoreを読み込めず、Git管理外を確認できません。")
    if not template["exists"]:
        diagnostic["warnings"].append("テンプレートがありません。")
    if template["warnings"]:
        diagnostic["warnings"].append("テンプレート: " + " / ".join(template["warnings"]))
    if template["errors"]:
        diagnostic["warnings"].append("テンプレートに不備があります。")
    elif template["status"] not in {"healthy", "warning"}:
        diagnostic["warnings"].append("テンプレートを正常に確認できません。")

    if diagnostic["status"] == "healthy" and diagnostic["warnings"]:
        diagnostic["status"] = "warning"
    diagnostic["status_label"] = RUNTIME_DIAGNOSIS_STATUS_LABELS.get(diagnostic["status"], "不明")
    return diagnostic
