from pathlib import Path

import pandas as pd


RUNTIME_STATUS_OK = "正常"
RUNTIME_STATUS_MISSING = "未作成"
RUNTIME_STATUS_CREATED = "自動生成済み"
RUNTIME_STATUS_BROKEN = "破損"
RUNTIME_STATUS_FALLBACK = "フォールバック中"
RUNTIME_SETTING_COLUMNS = ["設定名", "モデルキー", "モデル名", "根拠", "更新日時"]


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


def save_runtime_setting(runtime_path, template_path, columns, setting_name, model_key, model_name, reason, available_models, updated_at):
    columns = runtime_columns(columns)
    runtime_path = Path(runtime_path)
    ensure_result = ensure_runtime_csv(runtime_path, template_path, columns)
    row = _setting_row(setting_name, str(model_key or "").strip(), str(model_name or "").strip(), reason, updated_at)
    errors = validate_runtime_setting_row(row, setting_name, available_models)
    if errors:
        ensure_result["status"] = RUNTIME_STATUS_FALLBACK
        ensure_result["errors"].extend(errors)
        return ensure_result
    try:
        settings = read_runtime_csv(runtime_path, columns)
        if _missing_columns(settings, columns):
            settings = pd.DataFrame(columns=columns)
        settings = settings[settings["設定名"].astype(str) != setting_name] if not settings.empty else settings
        settings = pd.concat([settings.reindex(columns=columns), pd.DataFrame([row])], ignore_index=True)
        write_runtime_csv_atomic(settings, runtime_path, columns)
    except Exception as exc:
        ensure_result["status"] = RUNTIME_STATUS_FALLBACK
        ensure_result["errors"].append(str(exc))
        return ensure_result
    ensure_result["status"] = RUNTIME_STATUS_OK
    ensure_result["runtime_file_exists"] = runtime_path.exists()
    ensure_result["model_key"] = row["モデルキー"]
    ensure_result["model_name"] = row["モデル名"]
    ensure_result["reason"] = row["根拠"]
    return ensure_result


def diagnose_runtime_setting(runtime_path, template_path, columns, setting_name, available_models, create=False):
    columns = runtime_columns(columns)
    runtime_path = Path(runtime_path)
    if create:
        _, diagnostic = read_runtime_setting(runtime_path, template_path, columns, setting_name, available_models)
        return diagnostic

    diagnostic = _empty_diagnostic(runtime_path, setting_name)
    diagnostic["runtime_dir_exists"] = runtime_path.parent.exists()
    diagnostic["runtime_file_exists"] = runtime_path.exists()
    if not runtime_path.exists():
        diagnostic["status"] = RUNTIME_STATUS_MISSING
        return diagnostic
    try:
        settings = read_runtime_csv(runtime_path, columns)
    except Exception as exc:
        diagnostic["status"] = RUNTIME_STATUS_BROKEN
        diagnostic["errors"].append(str(exc))
        return diagnostic
    missing = _missing_columns(settings, columns)
    if missing:
        diagnostic["status"] = RUNTIME_STATUS_BROKEN
        diagnostic["errors"].append("不足列: " + ", ".join(missing))
        return diagnostic
    if settings.empty:
        diagnostic["status"] = RUNTIME_STATUS_FALLBACK
        diagnostic["errors"].append("実行時設定が空です。")
        return diagnostic
    active = settings[settings["設定名"].astype(str) == setting_name]
    if active.empty:
        diagnostic["status"] = RUNTIME_STATUS_FALLBACK
        diagnostic["errors"].append("対象設定名の行がありません。")
        return diagnostic
    row = active.tail(1).iloc[0].to_dict()
    errors = validate_runtime_setting_row(row, setting_name, available_models)
    if errors:
        diagnostic["status"] = RUNTIME_STATUS_FALLBACK
        diagnostic["errors"].extend(errors)
    else:
        diagnostic["status"] = RUNTIME_STATUS_OK
    diagnostic["model_key"] = str(row.get("モデルキー", ""))
    diagnostic["model_name"] = str(row.get("モデル名", ""))
    diagnostic["reason"] = str(row.get("根拠", ""))
    return diagnostic
