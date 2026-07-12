from pathlib import Path
from fnmatch import fnmatchcase

import pandas as pd


RUNTIME_STATUS_OK = "正常"
RUNTIME_STATUS_MISSING = "未作成"
RUNTIME_STATUS_CREATED = "自動生成済み"
RUNTIME_STATUS_BROKEN = "破損"
RUNTIME_STATUS_FALLBACK = "フォールバック中"
RUNTIME_SETTING_COLUMNS = ["設定名", "モデルキー", "モデル名", "根拠", "更新日時"]
RUNTIME_DIAGNOSIS_STATUS_LABELS = {
    "healthy": "正常",
    "warning": "注意",
    "fallback": "フォールバック中",
    "missing": "未作成",
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


def _clean_runtime_value(value):
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except Exception:
        pass
    return str(value).strip()


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
