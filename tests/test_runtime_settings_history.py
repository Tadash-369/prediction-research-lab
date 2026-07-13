import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "loto_lab" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import runtime_settings
from runtime_settings import (
    RUNTIME_SETTING_COLUMNS,
    RUNTIME_SETTING_HISTORY_COLUMNS,
    diagnose_runtime_setting_history,
    read_runtime_csv,
    read_runtime_setting_history,
    save_runtime_setting,
)


MODELS = {
    "machine_learning": "機械学習モデル",
    "markov_chain": "マルコフ連鎖",
}
SETTING_NAME = "active_next_prediction"


class RuntimeSettingHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.runtime = self.root / "runtime" / "model_settings.csv"
        self.template = self.root / "config" / "model_settings.template.csv"
        self.history = self.root / "runtime" / "model_settings_history.csv"
        self.initial_row = [
            SETTING_NAME,
            "machine_learning",
            MODELS["machine_learning"],
            "初期設定",
            "2026/07/13 08:00:00",
        ]
        self.write_csv(self.runtime, [self.initial_row], RUNTIME_SETTING_COLUMNS)
        self.write_csv(self.template, [self.initial_row], RUNTIME_SETTING_COLUMNS)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, path, rows, columns):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")

    def save(self, model_key, model_name, reason, history=None, change_source="backtest"):
        return save_runtime_setting(
            self.runtime,
            self.template,
            RUNTIME_SETTING_COLUMNS,
            SETTING_NAME,
            model_key,
            model_name,
            reason,
            MODELS,
            "2026/07/13 09:00:00",
            history_csv=self.history if history is None else history,
            game="loto6",
            change_source=change_source,
        )

    def test_success_updates_current_setting_and_adds_history(self):
        result = self.save("markov_chain", MODELS["markov_chain"], "直近評価で最上位")

        self.assertTrue(result["success"])
        self.assertEqual("success", result["status"])
        self.assertTrue(result["current_setting_saved"])
        self.assertTrue(result["history_saved"])
        self.assertTrue(result["history_id"].startswith("L6-RUNTIME-"))
        current = read_runtime_csv(self.runtime, RUNTIME_SETTING_COLUMNS).iloc[-1]
        self.assertEqual("markov_chain", current["モデルキー"])
        history = read_runtime_setting_history(self.history)
        self.assertEqual(1, len(history))
        self.assertEqual("machine_learning", history.iloc[0]["旧モデルキー"])
        self.assertEqual("markov_chain", history.iloc[0]["新モデルキー"])
        self.assertEqual("backtest", history.iloc[0]["変更元"])

    def test_no_change_keeps_current_timestamp_and_deduplicates_repeated_history(self):
        before_hash = hashlib.sha256(self.runtime.read_bytes()).hexdigest()
        before_mtime = self.runtime.stat().st_mtime_ns

        first = self.save("machine_learning", MODELS["machine_learning"], "初期設定")
        second = self.save("machine_learning", MODELS["machine_learning"], "初期設定")

        self.assertEqual("no_change", first["status"])
        self.assertEqual("no_change", second["status"])
        self.assertFalse(first["current_setting_saved"])
        self.assertEqual(before_hash, hashlib.sha256(self.runtime.read_bytes()).hexdigest())
        self.assertEqual(before_mtime, self.runtime.stat().st_mtime_ns)
        history = read_runtime_setting_history(self.history)
        self.assertEqual(1, len(history))
        self.assertEqual("no_change", history.iloc[0]["保存結果"])

    def test_rejected_setting_does_not_change_current_file(self):
        before_hash = hashlib.sha256(self.runtime.read_bytes()).hexdigest()

        invalid_key = self.save("invalid", "不正モデル", "テスト")
        corrupted_name = self.save("machine_learning", "?model", "テスト")

        self.assertEqual("rejected", invalid_key["status"])
        self.assertEqual("rejected", corrupted_name["status"])
        self.assertEqual(before_hash, hashlib.sha256(self.runtime.read_bytes()).hexdigest())
        history = read_runtime_setting_history(self.history)
        self.assertEqual(2, len(history))
        self.assertTrue((history["保存結果"] == "rejected").all())

    def test_current_write_failure_preserves_current_and_records_failed_history(self):
        before_hash = hashlib.sha256(self.runtime.read_bytes()).hexdigest()
        with mock.patch.object(runtime_settings, "write_runtime_csv_atomic", side_effect=OSError("write blocked")):
            result = self.save("markov_chain", MODELS["markov_chain"], "テスト")

        self.assertEqual("failed", result["status"])
        self.assertFalse(result["current_setting_saved"])
        self.assertEqual(before_hash, hashlib.sha256(self.runtime.read_bytes()).hexdigest())
        history = read_runtime_setting_history(self.history)
        self.assertEqual("failed", history.iloc[0]["保存結果"])

    def test_broken_history_is_not_overwritten_after_current_save(self):
        self.history.parent.mkdir(parents=True, exist_ok=True)
        self.history.write_text("bad,column\n1,2\n", encoding="utf-8")
        before_history = hashlib.sha256(self.history.read_bytes()).hexdigest()

        result = self.save("markov_chain", MODELS["markov_chain"], "テスト")

        self.assertTrue(result["success"])
        self.assertTrue(result["current_setting_saved"])
        self.assertFalse(result["history_saved"])
        self.assertTrue(result["warnings"])
        self.assertEqual(before_history, hashlib.sha256(self.history.read_bytes()).hexdigest())

    def test_history_read_and_diagnosis_cases(self):
        missing = diagnose_runtime_setting_history(self.root / "runtime" / "missing_history.csv")
        self.assertEqual("missing", missing["status"])

        self.save("markov_chain", MODELS["markov_chain"], "テスト")
        healthy = diagnose_runtime_setting_history(self.history)
        self.assertEqual("healthy", healthy["status"])
        self.assertEqual(1, healthy["history_count"])

        normal = read_runtime_setting_history(self.history).iloc[0].to_dict()
        duplicate_path = self.root / "runtime" / "duplicate_history.csv"
        self.write_csv(duplicate_path, [normal, normal], RUNTIME_SETTING_HISTORY_COLUMNS)
        self.assertEqual("corrupted", diagnose_runtime_setting_history(duplicate_path)["status"])

        invalid_result = dict(normal)
        invalid_result["履歴ID"] = "L6-RUNTIME-INVALID"
        invalid_result["保存結果"] = "invalid"
        invalid_path = self.root / "runtime" / "invalid_result.csv"
        self.write_csv(invalid_path, [invalid_result], RUNTIME_SETTING_HISTORY_COLUMNS)
        self.assertEqual("corrupted", diagnose_runtime_setting_history(invalid_path)["status"])

        missing_columns = self.root / "runtime" / "missing_columns_history.csv"
        self.write_csv(missing_columns, [["id"]], ["履歴ID"])
        self.assertEqual("corrupted", diagnose_runtime_setting_history(missing_columns)["status"])

        broken = self.root / "runtime" / "broken_history.csv"
        broken.write_text('"unterminated\n', encoding="utf-8")
        self.assertEqual("error", diagnose_runtime_setting_history(broken)["status"])


if __name__ == "__main__":
    unittest.main()
