import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "loto_lab" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from runtime_settings import RUNTIME_SETTING_COLUMNS, diagnose_runtime_setting


MODELS = {
    "machine_learning": "機械学習モデル",
    "markov_chain": "マルコフ連鎖",
}
SETTING_NAME = "active_next_prediction"


class RuntimeSettingDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / ".gitignore").write_text("runtime/model_settings.csv\n", encoding="utf-8")
        self.runtime = self.root / "runtime" / "model_settings.csv"
        self.template = self.root / "config" / "model_settings.template.csv"
        self.valid_row = [
            SETTING_NAME,
            "machine_learning",
            MODELS["machine_learning"],
            "初期設定",
            "2026/07/13 08:00:00",
        ]
        self.write_csv(self.runtime, [self.valid_row])
        self.write_csv(self.template, [self.valid_row])

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, path, rows, columns=RUNTIME_SETTING_COLUMNS):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")

    def diagnose(self, runtime=None, template=None):
        return diagnose_runtime_setting(
            runtime or self.runtime,
            template or self.template,
            SETTING_NAME,
            MODELS,
        )

    def test_healthy_diagnosis_is_read_only(self):
        before_hash = hashlib.sha256(self.runtime.read_bytes()).hexdigest()
        before_mtime = self.runtime.stat().st_mtime_ns

        diagnosis = self.diagnose()

        self.assertEqual("healthy", diagnosis["status"])
        self.assertFalse(diagnosis["fallback_active"])
        self.assertEqual("healthy", diagnosis["template_status"])
        self.assertTrue(diagnosis["gitignored"])
        self.assertEqual(before_hash, hashlib.sha256(self.runtime.read_bytes()).hexdigest())
        self.assertEqual(before_mtime, self.runtime.stat().st_mtime_ns)

    def test_missing_runtime_is_not_created(self):
        missing = self.root / "runtime" / "missing.csv"

        diagnosis = self.diagnose(runtime=missing)

        self.assertEqual("missing", diagnosis["status"])
        self.assertTrue(diagnosis["fallback_active"])
        self.assertFalse(missing.exists())

    def test_runtime_corruption_cases(self):
        cases = []

        empty = self.root / "runtime" / "empty.csv"
        empty.parent.mkdir(parents=True, exist_ok=True)
        empty.write_bytes(b"")
        cases.append(("empty", empty, "error"))

        missing_columns = self.root / "runtime" / "missing_columns.csv"
        self.write_csv(missing_columns, [[SETTING_NAME, "machine_learning"]], ["設定名", "モデルキー"])
        cases.append(("missing_columns", missing_columns, "corrupted"))

        invalid_key = self.root / "runtime" / "invalid_key.csv"
        self.write_csv(invalid_key, [[SETTING_NAME, "invalid", "不正モデル", "テスト", self.valid_row[4]]])
        cases.append(("invalid_key", invalid_key, "corrupted"))

        corrupted_name = self.root / "runtime" / "corrupted_name.csv"
        self.write_csv(corrupted_name, [[SETTING_NAME, "machine_learning", "?model", "テスト", self.valid_row[4]]])
        cases.append(("corrupted_name", corrupted_name, "corrupted"))

        duplicate = self.root / "runtime" / "duplicate.csv"
        self.write_csv(duplicate, [self.valid_row, self.valid_row])
        cases.append(("duplicate", duplicate, "warning"))

        parse_error = self.root / "runtime" / "parse_error.csv"
        parse_error.write_text('"unterminated\n', encoding="utf-8")
        cases.append(("parse_error", parse_error, "error"))

        for name, path, expected_status in cases:
            with self.subTest(name=name):
                self.assertEqual(expected_status, self.diagnose(runtime=path)["status"])

    def test_template_missing_and_corrupted_are_warnings_for_valid_runtime(self):
        missing_template = self.root / "config" / "missing.csv"
        missing = self.diagnose(template=missing_template)
        self.assertEqual("warning", missing["status"])
        self.assertEqual("missing", missing["template_status"])

        bad_template = self.root / "config" / "bad.csv"
        self.write_csv(bad_template, [[SETTING_NAME]], ["設定名"])
        corrupted = self.diagnose(template=bad_template)
        self.assertEqual("warning", corrupted["status"])
        self.assertEqual("corrupted", corrupted["template_status"])


if __name__ == "__main__":
    unittest.main()
