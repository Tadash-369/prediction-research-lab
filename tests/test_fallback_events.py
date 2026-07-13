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

from runtime_settings import (
    FALLBACK_EVENT_COLUMNS,
    diagnose_fallback_events,
    read_fallback_events,
    record_fallback_event,
    select_prediction_model,
)


MODELS = {
    "machine_learning": "機械学習モデル",
    "markov_chain": "マルコフ連鎖",
}


class FallbackEventTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.events = self.root / "runtime" / "loto6_fallback_events.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def record(self, **overrides):
        values = {
            "game": "loto6",
            "occurrence_location": "loto6_prediction_generation",
            "cause_code": "model_key_invalid",
            "cause_detail": "invalid runtime model",
            "source_setting_name": "active_next_prediction",
            "source_model_key": "invalid",
            "source_model_name": "不正モデル",
            "fallback_method": "safe_default",
            "fallback_model_key": "score_balance_v1",
            "fallback_model_name": "標準スコアバランス",
            "target_round": "2020",
            "runtime_state_token": "state-a",
            "occurred_at": "2026/07/13 10:00:00",
        }
        values.update(overrides)
        return record_fallback_event(self.events, **values)

    def write_events(self, path, rows, columns=FALLBACK_EVENT_COLUMNS):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")

    def test_records_one_valid_event(self):
        result = self.record()

        self.assertTrue(result["success"])
        self.assertTrue(result["recorded"])
        self.assertTrue(result["event_id"].startswith("L6-FALLBACK-"))
        events = read_fallback_events(self.events)
        self.assertEqual(FALLBACK_EVENT_COLUMNS, events.columns.tolist())
        self.assertEqual(1, len(events))
        self.assertEqual("score_balance_v1", events.iloc[0]["代替モデルキー"])
        self.assertEqual("unresolved", events.iloc[0]["復旧状態"])

    def test_suppresses_same_unresolved_event_within_24_hours(self):
        first = self.record()
        second = self.record(occurred_at="2026/07/13 10:05:00")

        self.assertTrue(first["recorded"])
        self.assertTrue(second["duplicate_suppressed"])
        self.assertFalse(second["recorded"])
        self.assertEqual(1, len(read_fallback_events(self.events)))

    def test_different_round_model_or_runtime_state_records_new_event(self):
        self.record()
        different_round = self.record(target_round="2021")
        different_model = self.record(
            target_round="2020",
            fallback_method="machine_learning_default",
            fallback_model_key="machine_learning",
            fallback_model_name=MODELS["machine_learning"],
        )
        different_state = self.record(runtime_state_token="state-b")

        self.assertTrue(different_round["recorded"])
        self.assertTrue(different_model["recorded"])
        self.assertTrue(different_state["recorded"])
        self.assertEqual(4, len(read_fallback_events(self.events)))

    def test_broken_existing_event_file_is_not_overwritten(self):
        self.events.parent.mkdir(parents=True, exist_ok=True)
        self.events.write_text("bad,column\n1,2\n", encoding="utf-8")
        before_hash = hashlib.sha256(self.events.read_bytes()).hexdigest()

        result = self.record()

        self.assertFalse(result["success"])
        self.assertFalse(result["recorded"])
        self.assertEqual(before_hash, hashlib.sha256(self.events.read_bytes()).hexdigest())

    def test_read_and_diagnose_are_read_only(self):
        missing = self.root / "runtime" / "missing.csv"
        missing_diagnosis = diagnose_fallback_events(missing)
        self.assertEqual("missing", missing_diagnosis["status"])
        self.assertFalse(missing.exists())
        self.assertTrue(read_fallback_events(missing).empty)
        self.assertFalse(missing.exists())

        self.record()
        before_hash = hashlib.sha256(self.events.read_bytes()).hexdigest()
        before_mtime = self.events.stat().st_mtime_ns
        events = read_fallback_events(self.events, limit=1)
        diagnosis = diagnose_fallback_events(self.events)

        self.assertEqual(1, len(events))
        self.assertEqual("warning", diagnosis["status"])
        self.assertEqual(1, diagnosis["unresolved_count"])
        self.assertEqual(before_hash, hashlib.sha256(self.events.read_bytes()).hexdigest())
        self.assertEqual(before_mtime, self.events.stat().st_mtime_ns)

    def test_diagnoses_corruption_cases(self):
        self.record()
        normal = read_fallback_events(self.events).iloc[0].to_dict()

        duplicate_path = self.root / "runtime" / "duplicate.csv"
        self.write_events(duplicate_path, [normal, normal])
        self.assertEqual("corrupted", diagnose_fallback_events(duplicate_path)["status"])

        invalid = dict(normal)
        invalid["イベントID"] = "L6-FALLBACK-INVALID"
        invalid["重複キー"] = "different-key"
        invalid["原因コード"] = "invalid_cause"
        invalid["代替方式"] = "invalid_method"
        invalid["代替モデル名"] = "?model"
        invalid_path = self.root / "runtime" / "invalid.csv"
        self.write_events(invalid_path, [invalid])
        invalid_diagnosis = diagnose_fallback_events(invalid_path)
        self.assertEqual("corrupted", invalid_diagnosis["status"])
        self.assertEqual(1, invalid_diagnosis["invalid_cause_codes"])
        self.assertEqual(1, invalid_diagnosis["invalid_fallback_methods"])
        self.assertEqual(1, invalid_diagnosis["corrupted_model_name_count"])

        missing_columns = self.root / "runtime" / "missing_columns.csv"
        self.write_events(missing_columns, [["id"]], ["イベントID"])
        self.assertEqual("corrupted", diagnose_fallback_events(missing_columns)["status"])

        broken = self.root / "runtime" / "broken.csv"
        broken.write_text('"unterminated\n', encoding="utf-8")
        self.assertEqual("error", diagnose_fallback_events(broken)["status"])

    def test_prediction_model_selection_preserves_existing_priorities(self):
        runtime = {"model_key": "markov_chain", "model_name": MODELS["markov_chain"]}

        loto6_runtime = select_prediction_model(None, runtime, MODELS, "score_balance_v1", "標準スコアバランス", "safe_default")
        loto6_fallback = select_prediction_model(None, None, MODELS, "score_balance_v1", "標準スコアバランス", "safe_default")
        loto7_best = select_prediction_model("markov_chain", None, MODELS, "machine_learning", MODELS["machine_learning"], "machine_learning_default")
        loto7_runtime = select_prediction_model(None, runtime, MODELS, "machine_learning", MODELS["machine_learning"], "machine_learning_default")
        loto7_default = select_prediction_model(None, None, MODELS, "machine_learning", MODELS["machine_learning"], "machine_learning_default")

        self.assertEqual("markov_chain", loto6_runtime["selected_model_key"])
        self.assertFalse(loto6_runtime["fallback_active"])
        self.assertEqual("score_balance_v1", loto6_fallback["selected_model_key"])
        self.assertEqual("safe_default", loto6_fallback["fallback_method"])
        self.assertEqual("markov_chain", loto7_best["selected_model_key"])
        self.assertEqual("best_key", loto7_best["fallback_method"])
        self.assertEqual("markov_chain", loto7_runtime["selected_model_key"])
        self.assertFalse(loto7_runtime["fallback_active"])
        self.assertEqual("machine_learning", loto7_default["selected_model_key"])
        self.assertEqual("machine_learning_default", loto7_default["fallback_method"])


if __name__ == "__main__":
    unittest.main()
