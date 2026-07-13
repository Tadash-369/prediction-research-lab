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
    diagnose_fallback_lifecycle,
    read_fallback_events,
    read_fallback_lifecycle,
    record_fallback_event,
    resolve_fallback_events,
    select_prediction_model,
    should_auto_resolve_fallback,
)


MODELS = {
    "machine_learning": "機械学習モデル",
    "markov_chain": "マルコフ連鎖",
}


class FallbackRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.events = self.root / "runtime" / "fallback_events.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def record(self, **overrides):
        values = {
            "game": "loto6",
            "occurrence_location": "loto6_prediction_generation",
            "cause_code": "runtime_missing",
            "fallback_method": "safe_default",
            "fallback_model_key": "score_balance_v1",
            "fallback_model_name": "標準スコアバランス",
            "source_setting_name": "active_next_prediction",
            "target_round": "2020",
            "runtime_state_token": "state-a",
            "occurred_at": "2026/07/13 08:00:00",
            "note": "fallback発生",
        }
        values.update(overrides)
        return record_fallback_event(self.events, **values)

    def resolve(self, **overrides):
        values = {
            "game": "loto6",
            "occurrence_location": "loto6_prediction_generation",
            "adopted_model_key": "markov_chain",
            "adopted_model_name": MODELS["markov_chain"],
            "target_round": "2021",
            "now": "2026/07/13 12:00:00",
        }
        values.update(overrides)
        return resolve_fallback_events(self.events, **values)

    def write_events(self, rows, columns=FALLBACK_EVENT_COLUMNS, path=None):
        target = path or self.events
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=columns).to_csv(target, index=False, encoding="utf-8-sig")
        return target

    def event_row(self, event_id, occurred_at, cause="runtime_missing", state="unresolved", recovered_at=""):
        return {
            "イベントID": event_id,
            "ゲーム": "loto6",
            "発生日時": occurred_at,
            "発生箇所": "loto6_prediction_generation",
            "原因コード": cause,
            "原因詳細": "test",
            "元設定名": "active_next_prediction",
            "元モデルキー": "",
            "元モデル名": "",
            "代替方式": "safe_default",
            "代替モデルキー": "score_balance_v1",
            "代替モデル名": "標準スコアバランス",
            "対象開催回": "2020",
            "重複キー": f"key-{event_id}",
            "復旧状態": state,
            "復旧日時": recovered_at,
            "備考": "original note",
        }

    def test_resolves_unresolved_event_and_preserves_immutable_fields(self):
        self.record()
        before = read_fallback_events(self.events).iloc[0].to_dict()

        result = self.resolve()
        after = read_fallback_events(self.events).iloc[0]

        self.assertTrue(result["success"])
        self.assertTrue(result["resolved"])
        self.assertEqual(1, result["resolved_count"])
        self.assertEqual("auto_resolved", after["復旧状態"])
        self.assertEqual("2026/07/13 12:00:00", after["復旧日時"])
        self.assertEqual(before["イベントID"], after["イベントID"])
        self.assertEqual(before["発生日時"], after["発生日時"])
        self.assertEqual(before["原因コード"], after["原因コード"])
        self.assertEqual(before["代替モデルキー"], after["代替モデルキー"])
        self.assertIn(str(before["備考"]), str(after["備考"]))
        self.assertIn("markov_chain", after["備考"])

    def test_resolves_only_same_game_and_occurrence(self):
        self.record(target_round="2020", runtime_state_token="a")
        self.record(target_round="2021", runtime_state_token="b")
        self.record(
            game="loto7",
            occurrence_location="loto7_prediction_generation",
            fallback_method="machine_learning_default",
            fallback_model_key="machine_learning",
            fallback_model_name=MODELS["machine_learning"],
            target_round="700",
            runtime_state_token="c",
        )
        self.record(occurrence_location="loto6_backtest", target_round="2022", runtime_state_token="d")

        result = self.resolve()
        events = read_fallback_events(self.events)

        self.assertEqual(2, result["resolved_count"])
        target = (events["ゲーム"] == "loto6") & (events["発生箇所"] == "loto6_prediction_generation")
        self.assertTrue((events.loc[target, "復旧状態"] == "auto_resolved").all())
        self.assertTrue((events.loc[~target, "復旧状態"] == "unresolved").all())

    def test_no_target_and_duplicate_resolution_do_not_rewrite_file(self):
        no_file = self.resolve()
        self.assertTrue(no_file["success"])
        self.assertFalse(no_file["resolved"])
        self.assertFalse(self.events.exists())

        self.record()
        first = self.resolve()
        first_events = read_fallback_events(self.events)
        first_time = first_events.iloc[0]["復旧日時"]
        before_hash = hashlib.sha256(self.events.read_bytes()).hexdigest()
        before_mtime = self.events.stat().st_mtime_ns
        second = self.resolve(now="2026/07/14 12:00:00")

        self.assertEqual(1, first["resolved_count"])
        self.assertEqual(0, second["resolved_count"])
        self.assertEqual(before_hash, hashlib.sha256(self.events.read_bytes()).hexdigest())
        self.assertEqual(before_mtime, self.events.stat().st_mtime_ns)
        self.assertEqual(first_time, read_fallback_events(self.events).iloc[0]["復旧日時"])

    def test_broken_or_missing_column_csv_is_not_overwritten(self):
        for name, text in (("broken", '"unterminated\n'), ("missing", "イベントID,ゲーム\nid,loto6\n")):
            with self.subTest(name=name):
                path = self.root / "runtime" / f"{name}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                result = resolve_fallback_events(
                    path,
                    "loto6",
                    "loto6_prediction_generation",
                    "markov_chain",
                    MODELS["markov_chain"],
                    now="2026/07/13 12:00:00",
                )
                self.assertFalse(result["success"])
                self.assertEqual(before_hash, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_lifecycle_diagnosis_calculates_durations_thresholds_and_recurrence(self):
        rows = [
            self.event_row("E1", "2026/07/20 11:00:00", cause="runtime_missing"),
            self.event_row("E2", "2026/07/19 06:00:00", cause="runtime_missing"),
            self.event_row("E3", "2026/07/17 04:00:00", cause="model_key_invalid"),
            self.event_row("E4", "2026/07/12 04:00:00", cause="model_key_invalid"),
            self.event_row("E5", "2026/07/20 00:00:00", cause="runtime_missing", state="auto_resolved", recovered_at="2026/07/20 05:00:00"),
        ]
        self.write_events(rows)

        diagnosis = diagnose_fallback_lifecycle(self.events, now="2026/07/20 12:00:00")
        lifecycle = read_fallback_lifecycle(self.events, now="2026/07/20 12:00:00")

        self.assertEqual("critical", diagnosis["status"])
        self.assertEqual(4, diagnosis["unresolved_count"])
        self.assertEqual(1, diagnosis["auto_resolved_count"])
        self.assertEqual(3, diagnosis["attention_24h_count"])
        self.assertEqual(2, diagnosis["warning_72h_count"])
        self.assertEqual(1, diagnosis["critical_168h_count"])
        self.assertEqual(5.0, diagnosis["average_recovery_hours"])
        self.assertEqual(3, diagnosis["recurrence_count"])
        self.assertIn("critical", lifecycle["未復旧レベル"].tolist())
        self.assertEqual(2, len(diagnosis["cause_summary"]))

    def test_lifecycle_diagnoses_missing_and_invalid_states_or_dates(self):
        missing = diagnose_fallback_lifecycle(self.root / "runtime" / "none.csv", now="2026/07/20 12:00:00")
        self.assertEqual("missing", missing["status"])

        cases = [
            ("before", self.event_row("E1", "2026/07/20 10:00:00", state="resolved", recovered_at="2026/07/20 09:00:00")),
            ("invalid_state", self.event_row("E2", "2026/07/20 10:00:00", state="bad_state")),
            ("bad_date", self.event_row("E3", "not-a-date")),
            ("missing_recovery", self.event_row("E4", "2026/07/20 10:00:00", state="auto_resolved", recovered_at="")),
            ("unresolved_with_recovery", self.event_row("E5", "2026/07/20 10:00:00", state="unresolved", recovered_at="2026/07/20 11:00:00")),
        ]
        for name, row in cases:
            with self.subTest(name=name):
                path = self.root / "runtime" / f"{name}.csv"
                self.write_events([row], path=path)
                self.assertEqual("corrupted", diagnose_fallback_lifecycle(path, now="2026/07/20 12:00:00")["status"])

    def test_auto_resolution_selection_conditions_for_both_games(self):
        runtime = {"model_key": "markov_chain", "model_name": MODELS["markov_chain"]}
        loto6_runtime = select_prediction_model(None, runtime, MODELS, "score_balance_v1", "標準スコアバランス", "safe_default")
        loto6_best = select_prediction_model("machine_learning", runtime, MODELS, "score_balance_v1", "標準スコアバランス", "safe_default")
        loto6_default = select_prediction_model(None, None, MODELS, "score_balance_v1", "標準スコアバランス", "safe_default")
        loto7_runtime = select_prediction_model(None, runtime, MODELS, "machine_learning", MODELS["machine_learning"], "machine_learning_default")
        loto7_best = select_prediction_model("machine_learning", runtime, MODELS, "machine_learning", MODELS["machine_learning"], "machine_learning_default")
        loto7_default = select_prediction_model(None, None, MODELS, "machine_learning", MODELS["machine_learning"], "machine_learning_default")

        self.assertTrue(should_auto_resolve_fallback(loto6_runtime, "markov_chain"))
        self.assertFalse(should_auto_resolve_fallback(loto6_best, "machine_learning"))
        self.assertFalse(should_auto_resolve_fallback(loto6_default, "score_balance_v1"))
        self.assertTrue(should_auto_resolve_fallback(loto7_runtime, "markov_chain"))
        self.assertFalse(should_auto_resolve_fallback(loto7_best, "machine_learning"))
        self.assertFalse(should_auto_resolve_fallback(loto7_default, "machine_learning"))
        self.assertFalse(should_auto_resolve_fallback(loto7_runtime, "machine_learning"))


if __name__ == "__main__":
    unittest.main()
