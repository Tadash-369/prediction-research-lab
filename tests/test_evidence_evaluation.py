import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

CORE_DIR = Path(__file__).resolve().parents[1] / "loto_lab" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import evidence_evaluation as ee  # noqa: E402
import prospective_evidence as pe  # noqa: E402


COMMIT = "a" * 40


def context(game="loto7"):
    size = 7 if game == "loto7" else 6
    history = pd.DataFrame([{"開催回": 10, "抽せん日": "2026/01/01", "numbers": list(range(1, size + 1)), "bonus": [size + 1]}])
    return {"game": game, "target_draw": 11, "draw_date": "2026/01/05", "status": "resolved", "reason": "", "history": history}


def diagnosis(game="loto7"):
    return pe.diagnose_prospective(context(game), datetime(2026, 1, 2), COMMIT, "f" * 64)


def official_result(game="loto7", **changes):
    row = {"開催回": 11, "抽せん日": "2026/01/05", "本数字": "01-02-03-04-05-06-07" if game == "loto7" else "01-02-03-04-05-06", "ボーナス数字": "08-09" if game == "loto7" else "07"}
    row.update(changes)
    return row


class EvidenceEvaluationTests(unittest.TestCase):
    def make_batch(self, root, game="loto7"):
        diag = diagnosis(game)
        preview = pe.generate_preview(context(game), diag, ["frequency_analysis", "random_baseline"], datetime(2026, 1, 2, 10))
        return pe.save_batch(preview["records"], diag, Path(root), datetime(2026, 1, 2, 11))["path"]

    def test_awaiting_result_and_result_available_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = self.make_batch(directory)
            waiting = ee.diagnose_evaluation(batch)
            self.assertEqual(waiting["status"], "awaiting_result")
            self.assertFalse(waiting["evaluation_ready"])
            ready = ee.diagnose_evaluation(batch, official_result(), "loto7")
            self.assertEqual(ready["result_status"], "result_available")
            self.assertEqual(ready["status"], "evaluation_ready")
            self.assertTrue(ready["evaluation_ready"])

    def test_integrity_and_result_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = self.make_batch(directory)
            self.assertEqual(ee.diagnose_evaluation(batch, official_result(), "loto6")["status"], "result_mismatch")
            self.assertEqual(ee.diagnose_evaluation(batch, official_result(開催回=12), "loto7")["status"], "result_mismatch")
            self.assertEqual(ee.diagnose_evaluation(batch, official_result(抽せん日="2026/01/06"), "loto7")["status"], "result_mismatch")
            csv_path = batch / "predictions.csv"
            csv_path.write_bytes(csv_path.read_bytes() + b"\n")
            self.assertEqual(ee.diagnose_evaluation(batch)["status"], "invalid_integrity")

    def test_invalid_main_and_bonus_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = self.make_batch(directory)
            bad_main = ee.diagnose_evaluation(batch, official_result(本数字="01-01-02-03-04-05-06"), "loto7")
            bad_bonus = ee.diagnose_evaluation(batch, official_result(ボーナス数字="08-08"), "loto7")
            self.assertIn("invalid_main_numbers", bad_main["reasons"])
            self.assertIn("invalid_bonus_numbers", bad_bonus["reasons"])

    def test_record_metrics_model_summary_and_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = self.make_batch(directory)
            ready = ee.diagnose_evaluation(batch, official_result(), "loto7")
            preview = ee.build_evaluation_preview(batch, ready, datetime(2026, 1, 6))
            self.assertEqual(len(preview["records"]), 2)
            for _, row in preview["records"].iterrows():
                predicted = set(int(value) for value in row["予測番号"].split("-"))
                self.assertEqual(row["本数字一致数"], len(predicted & set(range(1, 8))))
                self.assertEqual(row["一致した本数字"], "-".join(f"{value:02d}" for value in sorted(predicted & set(range(1, 8)))))
            self.assertTrue((preview["model_summary"]["評価期間区分"] == "単回結果").all())
            self.assertTrue((preview["model_summary"]["口数効率"] == preview["model_summary"]["平均一致数"]).all())
            random_avg = preview["model_summary"].loc[preview["model_summary"]["正規化モデル名"] == "random_baseline", "平均一致数"].iloc[0]
            self.assertTrue((preview["model_summary"]["ランダムモデルとの差"] == preview["model_summary"]["平均一致数"] - random_avg).all())
            self.assertEqual(len(ee.to_accuracy_ticket_metrics(preview["records"])), 2)

    def test_result_and_evaluation_hashes_are_reproducible(self):
        row = official_result()
        self.assertEqual(ee.result_row_sha256("loto7", row), ee.result_row_sha256("loto7", dict(row)))
        with tempfile.TemporaryDirectory() as directory:
            batch = self.make_batch(directory)
            ready = ee.diagnose_evaluation(batch, row, "loto7")
            first = ee.build_evaluation_preview(batch, ready, datetime(2026, 1, 6))
            second = ee.build_evaluation_preview(batch, ready, datetime(2026, 1, 6))
            pd.testing.assert_frame_equal(first["records"], second["records"])
            self.assertEqual(ee._evaluation_batch_hash(first["records"], first["model_summary"]), ee._evaluation_batch_hash(second["records"], second["model_summary"]))

    def test_atomic_save_verify_duplicate_and_immutability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_root, evaluation_root = root / "evidence", root / "evaluations"
            batch = self.make_batch(evidence_root)
            before = {path.name: path.read_bytes() for path in batch.iterdir()}
            result_path = root / "results.csv"
            pd.DataFrame([official_result()]).to_csv(result_path, index=False, encoding="utf-8-sig")
            ready = ee.diagnose_evaluation(batch, official_result(), "loto7", evaluation_root)
            preview = ee.build_evaluation_preview(batch, ready, datetime(2026, 1, 6))
            saved = ee.save_evaluation(preview, ready, evaluation_root, result_path, Path(__file__).resolve().parents[1], datetime(2026, 1, 6))
            checked = ee.verify_evaluation(saved["path"])
            self.assertEqual(checked["status"], "valid")
            self.assertEqual(before, {path.name: path.read_bytes() for path in batch.iterdir()})
            evaluated = ee.diagnose_evaluation(batch, official_result(), "loto7", evaluation_root)
            self.assertEqual(evaluated["status"], "evaluated")
            with self.assertRaises(FileExistsError):
                ee.save_evaluation(preview, ready, evaluation_root, result_path, Path(__file__).resolve().parents[1], datetime(2026, 1, 7))

    def test_different_result_or_commit_can_be_reevaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = self.make_batch(root / "evidence")
            evaluation_root = root / "evaluations"
            result_path = root / "results.csv"
            row = official_result()
            pd.DataFrame([row]).to_csv(result_path, index=False, encoding="utf-8-sig")
            ready = ee.diagnose_evaluation(batch, row, "loto7")
            preview = ee.build_evaluation_preview(batch, ready)
            ee.save_evaluation(preview, ready, evaluation_root, result_path, Path(__file__).resolve().parents[1], datetime(2026, 1, 6))
            with patch.object(ee, "get_git_commit", return_value="b" * 40):
                code_changed = ee.save_evaluation(preview, ready, evaluation_root, result_path, Path(__file__).resolve().parents[1], datetime(2026, 1, 6, 1))
            self.assertEqual(code_changed["manifest"]["evaluation_kind"], "reevaluation")
            corrected = official_result(本数字="02-03-04-05-06-07-08")
            corrected_ready = ee.diagnose_evaluation(batch, corrected, "loto7")
            corrected_preview = ee.build_evaluation_preview(batch, corrected_ready)
            pd.DataFrame([corrected]).to_csv(result_path, index=False, encoding="utf-8-sig")
            saved = ee.save_evaluation(corrected_preview, corrected_ready, evaluation_root, result_path, Path(__file__).resolve().parents[1], datetime(2026, 1, 7))
            self.assertEqual(saved["manifest"]["evaluation_kind"], "corrected_result_evaluation")

    def test_failure_leaves_no_incomplete_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = self.make_batch(root / "evidence")
            result_path = root / "results.csv"
            pd.DataFrame([official_result()]).to_csv(result_path, index=False, encoding="utf-8-sig")
            ready = ee.diagnose_evaluation(batch, official_result(), "loto7")
            preview = ee.build_evaluation_preview(batch, ready)
            with self.assertRaises(RuntimeError):
                ee.save_evaluation(preview, ready, root / "evaluations", result_path, Path(__file__).resolve().parents[1], fail_stage="after_write")
            self.assertEqual(list((root / "evaluations").rglob("evaluation_manifest.json")), [])


if __name__ == "__main__":
    unittest.main()
