import json
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

import prospective_evidence as pe  # noqa: E402


COMMIT_A = "a" * 40
COMMIT_B = "b" * 40


def merged_history(game="loto6"):
    size = 6 if game == "loto6" else 7
    return pd.DataFrame([{"開催回": 10, "抽せん日": "2026/01/01", "numbers": list(range(1, size + 1)), "bonus": [size + 1]}])


def context(game="loto6", target=11, draw_date="2026/01/05"):
    return {"game": game, "target_draw": target, "draw_date": draw_date, "status": "resolved", "reason": "", "history": merged_history(game)}


def diagnosis(game="loto6", commit=COMMIT_A, input_hash="f" * 64, now=None, model="catalog"):
    return pe.diagnose_prospective(context(game), now or datetime(2026, 1, 2), commit, input_hash, model)


def preview(game="loto6", commit=COMMIT_A, models=None):
    diag = diagnosis(game, commit)
    result = pe.generate_preview(context(game), diag, models or ["frequency_analysis", "random_baseline"], datetime(2026, 1, 2, 10))
    return diag, result


class ProspectiveEvidenceTests(unittest.TestCase):
    def test_games_do_not_mix(self):
        _, l6 = preview("loto6", models=["frequency_analysis"])
        _, l7 = preview("loto7", models=["frequency_analysis"])
        self.assertEqual(set(l6["records"]["game"]), {"loto6"})
        self.assertEqual(set(l7["records"]["game"]), {"loto7"})

    def test_registered_or_future_input_draw_is_rejected(self):
        bad = context()
        bad["target_draw"] = 10
        result = pe.diagnose_prospective(bad, datetime(2025, 12, 31), COMMIT_A, "f" * 64)
        self.assertFalse(result["prospective"])
        self.assertTrue(any("対象回以降" in reason for reason in result["reasons"]))

    def test_past_and_same_day_are_rejected(self):
        self.assertFalse(pe.diagnose_prospective(context(draw_date="2026/01/01"), datetime(2026, 1, 2), COMMIT_A, "f" * 64)["prospective"])
        self.assertFalse(pe.diagnose_prospective(context(), datetime(2026, 1, 5, 1), COMMIT_A, "f" * 64)["prospective"])

    def test_future_draw_with_prior_input_is_allowed(self):
        result = diagnosis()
        self.assertTrue(result["prospective"])
        self.assertLess(result["input_latest_draw"], result["target_draw"])

    def test_test_demo_sample_are_rejected(self):
        for name in ("test model", "demo", "sample"):
            self.assertFalse(diagnosis(model=name)["prospective"])

    def test_missing_model_commit_and_input_hash_are_rejected(self):
        self.assertFalse(diagnosis(model="")["prospective"])
        self.assertFalse(diagnosis(commit="")["prospective"])
        self.assertFalse(diagnosis(input_hash="")["prospective"])

    def test_fixed_seed_is_reproducible_and_game_scoped(self):
        _, first = preview("loto6", models=["random_baseline"])
        _, second = preview("loto6", models=["random_baseline"])
        _, l7 = preview("loto7", models=["random_baseline"])
        self.assertEqual(first["records"].iloc[0]["予測番号"], second["records"].iloc[0]["予測番号"])
        self.assertNotEqual(first["records"].iloc[0]["使用seed"], l7["records"].iloc[0]["使用seed"])

    def test_evidence_ids_are_unique_and_commit_scoped(self):
        _, first = preview(commit=COMMIT_A)
        _, second = preview(commit=COMMIT_B)
        self.assertFalse(first["records"]["evidence_id"].duplicated().any())
        self.assertNotEqual(set(first["records"]["evidence_id"]), set(second["records"]["evidence_id"]))

    def test_model_failure_is_isolated(self):
        original = pe.build_model_scores
        def failing(rows, key, *args, **kwargs):
            if key == "cold_analysis":
                raise RuntimeError("isolated")
            return original(rows, key, *args, **kwargs)
        with patch.object(pe, "build_model_scores", side_effect=failing):
            diag = diagnosis()
            result = pe.generate_preview(context(), diag, ["frequency_analysis", "cold_analysis"])
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(len(result["errors"]), 1)

    def test_empty_or_rejected_preview_is_safe(self):
        rejected = diagnosis(now=datetime(2026, 1, 6))
        result = pe.generate_preview(context(), rejected)
        self.assertTrue(result["records"].empty)

    def test_save_verify_duplicate_and_different_commit(self):
        diag, result = preview(commit=COMMIT_A, models=["frequency_analysis"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = pe.save_batch(result["records"], diag, root, datetime(2026, 1, 2, 11))
            self.assertEqual(pe.verify_batch(saved["path"])["status"], "valid")
            with self.assertRaises(FileExistsError):
                pe.save_batch(result["records"], diag, root, datetime(2026, 1, 2, 12))
            diag_b, result_b = preview(commit=COMMIT_B, models=["frequency_analysis"])
            second = pe.save_batch(result_b["records"], diag_b, root, datetime(2026, 1, 2, 13))
            self.assertTrue(second["path"].exists())

    def test_invalid_numbers_and_duplicate_ids_are_rejected(self):
        diag, result = preview(models=["frequency_analysis", "random_baseline"])
        with tempfile.TemporaryDirectory() as directory:
            broken = result["records"].copy()
            broken.at[0, "予測番号"] = "01-01-02-03-04-05"
            with self.assertRaises(ValueError):
                pe.save_batch(broken, diag, Path(directory), datetime(2026, 1, 2, 11))
            duplicate = pd.concat([result["records"].iloc[[0]], result["records"].iloc[[0]]], ignore_index=True)
            with self.assertRaises(ValueError):
                pe.save_batch(duplicate, diag, Path(directory), datetime(2026, 1, 2, 12))

    def test_csv_tamper_is_detected(self):
        diag, result = preview(models=["frequency_analysis"])
        with tempfile.TemporaryDirectory() as directory:
            saved = pe.save_batch(result["records"], diag, Path(directory), datetime(2026, 1, 2, 11))
            csv_path = saved["path"] / "predictions.csv"
            csv_path.write_bytes(csv_path.read_bytes() + b"\n")
            self.assertEqual(pe.verify_batch(saved["path"])["status"], "invalid_hash")

    def test_manifest_tamper_is_detected(self):
        diag, result = preview(models=["frequency_analysis"])
        with tempfile.TemporaryDirectory() as directory:
            saved = pe.save_batch(result["records"], diag, Path(directory), datetime(2026, 1, 2, 11))
            path = saved["path"] / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["draw_no"] = 999
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(pe.verify_batch(saved["path"])["status"], "invalid_hash")

    def test_atomic_failure_leaves_no_batch(self):
        diag, result = preview(models=["frequency_analysis"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                pe.save_batch(result["records"], diag, root, datetime(2026, 1, 2, 11), fail_stage="after_write")
            self.assertEqual(list(root.rglob("predictions.csv")), [])
            self.assertEqual(list(root.rglob("*.tmp-*")), [])

    def test_result_registration_does_not_modify_evidence(self):
        diag, result = preview(models=["frequency_analysis"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = pe.save_batch(result["records"], diag, root, datetime(2026, 1, 2, 11))
            before = {path.name: path.read_bytes() for path in saved["path"].iterdir()}
            results = pd.DataFrame([{"開催回": 11}])
            state = pe.scan_evidence(root, {"loto6": results})
            self.assertEqual(state.iloc[0]["evaluation_status"], "evaluation_ready")
            after = {path.name: path.read_bytes() for path in saved["path"].iterdir()}
            self.assertEqual(before, after)

    def test_source_hash_changes_with_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            path.write_text("a\n1\n", encoding="utf-8")
            first = pe.source_files_sha256([path])
            path.write_text("a\n2\n", encoding="utf-8")
            self.assertNotEqual(first, pe.source_files_sha256([path]))


if __name__ == "__main__":
    unittest.main()
