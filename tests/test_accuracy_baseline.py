import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd


CORE_DIR = Path(__file__).resolve().parents[1] / "loto_lab" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from accuracy_baseline import (  # noqa: E402
    REPORT_FILES,
    analyze_game,
    build_accuracy_baseline,
    build_draw_metrics,
    build_marginal_contribution,
    build_model_accuracy,
    build_overlap_matrix,
    build_random_draw_metrics,
    build_ticket_metrics,
    classify_predictions,
    planned_report_files,
    save_accuracy_baseline_report,
    _stable_seed,
)


def prediction(prediction_id, draw_no, numbers, model="model_a", saved_at="2026/01/01 10:00:00", reason="formal"):
    return {
        "予想ID": prediction_id,
        "開催回": draw_no,
        "予想日": saved_at.split(" ")[0],
        "候補番号": 1,
        "予想番号": numbers,
        "使用モデル": model,
        "予想理由": reason,
        "保存日時": saved_at,
    }


def result(draw_no, numbers, bonus="07", draw_date="2026/01/02"):
    return {"開催回": draw_no, "抽せん日": draw_date, "本数字": numbers, "ボーナス数字": bonus}


class AccuracyBaselineTests(unittest.TestCase):
    def setUp(self):
        self.loto6_results = pd.DataFrame(
            [
                result(1, "01-02-03-04-05-06", "07", "2026/01/02"),
                result(2, "02-03-04-05-06-07", "08", "2026/01/09"),
                result(3, "03-04-05-06-07-08", "09", "2026/01/16"),
            ]
        )

    def test_games_are_analyzed_separately(self):
        loto6 = pd.DataFrame([prediction("L6-1", 1, "01-02-03-10-11-12")])
        loto7 = pd.DataFrame([prediction("L7-1", 10, "01-02-03-04-05-06-07")])
        loto7_results = pd.DataFrame([result(10, "01-02-03-04-05-06-07", "08-09")])
        report = build_accuracy_baseline(loto6, self.loto6_results, loto7, loto7_results)
        self.assertEqual(set(report["games"]["loto6"]["ticket_metrics"]["game"]), {"loto6"})
        self.assertEqual(set(report["games"]["loto7"]["ticket_metrics"]["game"]), {"loto7"})

    def test_prediction_created_after_draw_is_excluded(self):
        frame = pd.DataFrame([prediction("late", 1, "01-02-03-04-05-06", saved_at="2026/01/03 10:00:00")])
        classified = classify_predictions(frame, self.loto6_results, "loto6")
        self.assertTrue(classified["eligible"].empty)
        self.assertIn("future_data_suspected", set(classified["quality"]["issue_code"]))

    def test_same_day_without_draw_time_is_indeterminate(self):
        frame = pd.DataFrame([prediction("same-day", 1, "01-02-03-04-05-06", saved_at="2026/01/02 08:00:00")])
        classified = classify_predictions(frame, self.loto6_results, "loto6")
        self.assertTrue(classified["eligible"].empty)
        self.assertIn("same_day_timing_indeterminate", classified["excluded"].iloc[0]["除外理由"])

    def test_unverified_formal_prediction_is_separate(self):
        frame = pd.DataFrame([prediction("pending", 99, "01-02-03-04-05-06")])
        classified = classify_predictions(frame, self.loto6_results, "loto6")
        self.assertEqual(len(classified["unverified"]), 1)
        self.assertEqual(classified["unverified"].iloc[0]["分類"], "未検証の正式保存予測")

    def test_existing_verification_history_is_classified_separately(self):
        frame = pd.DataFrame([prediction("verified-1", 1, "01-02-03-10-11-12")])
        verification = pd.DataFrame([{"予想ID": "verified-1", "開催回": 1}])
        classified = classify_predictions(frame, self.loto6_results, "loto6", verification)
        self.assertEqual(classified["eligible"].iloc[0]["分類"], "正式検証済み予測")

    def test_regenerated_and_demo_predictions_are_not_formal(self):
        frame = pd.DataFrame(
            [
                prediction("research-regenerated-1", 1, "01-02-03-04-05-06", reason="研究用再生成"),
                prediction("demo-1", 1, "01-02-03-04-05-06", reason="demo"),
            ]
        )
        classified = classify_predictions(frame, self.loto6_results, "loto6")
        self.assertTrue(classified["eligible"].empty)
        self.assertEqual(set(classified["classified"]["分類"]), {"現在のコードによる再生成予測", "サンプル・テスト・デモ予測"})

    def test_invalid_range_and_duplicate_numbers_are_detected(self):
        frame = pd.DataFrame(
            [
                prediction("range", 1, "01-02-03-04-05-44"),
                prediction("duplicate", 1, "01-02-03-04-05-05"),
            ]
        )
        quality = classify_predictions(frame, self.loto6_results, "loto6")["quality"]
        self.assertIn("number_out_of_range", set(quality["issue_code"]))
        self.assertIn("duplicate_numbers", set(quality["issue_code"]))

    def test_multiple_ticket_metrics_and_correction(self):
        frame = pd.DataFrame(
            [
                prediction("a1", 1, "01-02-03-10-11-12"),
                {**prediction("a2", 1, "04-05-06-13-14-15"), "候補番号": 2},
            ]
        )
        classified = classify_predictions(frame, self.loto6_results, "loto6")
        tickets = build_ticket_metrics(classified["eligible"], self.loto6_results, "loto6")
        draws = build_draw_metrics(tickets, self.loto6_results, "loto6")
        self.assertEqual(list(tickets["main_matches"]), [3, 3])
        self.assertEqual(draws.iloc[0]["best_main_matches"], 3)
        self.assertEqual(draws.iloc[0]["main_number_coverage"], 6)
        self.assertEqual(draws.iloc[0]["coverage_per_ticket"], 3)

    def test_average_rates_std_and_recent_periods(self):
        rows = [
            prediction("m1", 1, "01-10-11-12-13-14"),
            prediction("m2", 2, "02-03-10-11-12-13", saved_at="2026/01/08 10:00:00"),
            prediction("m3", 3, "03-04-05-10-11-12", saved_at="2026/01/15 10:00:00"),
        ]
        analysis = analyze_game(pd.DataFrame(rows), self.loto6_results, "loto6")
        model = analysis["model_accuracy"].iloc[0]
        self.assertAlmostEqual(model["average_main_matches"], 2.0)
        self.assertAlmostEqual(model["std_main_matches"], (2 / 3) ** 0.5)
        self.assertAlmostEqual(model["hit_2_plus_rate"], 2 / 3)
        self.assertEqual(model["recent_10_average"], 2.0)
        self.assertEqual(model["recent_30_average"], 2.0)
        self.assertTrue(model["ranking_eligible"])

    def test_fixed_seed_random_is_reproducible_and_game_scoped(self):
        frame = pd.DataFrame([prediction("a1", 1, "01-02-03-10-11-12")])
        analysis = analyze_game(frame, self.loto6_results, "loto6")
        first = build_random_draw_metrics(analysis["draw_metrics"], self.loto6_results, "loto6")
        second = build_random_draw_metrics(analysis["draw_metrics"], self.loto6_results, "loto6")
        pd.testing.assert_frame_equal(first, second)
        self.assertNotEqual(_stable_seed("loto6", 1, "model_a"), _stable_seed("loto7", 1, "model_a"))

    def test_saved_random_with_matching_ticket_count_is_preferred(self):
        draw_metrics = pd.DataFrame(
            [
                {"game": "loto6", "model": "model_a", "draw_no": 1, "ticket_count": 2, "average_main_matches": 1.5, "best_main_matches": 2, "main_number_coverage": 3},
                {"game": "loto6", "model": "ランダム予測モデル", "draw_no": 1, "ticket_count": 2, "average_main_matches": 0.5, "best_main_matches": 1, "main_number_coverage": 1},
            ]
        )
        baseline = build_random_draw_metrics(draw_metrics, self.loto6_results, "loto6")
        target = baseline[baseline["model"] == "model_a"].iloc[0]
        self.assertEqual(target["random_baseline_source"], "正式保存random予測")
        self.assertEqual(target["random_average_matches"], 0.5)

    def test_union_jaccard_does_not_use_optimal_ticket_matching(self):
        tickets = pd.DataFrame(
            [
                {"game": "loto6", "model": "a", "draw_no": 1, "numbers": "01-02-03-04-05-06"},
                {"game": "loto6", "model": "a", "draw_no": 1, "numbers": "07-08-09-10-11-12"},
                {"game": "loto6", "model": "b", "draw_no": 1, "numbers": "01-02-03-13-14-15"},
            ]
        )
        overlap = build_overlap_matrix(tickets, "loto6")
        pair = overlap[(overlap["model_a"] == "a") & (overlap["model_b"] == "b")].iloc[0]
        self.assertAlmostEqual(pair["average_jaccard"], 3 / 15)
        self.assertEqual(pair["average_common_numbers"], 3)
        self.assertEqual(pair["exact_match_tickets"], 0)

    def test_marginal_contribution_counts_unique_hits(self):
        frame = pd.DataFrame(
            [
                prediction("a1", 1, "01-02-10-11-12-13", model="a"),
                prediction("b1", 1, "03-04-14-15-16-17", model="b"),
            ]
        )
        analysis = analyze_game(frame, self.loto6_results, "loto6")
        marginal = build_marginal_contribution(analysis["ticket_metrics"], self.loto6_results, "loto6")
        self.assertEqual(set(marginal["model"]), {"a", "b"})
        self.assertEqual(set(marginal["unique_hit_numbers_provided"]), {2})
        self.assertTrue((marginal["positive_contribution_draws"] == 1).all())
        self.assertTrue(marginal["method"].str.contains("研究用近似").all())

    def test_empty_data_is_safe(self):
        analysis = analyze_game(pd.DataFrame(), pd.DataFrame(), "loto6")
        for key in ("eligible", "ticket_metrics", "draw_metrics", "model_accuracy", "overlap", "marginal"):
            self.assertTrue(analysis[key].empty)

    def test_insufficient_data_is_not_ranked(self):
        frame = pd.DataFrame([prediction("a1", 1, "01-02-03-10-11-12")])
        analysis = analyze_game(frame, self.loto6_results, "loto6")
        self.assertFalse(bool(analysis["model_accuracy"].iloc[0]["ranking_eligible"]))
        self.assertEqual(analysis["rankings"], {})

    def test_duplicate_prediction_id_is_reported(self):
        frame = pd.DataFrame(
            [
                prediction("same", 1, "01-02-03-10-11-12"),
                {**prediction("same", 1, "04-05-06-13-14-15"), "候補番号": 2},
            ]
        )
        quality = classify_predictions(frame, self.loto6_results, "loto6")["quality"]
        self.assertEqual((quality["issue_code"] == "duplicate_prediction_id").sum(), 2)

    def test_snapshot_save_is_explicit_and_never_overwrites(self):
        frame = pd.DataFrame([prediction("a1", 1, "01-02-03-10-11-12")])
        report = build_accuracy_baseline(frame, self.loto6_results, pd.DataFrame(), pd.DataFrame())
        fixed = datetime(2026, 1, 20, 12, 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = list(root.rglob("*"))
            planned = planned_report_files(root, "20260120_120000")
            self.assertEqual(before, [])
            self.assertEqual(len(planned), len(REPORT_FILES) + 2)
            saved = save_accuracy_baseline_report(report, root, fixed)
            self.assertEqual(len(saved["files"]), len(REPORT_FILES) + 1)
            self.assertTrue(saved["latest"].exists())
            latest = json.loads(saved["latest"].read_text(encoding="utf-8"))
            self.assertEqual(latest["snapshot"], "snapshots/20260120_120000")
            with self.assertRaises(FileExistsError):
                save_accuracy_baseline_report(report, root, fixed)

    def test_analysis_does_not_write_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "protected.csv"
            marker.write_text("protected\nunchanged\n", encoding="utf-8")
            before = marker.read_bytes()
            build_accuracy_baseline(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
            self.assertEqual(marker.read_bytes(), before)
            self.assertEqual(list(root.iterdir()), [marker])


if __name__ == "__main__":
    unittest.main()
