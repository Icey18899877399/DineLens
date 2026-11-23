from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs" / "autodl-20261123" / "runs"


class TestVisualizationData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from aspect_detection.visualization_data import load_visualization_data
        cls.data = load_visualization_data(RESULTS)

    def test_matched_aspects_and_summary_values(self):
        data = self.data
        self.assertEqual(data["test_reviews"], 4940)
        self.assertEqual(data["test_pairs"], 88920)
        self.assertEqual(len(data["aspects"]), 18)
        self.assertEqual(len({row["aspect"] for row in data["aspects"]}), 18)
        self.assertAlmostEqual(data["models"][0]["aspect_macro_f1"], 0.7601101611501203)
        self.assertAlmostEqual(data["models"][1]["end_to_end_negative_f1"], 0.6638176638176638)
        for row in data["aspects"]:
            self.assertIn("precision", row["baseline"])
            self.assertIn("recall", row["bert"])
            self.assertLessEqual(row["support"], data["test_reviews"])
        self.assertEqual(sum(row["bert"]["f1"] > row["baseline"]["f1"] for row in data["aspects"]), 17)

    def test_sentiment_normalizes_by_gold_labels(self):
        data = self.data
        self.assertEqual(data["sentiment_support"], 28362)
        bert = data["models"][1]["confusion"]
        self.assertEqual(bert["counts"][1][2], 1855)
        self.assertEqual(bert["row_support"], [3813, 7026, 17523])
        for model in data["models"]:
            matrix = model["confusion"]
            for raw, rates, n in zip(matrix["counts"], matrix["rates"], matrix["row_support"]):
                self.assertEqual(sum(raw), n)
                self.assertAlmostEqual(sum(rates), 1.0)
                self.assertEqual(rates, [count / n for count in raw])

    def test_candidate_coverage_and_verbatim_example(self):
        evidence = self.data["evidence"]
        self.assertEqual(evidence["candidate_counts"], {"zero": 274, "one": 75, "multiple": 11})
        self.assertEqual(sum(evidence["candidate_counts"].values()), evidence["n_pairs_with_gold_negative_aspect"])
        self.assertIsNone(evidence["formal_metrics_path"])
        example = evidence["example"]
        for span in example["methods"].values():
            self.assertEqual(example["text"][span["start"]:span["end"]], "等了四十分钟才上菜")
            self.assertEqual(example["text"][span["start"]:span["end"]], span["text"])

