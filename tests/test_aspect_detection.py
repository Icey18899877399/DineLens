import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = (
    "Location#Transportation", "Location#Downtown", "Location#Easy_to_find",
    "Service#Queue", "Service#Hospitality", "Service#Parking", "Service#Timely",
    "Price#Level", "Price#Cost_effective", "Price#Discount",
    "Ambience#Decoration", "Ambience#Noise", "Ambience#Space", "Ambience#Sanitary",
    "Food#Portion", "Food#Taste", "Food#Appearance", "Food#Recommend",
)


def write_split(folder, split, entries):
    with (folder / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "review", "star", *KEYS))
        for review_id, text, labels in entries:
            writer.writerow((review_id, text, 3, *labels))


def fixture_dataset(folder, varied_sentiment=False):
    folder.mkdir()
    for split, suffixes in (
        ("train", ("甲", "乙", "丙", "丁")),
        ("dev", ("戊", "己")),
        ("test", ("庚", "辛")),
    ):
        entries = []
        for i, suffix in enumerate(suffixes):
            labels = (1, -2) * 9 if i % 2 == 0 else (-2, -1) * 9
            if varied_sentiment and split == "train" and i == 2:
                labels = (-1, *labels[1:])
            review = ("口味 不错 服务 很好" if i % 2 == 0 else "排队 太久 环境 很吵") + " " + suffix
            entries.append((f"{split}{i}", review, labels))
        write_split(folder, split, entries)


class TestDataAndMetrics(unittest.TestCase):
    def test_all_polarities_are_mentions_and_test_wins_duplicate(self):
        from aspect_detection.data import load_splits

        with tempfile.TemporaryDirectory() as work:
            folder = Path(work)
            labels = (-2, -1, 0, 1) + (-2,) * 14
            write_split(folder, "train", [("t1", "Ａ　菜不错", labels), ("t2", "别的菜", labels)])
            write_split(folder, "dev", [("d1", "一个评价", labels)])
            write_split(folder, "test", [("e1", "a菜不错", labels)])

            splits = load_splits(folder)
            self.assertEqual(splits["train"].ids, ("t2",))
            self.assertEqual(splits["test"].texts, ("a菜不错",))
            self.assertEqual(splits["train"].removed_duplicates, 1)
            self.assertEqual(splits["test"].labels[0].tolist(), [0, 1, 1, 1] + [0] * 14)
            self.assertEqual(splits["test"].polarities[0].tolist(), [-2, -1, 0, 1] + [-2] * 14)

    def test_dev_threshold_and_f1_use_positive_mention_decisions(self):
        from aspect_detection.metrics import evaluate, fit_thresholds

        gold = np.tile(np.array([[1], [0], [1], [0]]), (1, 18))
        scores = np.tile(np.array([[0.8], [0.6], [0.4], [0.1]]), (1, 18))
        thresholds = fit_thresholds(gold, scores)
        np.testing.assert_allclose(thresholds, np.full(18, 0.4))
        metrics = evaluate(gold, scores >= thresholds)
        self.assertAlmostEqual(metrics["macro_f1"], 0.8)
        self.assertAlmostEqual(metrics["micro_f1"], 0.8)
        self.assertEqual(metrics["aspects"][0]["support"], 2)
        self.assertAlmostEqual(metrics["aspects"][0]["recall"], 1.0)

    def test_validation_can_calibrate_an_all_positive_aspect(self):
        from aspect_detection.metrics import fit_thresholds

        gold = np.zeros((2, 18), dtype=int)
        scores = np.full((2, 18), 0.3)
        gold[:, 0] = 1
        thresholds = fit_thresholds(gold, scores)
        self.assertAlmostEqual(thresholds[0], 0.3)
        self.assertAlmostEqual(thresholds[1], 0.5)


class TestSentiment(unittest.TestCase):
    def test_metrics_ignore_unmentioned_aspects_including_false_positive_scores(self):
        from aspect_detection.metrics import evaluate_sentiment

        gold = np.full((1, 18), -2)
        gold[0, :3] = (-1, 0, 1)
        predicted = np.ones((1, 18), dtype=int)
        predicted[0, :3] = (0, 0, 1)
        result = evaluate_sentiment(gold, predicted)
        self.assertEqual(result["support"], 3)
        self.assertEqual(result["confusion_matrix"], [[0, 1, 0], [0, 1, 0], [0, 0, 1]])
        self.assertAlmostEqual(result["macro_f1"], 5 / 9)
        self.assertEqual(result["classes"][0]["recall"], 0.0)
        self.assertIsNone(result["aspects"][3]["macro_f1"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
        "requires PyTorch and Transformers",
    )
    def test_sentiment_loss_has_no_gradient_for_unmentioned_aspects(self):
        import torch
        from aspect_detection.bert import sentiment_loss

        logits = torch.tensor([[[0.8, 0.3, 0.2], [9.0, 1.0, -8.0]]], requires_grad=True)
        loss = sentiment_loss(logits, torch.tensor([[-1, -2]]))
        loss.backward()
        self.assertGreater(abs(logits.grad[0, 0, 0].item()), 0)
        torch.testing.assert_close(logits.grad[0, 1], torch.zeros(3))
        self.assertEqual(sentiment_loss(logits, torch.tensor([[-2, -2]])).item(), 0)


class TestCommandLine(unittest.TestCase):
    def cli(self, *args):
        result = subprocess.run(
            [sys.executable, "-m", "aspect_detection", *map(str, args)],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONIOENCODING": "cp1252" if args[0] == "predict" else "utf-8",
                "OMP_NUM_THREADS": "2",
            },
            capture_output=True,
            encoding="utf-8",
            timeout=130,
        )
        if result.returncode:
            self.fail(f"CLI failed: {result.stderr}\n{result.stdout}")
        return result.stdout

    def check_outputs(self, run):
        with (run / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 36)
        self.assertEqual([row["aspect"] for row in rows[:18]], list(KEYS))
        self.assertEqual([int(row["gold_mentioned"]) for row in rows[:4]], [1, 0, 1, 0])
        metrics = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["n_reviews"], 2)
        self.assertEqual(len(metrics["aspects"]), 18)
        sample = json.loads(self.cli("predict", "--run", run, "--text", "口味不错，服务很好。"))
        self.assertEqual([row["aspect"] for row in sample], list(KEYS))
        self.assertEqual(sample[0]["group"], "位置")
        self.assertEqual(sample[15]["name"], "口味")
        for row in sample:
            self.assertTrue(0 <= row["score"] <= 1)
            self.assertEqual(row["mentioned"], int(row["score"] >= row["threshold"]))
        return sample

    def test_baseline_trains_exports_and_reloads(self):
        with tempfile.TemporaryDirectory() as work:
            dataset = Path(work) / "dataset"
            fixture_dataset(dataset)
            run = Path(work) / "baseline"
            self.cli("train", "--model", "baseline", "--dataset", dataset, "--output", run)
            self.check_outputs(run)

    def check_sentiment_outputs(self, run):
        sample = self.check_outputs(run)
        metrics = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
        sentiment = metrics["sentiment"]
        self.assertEqual(sentiment["support"], 18)
        self.assertEqual(sum(sum(row) for row in sentiment["confusion_matrix"]), 18)
        with (run / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["gold_sentiment"]) for row in rows[:4]], [1, -2, 1, -2])
        for row in rows:
            scores = [float(row[f"{name}_score"]) for name in ("negative", "neutral", "positive")]
            self.assertAlmostEqual(sum(scores), 1.0, places=5)
            expected = int(row["sentiment_if_mentioned"]) if int(row["mentioned"]) else -2
            self.assertEqual(int(row["sentiment"]), expected)
        for row in sample:
            if row["mentioned"]:
                self.assertIn(row["sentiment"], (-1, 0, 1))
                self.assertAlmostEqual(sum(row["sentiment_scores"].values()), 1.0, places=5)
            else:
                self.assertEqual(row["sentiment"], -2)
        return rows

    def test_sentiment_baseline_trains_exports_and_reloads(self):
        with tempfile.TemporaryDirectory() as work:
            dataset = Path(work) / "dataset"
            fixture_dataset(dataset, varied_sentiment=True)
            run = Path(work) / "sentiment_baseline"
            self.cli("train", "--task", "sentiment", "--model", "baseline",
                     "--dataset", dataset, "--output", run)
            first_aspect = self.check_sentiment_outputs(run)[0]
            self.assertGreater(float(first_aspect["negative_score"]), 0)
            self.assertEqual(float(first_aspect["neutral_score"]), 0)
            self.assertGreater(float(first_aspect["positive_score"]), 0)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
        "BERT integration requires the optional PyTorch and Transformers packages",
    )
    def test_bert_trains_and_reloads_local_checkpoint(self):
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

        with tempfile.TemporaryDirectory() as work:
            dataset = Path(work) / "dataset"
            fixture_dataset(dataset)
            source = Path(work) / "tiny_bert"
            source.mkdir()
            vocab = source / "vocab.txt"
            characters = list(dict.fromkeys("口味不错服务很好排队太久环境吵"))
            words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *characters]
            vocab.write_text("\n".join(words) + "\n", encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(vocab))
            tokenizer.save_pretrained(source)
            config = BertConfig(
                vocab_size=len(words),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=24,
                max_position_embeddings=32,
                num_labels=18,
            )
            BertForSequenceClassification(config).save_pretrained(source)
            run = Path(work) / "bert"
            self.cli("train", "--model", "bert", "--dataset", dataset, "--output", run,
                     "--checkpoint", source, "--device", "cpu",
                     "--epochs", 1, "--batch-size", 2, "--max-length", 24)
            self.check_outputs(run)

            joint = Path(work) / "joint"
            self.cli("train", "--task", "sentiment", "--model", "bert",
                     "--dataset", dataset, "--output", joint, "--checkpoint", source,
                     "--device", "cpu", "--epochs", 1, "--batch-size", 2, "--max-length", 24)
            self.check_sentiment_outputs(joint)


if __name__ == "__main__":
    unittest.main()
