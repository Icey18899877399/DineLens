import csv
import json
import tempfile
import unittest
from pathlib import Path

TASTE = "Food#Taste"
TIMELY = "Service#Timely"
QUEUE = "Service#Queue"


class TestReviewCatalog(unittest.TestCase):
    def setUp(self):
        from web.catalog import ReviewCatalog

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.reviews, self.predictions = root / "test.csv", root / "test_predictions.csv"
        with self.reviews.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "review", "star"])
            writer.writerow(["010", "菜好吃，但上菜太慢。", "3"])
            writer.writerow(["020", "排队太久，菜的味道不错。", "4"])
        with self.predictions.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "aspect", "mentioned", "score", "threshold", "sentiment",
                             "gold_sentiment", "negative_score", "neutral_score", "positive_score"])
            writer.writerow(["010", TASTE, "1", ".9", ".5", "1", "1", ".05", ".05", ".9"])
            writer.writerow(["010", TIMELY, "1", ".8", ".5", "-1", "-1", ".9", ".1", "0"])
            writer.writerow(["010", QUEUE, "0", ".2", ".5", "-2", "-1", ".8", ".1", ".1"])
            writer.writerow(["020", TASTE, "1", ".8", ".5", "1", "1", ".1", ".1", ".8"])
            writer.writerow(["020", QUEUE, "1", ".7", ".5", "-1", "-2", ".8", ".1", ".1"])
        self.catalog = ReviewCatalog.from_files(self.reviews, self.predictions)

    def test_counts_unique_reviews_and_uses_gated_sentiment(self):
        result = self.catalog.overview()
        self.assertEqual((result["total_reviews"], result["negative_reviews"]), (2, 2))
        by_key = {row["aspect"]: row for row in result["aspects"]}
        self.assertEqual(len(by_key), 18)
        self.assertEqual(by_key[TASTE]["positive"], 2)
        self.assertEqual(by_key[QUEUE]["negative"], 1)
        self.assertEqual(by_key[QUEUE]["gold_negative"], 1)
        self.assertEqual(by_key[QUEUE]["false_positive"], 1)
        self.assertEqual(by_key[QUEUE]["false_negative"], 1)

    def test_filters_include_only_reported_aspects_and_paginate(self):
        negative = self.catalog.browse(QUEUE, -1, offset=0)
        self.assertEqual([review["id"] for review in negative["items"]], ["020"])
        self.assertEqual(negative["items"][0]["tags"][0]["aspect"], QUEUE)
        all_negative = self.catalog.browse(None, -1, offset=0, limit=1)
        self.assertEqual((all_negative["total"], all_negative["next_offset"]), (2, 1))
        next_page = self.catalog.browse(None, -1, offset=1, limit=1)
        self.assertEqual([review["id"] for review in next_page["items"]], ["020"])
        self.assertIsNone(next_page["next_offset"])
        self.assertFalse(self.catalog.browse(TASTE, -1)["items"])
        with self.assertRaises(ValueError):
            self.catalog.browse("unknown", -1)

    def test_detail_preserves_raw_text_and_identifiers(self):
        detail = self.catalog.detail("010")
        self.assertEqual(detail["text"], "菜好吃，但上菜太慢。")
        self.assertEqual(detail["id"], "010")
        self.assertEqual(detail["star"], "3")
        self.assertEqual(json.loads(json.dumps(detail))["aspects"][0]["aspect"], QUEUE)
        with self.assertRaises(KeyError):
            self.catalog.detail("10")

