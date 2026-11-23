import csv
from pathlib import Path
import tempfile
import unittest

from sklearn.feature_extraction.text import TfidfVectorizer

from aspect_detection.baseline import tokenize
from test_aspect_detection import KEYS, write_split

REVIEW = "🍜菜很好吃，点完菜等了四十分钟才上桌。服务员很客气。"


def extractor():
    from aspect_detection.evidence import EvidenceExtractor

    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    vectorizer.fit([
        REVIEW,
        "不过上菜太慢，还催了三次菜，最后上菜太慢。",
        "等位排队等了四十分钟才入座。",
        "房间一点也不吵，环境很安静。",
    ])
    return EvidenceExtractor(vectorizer)


class TestEvidence(unittest.TestCase):
    def test_target_and_offsets_avoid_cross_aspect_evidence(self):
        result = extractor().rank(REVIEW, "Service#Timely")
        for ranked in result.values():
            top = ranked[0]
            self.assertEqual(top["text"], "点完菜等了四十分钟才上桌")
            self.assertEqual(REVIEW[top["start"]:top["end"]], top["text"])
            self.assertEqual(top["start"], 6)
            self.assertEqual(top["start_utf16"], 7)
        self.assertEqual(extractor().rank(REVIEW, "Service#Queue"), {"tfidf": [], "textrank": []})

    def test_same_candidates_and_graph_scores_across_methods(self):
        text = "第一轮上菜太慢，第二轮上菜也太慢，最后一轮上菜太慢。"
        result = extractor().rank(text, "Service#Timely", top_k=9)
        self.assertEqual(len(result["tfidf"]), 3)
        self.assertEqual(len(result["textrank"]), 3)
        self.assertEqual(
            {row["start"] for row in result["tfidf"]},
            {row["start"] for row in result["textrank"]},
        )
        self.assertAlmostEqual(sum(row["score"] for row in result["textrank"]), 1.0)

    def test_positive_negation_returns_no_evidence(self):
        result = extractor().rank("环境一点也不吵，菜也不难吃。", "Ambience#Noise")
        self.assertFalse(result["tfidf"])
        self.assertFalse(result["textrank"])

    def test_comma_does_not_join_praise_to_another_aspects_complaint(self):
        text = "菜的味道很好，服务员态度太差。"
        result = extractor().rank(text, "Food#Taste")
        self.assertEqual(result, {"tfidf": [], "textrank": []})

    def test_complaint_matches_its_own_aspect(self):
        model = extractor()
        examples = [
            ("找不到车位。", "Service#Parking", "Location#Easy_to_find"),
            ("桌子上太脏。", "Ambience#Sanitary", "Ambience#Space"),
            ("菜的分量太少。", "Food#Portion", "Food#Taste"),
            ("菜的分量太少而且味道太咸。", "Food#Portion", None),
            ("停车场没有车位。", "Service#Parking", None),
            ("排队时间太久。", "Service#Queue", None),
            ("交通很不方便。", "Location#Transportation", None),
            ("装修很破旧。", "Ambience#Decoration", None),
        ]
        for review, correct, unrelated in examples:
            with self.subTest(review=review):
                for ranked in model.rank(review, correct).values():
                    self.assertEqual(ranked[0]["text"], review[:-1])
                if unrelated:
                    self.assertEqual(model.rank(review, unrelated), {"tfidf": [], "textrank": []})

    def test_escaped_asap_line_breaks_preserve_source_offsets(self):
        text = r"头一回过来\\r\\n这家装修挺好\\n不过价格太贵了\\n没别的意见"
        for ranked in extractor().rank(text, "Price#Level").values():
            top = ranked[0]
            self.assertEqual(top["text"], "不过价格太贵了")
            self.assertEqual(text[top["start"]:top["end"]], top["text"])

    def test_utf16_offsets_and_adjacent_clause(self):
        text = "😀先说优点。服务员，半天也没人理。"
        result = extractor().rank(text, "Service#Hospitality")
        for ranked in result.values():
            top = ranked[0]
            self.assertEqual(top["text"], "服务员，半天也没人理")
            self.assertEqual(text[top["start"]:top["end"]], top["text"])
            units = text.encode("utf-16-le")
            self.assertEqual(units[2*top["start_utf16"]:2*top["end_utf16"]].decode("utf-16-le"), top["text"])

    def test_attach_only_to_predicted_negative_aspects(self):
        predictions = [
            {"aspect": "Service#Timely", "mentioned": 1, "sentiment": -1},
            {"aspect": "Food#Taste", "mentioned": 1, "sentiment": 1},
            {"aspect": "Ambience#Noise", "mentioned": 0, "sentiment": -2},
        ]
        result = extractor().annotate_predictions(REVIEW, predictions)
        self.assertEqual(result[0]["evidence"]["tfidf"]["text"], "点完菜等了四十分钟才上桌")
        self.assertIsNone(result[1]["evidence"])
        self.assertIsNone(result[2]["evidence"])
        self.assertNotIn("evidence", predictions[0])


class TestEvidenceEvaluation(unittest.TestCase):
    def test_annotation_template_and_evaluation_keep_pending_out(self):
        from aspect_detection.evidence_eval import evaluate_annotations, sample_annotations

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            labels = [-2] * len(KEYS)
            labels[KEYS.index("Service#Timely")] = -1
            write_split(folder, "train", [("t", "一条训练评论。", labels)])
            write_split(folder, "dev", [("d", "一条验证评论。", labels)])
            write_split(folder, "test", [
                ("01", REVIEW, labels),
                ("02", "这次不满意。", labels),
                ("03", "上菜太慢，要催了才来。", labels),
            ])
            path = folder / "annotations.csv"
            info = sample_annotations(folder, "test", path, per_aspect=3, seed=42)
            self.assertEqual(info["rows"], 3)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            self.assertTrue(all(not row["label"] for row in rows))
            for row in rows:
                if row["id"] == "01":
                    row["label"] = "evidence"
                    row["evidence_1"] = "点完菜等了四十分钟才上桌"
                elif row["id"] == "02":
                    row["label"] = "none"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            report = evaluate_annotations(extractor(), folder, "test", path)
            self.assertEqual((report["annotated"], report["pending"]), (2, 1))
            for metrics in report["methods"].values():
                self.assertEqual(metrics["hit_at_1"], 1.0)
                self.assertEqual(metrics["span_f1"], 1.0)
                self.assertEqual(metrics["false_highlight_rate"], 0.0)

