import csv
import re
from collections import Counter
from pathlib import Path

from aspect_detection.data import ASPECTS, SENTIMENTS, UNMENTIONED

ASPECT_BY_KEY = {aspect.key: aspect for aspect in ASPECTS}
SHORT_NAMES = {
    "Location#Downtown": "商圈距离",
    "Location#Easy_to_find": "好不好找",
    "Service#Queue": "排队",
    "Service#Hospitality": "服务态度",
    "Service#Parking": "停车",
    "Service#Timely": "上菜速度",
    "Price#Level": "价格",
    "Price#Discount": "折扣",
    "Ambience#Space": "座位空间",
    "Food#Appearance": "卖相",
}
SENTIMENT_NAMES = {value: name for value, _, name in SENTIMENTS}
NEWLINES = re.compile(r"\\+[rn]|[\r\n]+")
POLARITY_ORDER = {-1: 0, 1: 1, 0: 2}


class ReviewCatalog:
    def __init__(self, reviews, predictions):
        self.reviews = reviews
        self.predictions = predictions
        self.by_aspect = {key: [] for key in ASPECT_BY_KEY}
        self.counts = {key: Counter() for key in ASPECT_BY_KEY}
        self.negative_reviews = 0
        for review_id, rows in predictions.items():
            self.negative_reviews += any(row["sentiment"] == -1 for row in rows.values())
            for key, row in rows.items():
                sentiment, gold = row["sentiment"], row["gold_sentiment"]
                counts = self.counts[key]
                counts[sentiment] += 1
                counts["gold_negative"] += gold == -1
                counts["false_positive"] += sentiment == -1 and gold != -1
                counts["false_negative"] += sentiment != -1 and gold == -1
                if row["mentioned"]:
                    self.by_aspect[key].append(review_id)
        for key, review_ids in self.by_aspect.items():
            review_ids.sort(key=lambda review_id: -self.predictions[review_id][key]["score"])

    @classmethod
    def from_files(cls, review_csv: Path, predictions_csv: Path):
        with review_csv.open(encoding="utf-8-sig", newline="") as stream:
            reviews = {row["id"]: (row["review"], row["star"]) for row in csv.DictReader(stream)}
        predictions = {}
        with predictions_csv.open(encoding="utf-8-sig", newline="") as stream:
            for raw in csv.DictReader(stream):
                review_id, key = raw["id"], raw["aspect"]
                if review_id not in reviews or key not in ASPECT_BY_KEY:
                    raise ValueError(f"无法关联评论预测：{review_id} / {key}")
                rows = predictions.setdefault(review_id, {})
                if key in rows:
                    raise ValueError(f"评论预测重复：{review_id} / {key}")
                sentiment = int(raw["sentiment"])
                mentioned = int(raw["mentioned"])
                if mentioned not in (0, 1) or bool(mentioned) != (sentiment in SENTIMENT_NAMES):
                    raise ValueError(f"评论预测的提及状态无效：{review_id} / {key}")
                aspect = ASPECT_BY_KEY[key]
                rows[key] = {
                    "aspect": key,
                    "group": aspect.group,
                    "name": aspect.name,
                    "mentioned": mentioned,
                    "score": float(raw["score"]),
                    "threshold": float(raw["threshold"]),
                    "sentiment": sentiment,
                    "sentiment_name": SENTIMENT_NAMES.get(sentiment, "未提及"),
                    "sentiment_scores": (
                        {name: float(raw[f"{name}_score"]) for _, name, _ in SENTIMENTS}
                        if mentioned else None
                    ),
                    "gold_sentiment": int(raw["gold_sentiment"]),
                }
        return cls({review_id: reviews[review_id] for review_id in predictions}, predictions)

    def overview(self):
        return {
            "total_reviews": len(self.reviews),
            "negative_reviews": self.negative_reviews,
            "aspects": [
                {
                    "aspect": key,
                    "name": aspect.name,
                    "label": SHORT_NAMES.get(key, aspect.name),
                    "group": aspect.group,
                    "positive": self.counts[key][1],
                    "neutral": self.counts[key][0],
                    "negative": self.counts[key][-1],
                    "gold_negative": self.counts[key]["gold_negative"],
                    "false_positive": self.counts[key]["false_positive"],
                    "false_negative": self.counts[key]["false_negative"],
                }
                for key, aspect in ASPECT_BY_KEY.items()
            ],
        }

    def browse(self, aspect=None, sentiment=None, offset=0, limit=6):
        if aspect is not None and aspect not in ASPECT_BY_KEY:
            raise ValueError("请选择有效的评价方面")
        if sentiment is not None and sentiment not in SENTIMENT_NAMES:
            raise ValueError("请选择有效的情感")
        if offset < 0 or not 1 <= limit <= 24:
            raise ValueError("评论页码无效")
        identifiers = self.by_aspect[aspect] if aspect else self.reviews
        if sentiment is not None:
            identifiers = [
                review_id for review_id in identifiers
                if (
                    self.predictions[review_id][aspect]["sentiment"] == sentiment if aspect else
                    any(row["sentiment"] == sentiment for row in self.predictions[review_id].values())
                )
            ]
        identifiers = list(identifiers)
        return {
            "items": [self._preview(review_id, aspect, sentiment) for review_id in identifiers[offset:offset + limit]],
            "total": len(identifiers),
            "next_offset": offset + limit if offset + limit < len(identifiers) else None,
        }

    def _preview(self, review_id, focus, sentiment):
        text, star = self.reviews[review_id]
        text = NEWLINES.sub(" ", text).strip()
        rows = sorted(
            (row for row in self.predictions[review_id].values() if row["sentiment"] != UNMENTIONED),
            key=lambda row: (
                row["aspect"] != focus,
                sentiment is not None and row["sentiment"] != sentiment,
                POLARITY_ORDER[row["sentiment"]],
                -row["score"],
            ),
        )
        return {
            "id": review_id,
            "preview": text[:112].rstrip() + ("…" if len(text) > 112 else ""),
            "star": star,
            "tags": [
                {"aspect": row["aspect"], "name": SHORT_NAMES.get(row["aspect"], row["name"]),
                 "sentiment": row["sentiment"]}
                for row in rows[:3]
            ],
        }

    def detail(self, review_id):
        text, star = self.reviews[review_id]
        rows = self.predictions[review_id]
        return {
            "id": review_id,
            "text": text,
            "star": star,
            "aspects": [rows[key] for key in ASPECT_BY_KEY if key in rows],
        }

