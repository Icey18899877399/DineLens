import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Aspect:
    key: str
    group: str
    name: str


ASPECTS = (
    Aspect("Location#Transportation", "位置", "交通便利"),
    Aspect("Location#Downtown", "位置", "距商圈远近"),
    Aspect("Location#Easy_to_find", "位置", "是否容易找到"),
    Aspect("Service#Queue", "服务", "排队时间"),
    Aspect("Service#Hospitality", "服务", "服务人员态度"),
    Aspect("Service#Parking", "服务", "停车便利"),
    Aspect("Service#Timely", "服务", "点菜／上菜速度"),
    Aspect("Price#Level", "价格", "价格水平"),
    Aspect("Price#Cost_effective", "价格", "性价比"),
    Aspect("Price#Discount", "价格", "折扣力度"),
    Aspect("Ambience#Decoration", "环境", "装修"),
    Aspect("Ambience#Noise", "环境", "噪声"),
    Aspect("Ambience#Space", "环境", "就餐空间"),
    Aspect("Ambience#Sanitary", "环境", "卫生"),
    Aspect("Food#Portion", "菜品", "分量"),
    Aspect("Food#Taste", "菜品", "口味"),
    Aspect("Food#Appearance", "菜品", "外观"),
    Aspect("Food#Recommend", "菜品", "推荐程度"),
)
ASPECT_KEYS = tuple(aspect.key for aspect in ASPECTS)
SPLITS = ("train", "dev", "test")
UNMENTIONED = -2
SENTIMENTS = ((-1, "negative", "负面"), (0, "neutral", "中性"), (1, "positive", "正面"))
SENTIMENT_VALUES = tuple(value for value, _, _ in SENTIMENTS)
POLARITY_VALUES = {str(value) for value in (UNMENTIONED, *SENTIMENT_VALUES)}
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Split:
    ids: tuple[str, ...]
    texts: tuple[str, ...]
    labels: np.ndarray
    removed_duplicates: int
    polarities: np.ndarray


def _read_rows(path):
    required = {"id", "review", *ASPECT_KEYS}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} 缺少列：{', '.join(sorted(missing))}")
        for row in reader:
            review_id = row["id"]
            review = row["review"]
            if not review_id or not review or not review.strip():
                raise ValueError(f"{path.name} 第 {reader.line_num} 行缺少评论 ID 或正文")
            values = [row[key] for key in ASPECT_KEYS]
            if not set(values) <= POLARITY_VALUES:
                raise ValueError(f"{path.name} 第 {reader.line_num} 行包含无效的 ASAP 标签")
            yield review_id, review, [int(value) for value in values]


def load_splits(folder: Path) -> dict[str, Split]:
    """Read ASAP labels and protect the official test texts from leakage."""
    results = {}
    seen_ids = set()
    reserved_texts = set()
    for split in reversed(SPLITS):
        ids, texts, polarities = [], [], []
        duplicates = 0
        for review_id, review, row_polarities in _read_rows(folder / f"{split}.csv"):
            if review_id in seen_ids:
                raise ValueError(f"ASAP 评论 ID 重复：{review_id}")
            seen_ids.add(review_id)
            key = WHITESPACE.sub("", unicodedata.normalize("NFKC", review)).casefold()
            if split != "test" and key in reserved_texts:
                duplicates += 1
                continue
            reserved_texts.add(key)
            ids.append(review_id)
            texts.append(review)
            polarities.append(row_polarities)
        if not ids:
            raise ValueError(f"{split}.csv 没有可用的评论")
        polarity_matrix = np.array(polarities, dtype=np.int8)
        labels = (polarity_matrix != UNMENTIONED).astype(np.int8)
        results[split] = Split(tuple(ids), tuple(texts), labels, duplicates, polarity_matrix)
    return {split: results[split] for split in SPLITS}

