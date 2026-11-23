import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.validation import check_is_fitted

from .data import ASPECTS

METHODS = ("tfidf", "textrank")
ASPECT_BY_KEY = {aspect.key: aspect for aspect in ASPECTS}

# The left column anchors the subject. The right column includes expressions
# that can identify both the aspect and the complaint without a subject noun.
CLUES = {
    "Location#Transportation": (
        "交通 公交 地铁 打车 车站 地铁口 通勤", "交通不便 交通不方便 打不到车 没有公交 离地铁远"
    ),
    "Location#Downtown": (
        "商圈 市中心 市区 商业区 地段 位置 路程", "不在商圈 离市区太远 位置偏僻 离市中心远 太偏了"
    ),
    "Location#Easy_to_find": (
        "地址 位置 路标 招牌 门牌 导航 店面 门口 找到", "找不到 找不着 不好找 难找 找了半天 导航不准"
    ),
    "Service#Queue": (
        "排队 等位 取号 叫号 入座 排号 等座 等桌", "等位太久 排队太久 排队很久 排队太长 排了半天 排了很久 插队 叫号太慢"
    ),
    "Service#Hospitality": (
        "服务 服务员 店员 态度 老板 接待 前台 服务人员",
        "态度差 态度恶劣 态度不好 冷脸 甩脸 没人理 爱答不理 不耐烦 不热情 服务差",
    ),
    "Service#Parking": (
        "停车 车位 停车场 停车费 泊车", "没车位 没有车位 找不到车位 车位少 停车困难 不好停车 停车不方便"
    ),
    "Service#Timely": (
        "点菜 点单 下单 点完菜 上菜 出菜 催菜 催了 上桌 菜上 菜还没上",
        "上菜太慢 出菜太慢 上菜慢 半天不上菜 迟迟不上 催了几次 漏上 漏单",
    ),
    "Price#Level": (
        "价格 价钱 人均 价位 收费 菜价 费用 贵", "价格虚高 价格太高 价格偏高 价格贵 太贵 过于贵 偏贵 乱收费 多收费"
    ),
    "Price#Cost_effective": (
        "性价比 划算 值不值 这个价 这个价格 物有所值 预算", "不划算 不值这个价 不值这价 不值这个价格 性价比低 性价比不高"
    ),
    "Price#Discount": (
        "折扣 优惠 优惠券 活动 团购 满减 代金券 打折", "不能用券 优惠不给用 折扣少 优惠少 不给打折 不参加活动"
    ),
    "Ambience#Decoration": (
        "装修 装潢 装饰 格调 布置 陈设 装修风格", "装修陈旧 装修破旧 装修老旧 装修简陋 装潢粗糙"
    ),
    "Ambience#Noise": (
        "噪声 噪音 嘈杂 吵 声音 音量 隔音 音乐", "太吵 特别吵 有点吵 很吵 吵得 太嘈杂 隔音差 说话听不清"
    ),
    "Ambience#Space": (
        "空间 店里 座位 桌距 桌子 间距 走道 过道 店面 坐", "太挤 拥挤 空间狭小 座位太挤 桌子太近 走不开"
    ),
    "Ambience#Sanitary": (
        "卫生 清洁 桌面 餐具 碗筷 地面 厕所 桌子", "不干净 油腻腻 桌面油乎乎 卫生差 卫生堪忧 太脏 有蟑螂 有虫"
    ),
    "Food#Portion": (
        "分量 份量 量 份 盘子 份额 够吃", "量太少 分量少 份量少 量少 不够吃 份小 一点点就没了"
    ),
    "Food#Taste": (
        "口味 味道 口感 调味 咸 淡 油 食材 热度 菜", "难吃 不好吃 不太好吃 不算太好吃 不怎么好吃 不新鲜 太咸 太淡 太油 菜凉了 齁咸 涩口"
    ),
    "Food#Appearance": (
        "外观 卖相 摆盘 看相 颜色 色泽 造型", "卖相差 不好看 摆盘粗糙 颜色发黑 摆盘很乱 色泽暗淡"
    ),
    "Food#Recommend": (
        "推荐 值得点 踩雷 避雷 必点 回购", "不推荐 不建议点 不值得点 踩雷 避雷 不值得尝 不想再点"
    ),
}

ASPECT_TERMS = {
    key: frozenset((anchors + " " + complaints).split())
    for key, (anchors, complaints) in CLUES.items()
}
ALL_TERMS = frozenset(term for terms in ASPECT_TERMS.values() for term in terms if len(term) > 1)


def _clue_pattern(column):
    owners = {}
    for aspect, clues in CLUES.items():
        for term in clues[column].split():
            owners.setdefault(term, set()).add(aspect)
    pattern = re.compile("|".join(map(re.escape, sorted(owners, key=lambda term: (-len(term), term)))))
    return pattern, owners


ANCHOR_PATTERN, ANCHOR_OWNERS = _clue_pattern(0)
SPECIFIC_PATTERN, SPECIFIC_OWNERS = _clue_pattern(1)
SENTENCE_BREAK = re.compile(r"[。！？!?；;\r\n]+|\\+[nr]")

CLAUSE_BREAK = re.compile(r"(?<!\d)[，,]|[，,](?!\d)|(?<!不)但(?:是)?|然而")
WEAK_BREAK = re.compile(r"\s*[，,]\s*")
WHITESPACE = re.compile(r"\s+")
NEGATED_COMPLAINT = re.compile(
    r"(?:并不|并没有|没有|不|没)(?:算|太|怎么|那么)?"
    r"(?:难找|难吃|难停|贵|吵|慢|差|挤|脏|冷|硬|久|远|咸|淡|油|偏|少)"
)
NEGATIVE = re.compile(
    r"不好|不佳|不行|不满意|太(?:差|慢|久|贵|吵|小|少|挤|咸|淡|油|远|偏|冷|硬|脏)|"
    r"(?:很|特别|非常|有点|实在|真的)(?:差|慢|贵|吵|挤|脏|少|咸|淡|油|远|偏)|"
    r"不方便|不便|破旧|陈旧|简陋|差劲|难受|糟糕|失望|没人理|迟迟|糊弄|半天(?:也)?没|"
    r"等(?:了|过了)?[^，,。]{0,12}(?:分钟|小时)[^，,。]{0,8}才"
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int


def _trimmed(review, start, end):
    source = review[start:end]
    return Span(start + len(source) - len(source.lstrip()), start + len(source.rstrip()))


def _segments(review, start, end, separator_pattern):
    cursor = start
    for separator in separator_pattern.finditer(review, start, end):
        span = _trimmed(review, cursor, separator.start())
        if span.start < span.end:
            yield span
        cursor = separator.end()
    span = _trimmed(review, cursor, end)
    if span.start < span.end:
        yield span


def _wording(text):
    normalized = WHITESPACE.sub("", unicodedata.normalize("NFKC", text))
    return NEGATED_COMPLAINT.sub("", normalized)


def _anchor_priority(anchor, complaint):
    distance = max(complaint.start() - anchor.end(), anchor.start() - complaint.end(), 0)
    return distance, anchor.start() > complaint.start(), -len(anchor.group())


def _local_complaints(wording, anchors):
    specific_matches = list(SPECIFIC_PATTERN.finditer(wording))
    specific_aspects = {aspect for match in specific_matches for aspect in SPECIFIC_OWNERS[match.group()]}
    inferred_aspects = set()
    unassigned = False
    for complaint in NEGATIVE.finditer(wording):
        # When an explicit phrase claims these characters, a broad adjective
        # inside it cannot also be used for an unrelated subject.
        if any(match.start() < complaint.end() and complaint.start() < match.end()
               for match in specific_matches):
            continue
        if anchors:
            closest = min(anchors, key=lambda anchor: _anchor_priority(anchor, complaint))
            owners = ANCHOR_OWNERS[closest.group()]
            if len(owners) == 1:
                inferred_aspects.update(owners)
        else:
            unassigned = True
    return specific_aspects, inferred_aspects, unassigned


def _candidates(review, aspect):
    other_terms = ALL_TERMS - ASPECT_TERMS[aspect]
    for sentence in _segments(review, 0, len(review), SENTENCE_BREAK):
        previous = None
        previous_anchor_only = False
        for span in _segments(review, sentence.start, sentence.end, CLAUSE_BREAK):
            wording = _wording(review[span.start:span.end])
            anchors = list(ANCHOR_PATTERN.finditer(wording))
            anchored = any(aspect in ANCHOR_OWNERS[anchor.group()] for anchor in anchors)
            specific, inferred, unassigned = _local_complaints(wording, anchors)
            qualifies = aspect in specific or aspect in inferred
            if (not anchored and previous_anchor_only
                    and WEAK_BREAK.fullmatch(review[previous.end:span.start])
                    and (aspect in specific
                         or (unassigned and not specific and not any(term in wording for term in other_terms)))):
                yield Span(previous.start, span.end)
            elif qualifies:
                yield span
            previous = span
            previous_anchor_only = anchored and not (specific or inferred)


def _textrank(features, relevance):
    graph_weights = cosine_similarity(features)
    np.fill_diagonal(graph_weights, 0)
    graph = nx.from_numpy_array(np.clip(graph_weights, 0, 1))
    prior = dict(enumerate(relevance + 1e-12))
    ranking = nx.pagerank(graph, alpha=0.85, personalization=prior, max_iter=200, tol=1e-8)
    return np.array([ranking[index] for index in range(len(relevance))])


class EvidenceExtractor:
    def __init__(self, vectorizer):
        check_is_fitted(vectorizer, "vocabulary_")
        self.vectorizer = vectorizer
        queries = [" ".join((ASPECT_BY_KEY[key].name, *CLUES[key])) for key in ASPECT_BY_KEY]
        vectors = vectorizer.transform(queries)
        self.query_vectors = {key: vectors[index] for index, key in enumerate(ASPECT_BY_KEY)}

    def rank(self, review, aspect, top_k=1):
        if aspect not in ASPECT_BY_KEY:
            raise ValueError(f"未知的 ASAP 方面：{aspect}")
        if not review.strip():
            raise ValueError("请输入非空评论")
        if top_k < 1:
            raise ValueError("top_k 必须为正整数")

        spans = list(_candidates(review, aspect))
        if not spans:
            return {method: [] for method in METHODS}

        fragments = [review[span.start:span.end] for span in spans]
        features = self.vectorizer.transform(fragments)
        relevance = cosine_similarity(features, self.query_vectors[aspect]).ravel().clip(0, 1)
        rankings = (relevance, _textrank(features, relevance))
        result = {}
        for method, scores in zip(METHODS, rankings):
            order = sorted(
                range(len(spans)),
                key=lambda i: (-scores[i], spans[i].end - spans[i].start, spans[i].start),
            )
            result[method] = [
                {
                    "text": fragments[i],
                    "start": spans[i].start,
                    "end": spans[i].end,
                    "start_utf16": len(review[:spans[i].start].encode("utf-16-le")) // 2,
                    "end_utf16": len(review[:spans[i].end].encode("utf-16-le")) // 2,
                    "score": float(scores[i]),
                }
                for i in order[:top_k]
            ]
        return result

    def annotate_predictions(self, review, predictions):
        if any("sentiment" not in row for row in predictions):
            raise ValueError("证据联动需要功能 2 的情感模型")
        rows = []
        for row in predictions:
            evidence = None
            if row["mentioned"] and row["sentiment"] == -1:
                ranked = self.rank(review, row["aspect"])
                evidence = {
                    method: fragments[0] if fragments else None for method, fragments in ranked.items()
                }
            rows.append({**row, "evidence": evidence})
        return rows


def load_evidence_extractor(run: Path):
    model = joblib.load(run / "baseline.joblib")
    pipeline = getattr(model, "mention_pipeline", model)
    return EvidenceExtractor(pipeline["tfidf"])

