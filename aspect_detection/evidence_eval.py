import csv
import random
from pathlib import Path

import numpy as np

from .data import ASPECTS, load_splits
from .evidence import ASPECT_BY_KEY, METHODS, Span

ANNOTATION_COLUMNS = ("id", "aspect", "group", "name", "review", "label", "evidence_1", "evidence_2")


def _display_review(review):
    # Keep operator-prefixed reviews as literal text when opening the CSV in Excel.
    if review.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + review
    return review


def sample_annotations(dataset, split_name, output, per_aspect=20, seed=42):
    split = load_splits(dataset)[split_name]
    generator = random.Random(seed)
    selections = []
    counts = {}
    for index, aspect in enumerate(ASPECTS):
        pool = np.flatnonzero(split.polarities[:, index] == -1).tolist()
        chosen = generator.sample(pool, min(per_aspect, len(pool)))
        counts[aspect.key] = len(chosen)
        selections.extend((row, aspect) for row in chosen)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(ANNOTATION_COLUMNS)
        for row, aspect in selections:
            writer.writerow(
                (split.ids[row], aspect.key, aspect.group, aspect.name,
                 _display_review(split.texts[row]), "", "", "")
            )
    return {"output": str(output.resolve()), "split": split_name, "rows": len(selections), "aspects": counts}


def _references(review, phrases, location):
    spans = []
    for phrase in phrases:
        start = review.find(phrase)
        if start < 0 or review.find(phrase, start + 1) >= 0:
            raise ValueError(f"{location} 的证据必须是原文中可唯一定位的连续片段：{phrase}")
        spans.append(Span(start, start + len(phrase)))
    return spans


def _read_annotations(dataset, split_name, path):
    split = load_splits(dataset)[split_name]
    indices = {review_id: row for row, review_id in enumerate(split.ids)}
    aspect_indices = {aspect.key: index for index, aspect in enumerate(ASPECTS)}
    seen = set()
    annotated = []
    pending = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "aspect", "label", "evidence_1", "evidence_2"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("标注 CSV 缺少 id、aspect、label、evidence_1 或 evidence_2")
        for row in reader:
            location = f"标注表第 {reader.line_num} 行"
            review_id = (row["id"] or "").strip()
            aspect_key = (row["aspect"] or "").strip()
            key = (review_id, aspect_key)
            if key in seen:
                raise ValueError(f"{location} 的评论 ID 与方面组合重复")
            seen.add(key)
            if (review_id not in indices or aspect_key not in aspect_indices
                    or split.polarities[indices[review_id], aspect_indices[aspect_key]] != -1):
                raise ValueError(f"{location} 未对应 {split_name} 中人工标注的负面方面")
            review = split.texts[indices[review_id]]
            label = (row["label"] or "").strip().lower()
            phrases = [(row[column] or "").strip() for column in ("evidence_1", "evidence_2")]
            phrases = [phrase for phrase in phrases if phrase]
            if not label and not phrases:
                pending += 1
                continue
            if label == "evidence" and phrases:
                spans = _references(review, phrases, location)
            elif label == "none" and not phrases:
                spans = []
            else:
                raise ValueError(f"{location} 请填写 evidence 并提供原文证据，或填写 none 并留空证据")
            annotated.append((review_id, aspect_key, review, spans))
    if not annotated:
        raise ValueError("标注表尚无已完成样本，填写后才能计算证据指标")
    return annotated, pending


def _score_evidence(prediction, references):
    if not references:
        return {"hit_at_1": None, "span_f1": None, "false_highlight": prediction is not None}
    if prediction is None:
        return {"hit_at_1": False, "span_f1": 0.0, "false_highlight": None}
    start, end = prediction["start"], prediction["end"]
    best_f1 = 0.0
    hit = False
    for span in references:
        overlap = max(0, min(end, span.end) - max(start, span.start))
        reference_length = span.end - span.start
        score = 2 * overlap / (end - start + reference_length)
        best_f1 = max(best_f1, score)
        hit |= overlap == reference_length and score >= 0.5
    return {"hit_at_1": hit, "span_f1": best_f1, "false_highlight": None}


def _metrics(rows, method):
    results = [row["methods"][method] for row in rows]
    answerable = [result for result in results if result["hit_at_1"] is not None]
    no_span = [result for result in results if result["false_highlight"] is not None]
    return {
        "answerable": len(answerable),
        "no_evidence": len(no_span),
        "hit_at_1": sum(row["hit_at_1"] for row in answerable) / len(answerable) if answerable else None,
        "span_f1": sum(row["span_f1"] for row in answerable) / len(answerable) if answerable else None,
        "false_highlight_rate": (
            sum(row["false_highlight"] for row in no_span) / len(no_span) if no_span else None
        ),
    }


def evaluate_annotations(extractor, dataset, split_name, annotation_path):
    annotations, pending = _read_annotations(dataset, split_name, annotation_path)
    rows = []
    by_aspect = {key: [] for key in ASPECT_BY_KEY}
    for review_id, aspect, review, spans in annotations:
        ranked = extractor.rank(review, aspect)
        methods = {}
        for method, predictions in ranked.items():
            top = predictions[0] if predictions else None
            methods[method] = {"prediction": top, **_score_evidence(top, spans)}
        result = {
            "id": review_id,
            "aspect": aspect,
            "review": review,
            "references": [
                {"text": review[span.start:span.end], "start": span.start, "end": span.end}
                for span in spans
            ],
            "methods": methods,
        }
        rows.append(result)
        by_aspect[aspect].append(result)

    return {
        "split": split_name,
        "annotated": len(rows),
        "pending": pending,
        "methods": {method: _metrics(rows, method) for method in METHODS},
        "aspects": [
            {
                "aspect": aspect.key,
                "group": aspect.group,
                "name": aspect.name,
                "methods": {method: _metrics(by_aspect[aspect.key], method) for method in METHODS},
            }
            for aspect in ASPECTS
        ],
        "examples": rows,
    }

