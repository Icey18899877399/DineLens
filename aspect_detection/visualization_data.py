"""Read the saved ASAP test results for the web and report figures."""

import json
from pathlib import Path

from .data import ASPECTS, SENTIMENTS

RESULT_NAMES = (("sentiment_baseline", "传统基线"), ("joint_bert", "联合 BERT"))
SENTIMENT_LABELS = [name for _, _, name in SENTIMENTS]
DEMO_REVIEW = "菜很好吃，但是等了四十分钟才上菜，前台态度很差。"
DEMO_ASPECT = "Service#Timely"


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _confusion(sentiment):
    counts = sentiment["confusion_matrix"]
    support = [sum(row) for row in counts]
    if sum(support) != sentiment["support"]:
        raise ValueError("情感矩阵合计与评估样本数不一致")
    return {
        "counts": counts,
        "row_support": support,
        "rates": [[value / n if n else None for value in row] for row, n in zip(counts, support)],
    }


def _example(path):
    row = next(item for item in _read_json(path) if item["aspect"] == DEMO_ASPECT)
    for method, span in row["evidence"].items():
        if not span or DEMO_REVIEW[span["start"]:span["end"]] != span["text"]:
            raise ValueError(f"示例评论的 {method} 证据位置不匹配")
    return {"text": DEMO_REVIEW, "aspect": row["name"], "methods": row["evidence"]}


def load_visualization_data(results: Path):
    report = _read_json(results / "experiment_report.json")
    models = []
    aspect_scores = []
    n_reviews = report["joint_bert"]["n_reviews"]
    n_pairs = report["joint_bert"]["n_review_aspect_pairs"]
    for folder, name in RESULT_NAMES:
        saved = report[folder]
        metrics = _read_json(results / folder / "test_metrics.json")
        if metrics["n_reviews"] != n_reviews or saved["n_review_aspect_pairs"] != n_pairs:
            raise ValueError("两个模型的测试样本数不一致")
        models.append({
            "name": name,
            "aspect_macro_f1": metrics["macro_f1"],
            "sentiment_macro_f1": metrics["sentiment"]["macro_f1"],
            "end_to_end_negative_f1": saved["end_to_end_negative"]["micro"]["f1"],
            "confusion": _confusion(metrics["sentiment"]),
        })
        aspect_scores.append({row["aspect"]: row for row in metrics["aspects"]})

    if models[0]["confusion"]["row_support"] != models[1]["confusion"]["row_support"]:
        raise ValueError("两个模型的人工情感标签支持量不一致")
    aspects = []
    for aspect in ASPECTS:
        baseline, bert = (scores[aspect.key] for scores in aspect_scores)
        if baseline["support"] != bert["support"]:
            raise ValueError(f"{aspect.name} 的人工提及数不一致")
        aspects.append({
            "aspect": aspect.key,
            "name": aspect.name,
            "group": aspect.group,
            "support": bert["support"],
            "baseline": {key: baseline[key] for key in ("f1", "precision", "recall")},
            "bert": {key: bert[key] for key in ("f1", "precision", "recall")},
        })

    evidence = report["evidence_diagnostics"]
    if sum(evidence["candidate_counts"].values()) != evidence["n_pairs_with_gold_negative_aspect"]:
        raise ValueError("证据候选构成与评估样本数不一致")
    return {
        "test_reviews": n_reviews,
        "test_pairs": n_pairs,
        "sentiment_labels": SENTIMENT_LABELS,
        "sentiment_support": sum(models[1]["confusion"]["row_support"]),
        "models": models,
        "aspects": aspects,
        "evidence": {**evidence, "example": _example(results / "bert_evidence_demo.json")},
    }

