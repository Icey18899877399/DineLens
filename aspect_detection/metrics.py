import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_recall_fscore_support

from .data import ASPECTS, SENTIMENTS, SENTIMENT_VALUES, UNMENTIONED


def fit_thresholds(gold, scores):
    """Choose one F1 threshold per aspect using validation scores."""
    thresholds = np.full(len(ASPECTS), 0.5)
    for index, truth in enumerate(gold.T):
        if not truth.any():
            continue
        precision, recall, candidates = precision_recall_curve(truth, scores[:, index])
        sums = precision[:-1] + recall[:-1]
        f1 = np.divide(
            2 * precision[:-1] * recall[:-1], sums, out=np.zeros_like(sums), where=sums > 0
        )
        best = candidates[np.isclose(f1, f1.max())]
        thresholds[index] = min(best, key=lambda value: (abs(value - 0.5), -value))
    return thresholds


def evaluate(gold, predicted):
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, average=None, zero_division=0
    )
    return {
        "n_reviews": len(gold),
        "macro_f1": float(f1.mean()),
        "micro_f1": float(f1_score(gold, predicted, average="micro", zero_division=0)),
        "aspects": [
            {
                "aspect": aspect.key,
                "group": aspect.group,
                "name": aspect.name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, aspect in enumerate(ASPECTS)
        ],
    }



def _sentiment_report(gold, predicted):
    if len(gold):
        precision, recall, f1, support = precision_recall_fscore_support(
            gold, predicted, labels=SENTIMENT_VALUES, average=None, zero_division=0
        )
        matrix = confusion_matrix(gold, predicted, labels=SENTIMENT_VALUES)
    else:
        n_classes = len(SENTIMENTS)
        precision, recall, f1, support = np.zeros((4, n_classes))
        matrix = np.zeros((n_classes, n_classes), dtype=int)
    return {
        "support": int(len(gold)),
        "macro_f1": float(f1.mean()) if len(gold) else None,
        "classes": [
            {
                "value": value,
                "key": key,
                "name": name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, (value, key, name) in enumerate(SENTIMENTS)
        ],
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_sentiment(gold, predicted):
    """Score three-class predictions only where the human annotators marked a mention."""
    mask = gold != UNMENTIONED
    metrics = _sentiment_report(gold[mask], predicted[mask])
    metrics["aspects"] = []
    for index, aspect in enumerate(ASPECTS):
        selected = mask[:, index]
        report = _sentiment_report(gold[selected, index], predicted[selected, index])
        report.update(aspect=aspect.key, group=aspect.group, name=aspect.name)
        metrics["aspects"].append(report)
    return metrics
