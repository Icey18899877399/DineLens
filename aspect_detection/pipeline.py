import csv
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

from .data import ASPECT_KEYS, ASPECTS, SENTIMENTS, SENTIMENT_VALUES, UNMENTIONED, load_splits
from .metrics import evaluate, evaluate_sentiment, fit_thresholds

SENTIMENT_NAMES = {UNMENTIONED: "未提及", **{value: name for value, _, name in SENTIMENTS}}


def _score_outputs(score_reviews, texts):
    result = score_reviews(texts)
    return result if isinstance(result, tuple) else (result, None)


class AspectPredictor:
    def __init__(self, score_reviews, thresholds):
        self.score_reviews = score_reviews
        self.thresholds = np.asarray(thresholds, dtype=float)

    def predict(self, review):
        if not review.strip():
            raise ValueError("请输入非空评论")
        mentions, sentiments = _score_outputs(self.score_reviews, [review])
        scores = mentions[0]
        rows = []
        for index, aspect in enumerate(ASPECTS):
            mentioned = int(scores[index] >= self.thresholds[index])
            row = {
                "aspect": aspect.key,
                "group": aspect.group,
                "name": aspect.name,
                "mentioned": mentioned,
                "score": float(scores[index]),
                "threshold": float(self.thresholds[index]),
            }
            if sentiments is not None:
                probabilities = sentiments[0, index]
                polarity = SENTIMENT_VALUES[probabilities.argmax()] if mentioned else UNMENTIONED
                row.update(
                    sentiment=polarity,
                    sentiment_name=SENTIMENT_NAMES[polarity],
                    sentiment_scores={
                        key: float(probabilities[i]) for i, (_, key, _) in enumerate(SENTIMENTS)
                    } if mentioned else None,
                )
            rows.append(row)
        return rows


def load_predictor(run: Path, device="auto") -> AspectPredictor:
    metadata = json.loads((run / "model.json").read_text(encoding="utf-8"))
    if tuple(metadata["aspect_keys"]) != ASPECT_KEYS:
        raise ValueError("模型的方面标签与当前数据定义不一致")
    with_sentiment = metadata.get("task") == "sentiment"
    if metadata["model"] == "baseline":
        model = joblib.load(run / "baseline.joblib")
        score_reviews = model.score_reviews if with_sentiment else model.predict_proba
    elif metadata["model"] == "bert":
        from .bert import load_bert

        score_reviews = load_bert(
            run / "checkpoint", device, metadata["max_length"], metadata["batch_size"], with_sentiment
        )
    else:
        raise ValueError(f"无法加载模型类型：{metadata['model']}")
    return AspectPredictor(score_reviews, metadata["thresholds"])


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _report(split, decisions, polarities):
    metrics = evaluate(split.labels, decisions)
    if polarities is not None:
        metrics["sentiment"] = evaluate_sentiment(split.polarities, polarities)
    return metrics


def _polarities(scores):
    if scores is None:
        return None
    return np.asarray(SENTIMENT_VALUES)[scores.argmax(axis=-1)]


def _export_test(run, split, scores, thresholds, sentiment_scores):
    decisions = scores >= thresholds
    polarities = _polarities(sentiment_scores)
    columns = ("id", "aspect", "group", "name", "gold_mentioned", "mentioned", "score", "threshold")
    if polarities is not None:
        columns += (
            "gold_sentiment", "sentiment", "sentiment_if_mentioned",
            *(f"{key}_score" for _, key, _ in SENTIMENTS),
        )

    with (run / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row, review_id in enumerate(split.ids):
            for index, aspect in enumerate(ASPECTS):
                mentioned = int(decisions[row, index])
                record = (
                    review_id,
                    aspect.key,
                    aspect.group,
                    aspect.name,
                    int(split.labels[row, index]),
                    mentioned,
                    float(scores[row, index]),
                    float(thresholds[index]),
                )
                if polarities is not None:
                    polarity = int(polarities[row, index])
                    record += (
                        int(split.polarities[row, index]),
                        polarity if mentioned else UNMENTIONED,
                        polarity,
                        *(float(value) for value in sentiment_scores[row, index]),
                    )
                writer.writerow(record)
    return _report(split, decisions, polarities)


def train_run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请指定新的 --output：{output}")
    with_sentiment = getattr(args, "task", "aspect") == "sentiment"
    splits = load_splits(args.dataset)
    for name, split in splits.items():
        print(f"{name}: {len(split.ids)} 条，去重移除 {split.removed_duplicates} 条", file=sys.stderr)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "aspect_keys": list(ASPECT_KEYS),
        "seed": args.seed,
        "splits": {
            name: {"reviews": len(split.ids), "removed_duplicates": split.removed_duplicates}
            for name, split in splits.items()
        },
    }
    if with_sentiment:
        metadata["task"] = "sentiment"

    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".aspect-training-") as temporary:
        run = Path(temporary) / "output"
        run.mkdir()
        train, dev, test = (splits[key] for key in ("train", "dev", "test"))
        if args.model == "baseline":
            from .baseline import train_baseline, train_sentiment_baseline

            print("正在训练 Jieba + TF-IDF + 逻辑回归…", file=sys.stderr)
            if with_sentiment:
                model = train_sentiment_baseline(train, args.seed)
                score_reviews = model.score_reviews
            else:
                model = train_baseline(train, args.seed)
                score_reviews = model.predict_proba
            joblib.dump(model, run / "baseline.joblib", compress=3)
        else:
            from .bert import load_bert, train_bert

            metadata.update(train_bert(train, dev, run / "checkpoint", args, with_sentiment))
            score_reviews = load_bert(
                run / "checkpoint", args.device, args.max_length, args.batch_size, with_sentiment
            )

        dev_scores, dev_sentiment = _score_outputs(score_reviews, dev.texts)
        thresholds = fit_thresholds(dev.labels, dev_scores)
        metadata["thresholds"] = thresholds.tolist()
        test_scores, test_sentiment = _score_outputs(score_reviews, test.texts)
        metrics = _export_test(run, test, test_scores, thresholds, test_sentiment)
        metrics["model"] = args.model
        _write_json(run / "model.json", metadata)
        _write_json(
            run / "dev_metrics.json", _report(dev, dev_scores >= thresholds, _polarities(dev_sentiment))
        )
        _write_json(run / "test_metrics.json", metrics)
        del score_reviews
        run.rename(output)
    return metrics

