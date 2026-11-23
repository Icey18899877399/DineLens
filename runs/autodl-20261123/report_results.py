import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch
import transformers
from sklearn.metrics import precision_recall_fscore_support
from aspect_detection.data import ASPECT_KEYS, load_splits
from aspect_detection.evidence import load_evidence_extractor

ROOT = Path("runs")
CHECKPOINT_REVISION = "8f23c25b06e129b6c986331a13d8d025a92cf0ea"
TEST = load_splits(Path("dataset"))["test"]
METHODS = ("tfidf", "textrank")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_for(run, with_sentiment):
    test_metrics = read_json(run / "test_metrics.json")
    metadata = read_json(run / "model.json")
    assert test_metrics["n_reviews"] == len(TEST.ids)
    rows = 0
    labels = []
    mention_predictions = []
    polarity_predictions = []
    with (run / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        for index, record in enumerate(csv.DictReader(stream)):
            doc, label = divmod(index, len(ASPECT_KEYS))
            assert record["id"] == TEST.ids[doc] and record["aspect"] == ASPECT_KEYS[label]
            assert int(record["gold_mentioned"]) == int(TEST.labels[doc, label])
            mention_predictions.append(int(record["mentioned"]))
            if with_sentiment:
                assert int(record["gold_sentiment"]) == int(TEST.polarities[doc, label])
                predicted = int(record["sentiment"])
                assert (predicted != -2) == bool(int(record["mentioned"]))
                labels.append(int(record["gold_sentiment"]))
                polarity_predictions.append(predicted)
            rows += 1
    assert rows == TEST.polarities.size
    result = {
        "n_reviews": test_metrics["n_reviews"],
        "n_review_aspect_pairs": rows,
        "aspect": {name: test_metrics[name] for name in ("macro_f1", "micro_f1")},
        "training": metadata,
    }
    if not with_sentiment:
        return result
    assert test_metrics["sentiment"]["support"] == int(TEST.labels.sum())
    actual = np.asarray(labels).reshape(TEST.polarities.shape) == -1
    predicted = np.asarray(polarity_predictions).reshape(TEST.polarities.shape) == -1
    end_to_end = {}
    for average in ("micro", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, average=average, zero_division=0
        )
        end_to_end[average] = {"precision": precision, "recall": recall, "f1": f1}
    result["sentiment_given_gold_mention"] = {
        name: test_metrics["sentiment"][name] for name in ("support", "macro_f1", "classes")
    }
    result["end_to_end_negative"] = {
        "gold_pairs": int(actual.sum()),
        "predicted_pairs": int(predicted.sum()),
        "true_positive_pairs": int((actual & predicted).sum()),
        **end_to_end,
    }
    return result


def evidence_diagnostics():
    extractor = load_evidence_extractor(ROOT / "aspect_baseline")
    indices = {review_id: index for index, review_id in enumerate(TEST.ids)}
    n_pairs = 0
    candidates = {"zero": 0, "one": 0, "multiple": 0}
    top1_different = 0
    completed = 0
    with Path("annotations/asap_evidence_test.csv").open(encoding="utf-8-sig", newline="") as stream:
        for item in csv.DictReader(stream):
            review_index = indices[item["id"]]
            aspect = item["aspect"]
            assert TEST.polarities[review_index, ASPECT_KEYS.index(aspect)] == -1
            review = TEST.texts[review_index]
            extractions = extractor.rank(review, aspect, top_k=2)
            candidates[("zero", "one", "multiple")[len(extractions["tfidf"])]] += 1
            for method in METHODS:
                for extraction in extractions[method]:
                    assert review[extraction["start"]:extraction["end"]] == extraction["text"]
            if extractions["tfidf"]:
                top1_different += (extractions["tfidf"][0]["start"] != extractions["textrank"][0]["start"])
            n_pairs += 1
            completed += bool(item["label"].strip())
    return {
        "n_pairs_with_gold_negative_aspect": n_pairs,
        "n_pairs_with_manual_evidence_label": completed,
        "candidate_counts": candidates,
        "top1_different": int(top1_different),
        "formal_metrics_path": (
            "runs/evidence_evaluation.json" if Path("runs/evidence_evaluation.json").is_file() else None
        ),
    }


report = {
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "scikit_learn": sklearn.__version__,
        "gpu": torch.cuda.get_device_name(),
        "pretrained_revision": CHECKPOINT_REVISION,
        "dataset_sha256": {
            name: hashlib.file_digest(Path(f"dataset/{name}.csv").open("rb"), "sha256").hexdigest()
            for name in ("train", "dev", "test")
        },
    },
    "aspect_baseline": metrics_for(ROOT / "aspect_baseline", False),
    "sentiment_baseline": metrics_for(ROOT / "sentiment_baseline", True),
    "joint_bert": metrics_for(ROOT / "joint_bert", True),
    "evidence_diagnostics": evidence_diagnostics(),
}
with (ROOT / "experiment_report.json").open("x", encoding="utf-8") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
for name in ("aspect_baseline", "sentiment_baseline", "joint_bert"):
    result = report[name]
    print(name, result["aspect"], result.get("end_to_end_negative"), flush=True)
print(report["evidence_diagnostics"], flush=True)
