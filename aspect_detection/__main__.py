import argparse
import json
from pathlib import Path

from .pipeline import load_predictor, train_run

ROOT = Path(__file__).resolve().parents[1]
DEVICE_OPTIONS = ("auto", "cuda", "cpu")
TFIDF_RUN = ROOT / "runs" / "aspect_detection" / "baseline"


def _positive_int(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def main():
    parser = argparse.ArgumentParser(description="ASAP 餐饮评论的方面识别、情感判断和负面证据抽取")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="训练模型，在验证集选阈值并导出测试结果")
    train.add_argument("--task", choices=("aspect", "sentiment"), default="aspect")
    train.add_argument("--model", choices=("baseline", "bert"), default="baseline")
    train.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    train.add_argument("--output", type=Path, help="新的运行目录，按任务默认为 runs/aspect_detection 或 runs/aspect_sentiment")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--checkpoint", default="google-bert/bert-base-chinese", help="BERT 模型 ID 或本地目录")
    train.add_argument("--device", choices=DEVICE_OPTIONS, default="auto")
    train.add_argument("--epochs", type=_positive_int, default=3)
    train.add_argument("--batch-size", type=_positive_int, default=8)
    train.add_argument("--max-length", type=_positive_int, default=512)
    train.add_argument("--learning-rate", type=float, default=2e-5)

    predict = commands.add_parser("predict", help="对单条评论预测 18 个方面并输出 JSON")
    predict.add_argument("--run", type=Path, required=True)
    predict.add_argument("--text", required=True)
    predict.add_argument("--device", choices=DEVICE_OPTIONS, default="auto")
    predict.add_argument("--evidence", action="store_true", help="对功能 2 识别的负面方面附加原文证据")
    predict.add_argument("--tfidf-run", type=Path, default=TFIDF_RUN)

    evidence = commands.add_parser("evidence", help="给定方面，对比两种原文证据排序")
    evidence.add_argument("--text", required=True)
    evidence.add_argument("--aspect", required=True, help="ASAP 方面键，例如 Service#Timely")
    evidence.add_argument("--top-k", type=_positive_int, default=1)
    evidence.add_argument("--tfidf-run", type=Path, default=TFIDF_RUN)

    sample = commands.add_parser("evidence-sample", help="抽取 ASAP 负面样本并生成空白补标表")
    sample.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    sample.add_argument("--split", choices=("dev", "test"), default="test")
    sample.add_argument("--per-aspect", type=_positive_int, default=20)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--output", type=Path, required=True)

    evaluation = commands.add_parser("evidence-evaluate", help="用人工证据标注评估两种排序")
    evaluation.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    evaluation.add_argument("--split", choices=("dev", "test"), default="test")
    evaluation.add_argument("--annotations", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--tfidf-run", type=Path, default=TFIDF_RUN)

    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.output is None:
                folder = "aspect_sentiment" if args.task == "sentiment" else "aspect_detection"
                args.output = ROOT / "runs" / folder / args.model
            metrics = train_run(args)
            result = {
                "output": str(args.output.resolve()),
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
            }
            if args.task == "sentiment":
                result["sentiment_macro_f1"] = metrics["sentiment"]["macro_f1"]
                result["sentiment_negative_f1"] = metrics["sentiment"]["classes"][0]["f1"]
        elif args.command == "predict":
            result = load_predictor(args.run, args.device).predict(args.text)
            if args.evidence:
                from .evidence import load_evidence_extractor

                extractor = load_evidence_extractor(args.tfidf_run)
                result = extractor.annotate_predictions(args.text, result)
        elif args.command == "evidence":
            from .evidence import load_evidence_extractor

            extractor = load_evidence_extractor(args.tfidf_run)
            result = {"aspect": args.aspect, **extractor.rank(args.text, args.aspect, args.top_k)}
        elif args.command == "evidence-sample":
            from .evidence_eval import sample_annotations

            result = sample_annotations(
                args.dataset, args.split, args.output, args.per_aspect, args.seed
            )
        else:
            from .evidence import load_evidence_extractor
            from .evidence_eval import evaluate_annotations

            extractor = load_evidence_extractor(args.tfidf_run)
            report = evaluate_annotations(extractor, args.dataset, args.split, args.annotations)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            result = {
                "output": str(args.output.resolve()),
                "annotated": report["annotated"],
                "pending": report["pending"],
                "methods": report["methods"],
            }
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")
    # ASCII escapes preserve Chinese names in redirected Windows JSON output.
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

