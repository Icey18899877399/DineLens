import sys
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import ASPECT_KEYS, SENTIMENTS, SENTIMENT_VALUES, UNMENTIONED
from .metrics import evaluate, evaluate_sentiment, fit_thresholds

N_ASPECTS = len(ASPECT_KEYS)
N_SENTIMENTS = len(SENTIMENTS)
JOINT_OUTPUT_KEYS = ASPECT_KEYS + tuple(
    f"{aspect}::{key}" for aspect in ASPECT_KEYS for _, key, _ in SENTIMENTS
)


def sentiment_loss(logits, polarities):
    """Only human-annotated mentions contribute to the polarity loss."""
    targets = polarities.long() - SENTIMENT_VALUES[0]
    ignored = UNMENTIONED - SENTIMENT_VALUES[0]
    loss = F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), ignore_index=ignored, reduction="sum"
    )
    return loss / (targets != ignored).sum().clamp_min(1)


def _device(preference):
    cuda_available = torch.cuda.is_available()
    if preference == "cuda" and not cuda_available:
        raise ValueError(
            "CUDA 不可用。请检查 GPU 和 PyTorch 的 CUDA 版本，或使用 --device cpu。"
        )
    if preference == "auto":
        preference = "cuda" if cuda_available else "cpu"
    return torch.device(preference)


def _encode(texts, tokenizer, max_length, device):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)


@torch.inference_mode()
def _score_reviews(texts, *, model, tokenizer, device, max_length, batch_size, with_sentiment):
    model.eval()
    mentions, sentiments = [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = _encode(batch, tokenizer, max_length, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**inputs).logits
        logits = logits.float()
        mentions.append(logits[:, :N_ASPECTS].sigmoid().cpu().numpy())
        if with_sentiment:
            polarities = logits[:, N_ASPECTS:].reshape(-1, N_ASPECTS, N_SENTIMENTS)
            sentiments.append(polarities.softmax(dim=-1).cpu().numpy())
    mention_scores = np.concatenate(mentions) if mentions else np.empty((0, N_ASPECTS))
    if not with_sentiment:
        return mention_scores
    sentiment_scores = (
        np.concatenate(sentiments) if sentiments else np.empty((0, N_ASPECTS, N_SENTIMENTS))
    )
    return mention_scores, sentiment_scores


def load_bert(checkpoint, preference, max_length, batch_size, with_sentiment=False):
    device = _device(preference)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, use_safetensors=True
    ).to(device)
    keys = tuple(model.config.id2label[index] for index in range(model.config.num_labels))
    expected = JOINT_OUTPUT_KEYS if with_sentiment else ASPECT_KEYS
    if keys != expected:
        raise ValueError("BERT 检查点的输出顺序与当前任务不一致")
    return partial(
        _score_reviews,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        with_sentiment=with_sentiment,
    )


def train_bert(train, dev, checkpoint, args, with_sentiment=False):
    device = _device(args.device)
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate 必须为有限正数")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    output_keys = JOINT_OUTPUT_KEYS if with_sentiment else ASPECT_KEYS
    if with_sentiment and not train.labels.any():
        raise ValueError("训练集没有已提及方面的人工情感标签")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=len(output_keys),
        id2label=dict(enumerate(output_keys)),
        label2id={key: index for index, key in enumerate(output_keys)},
        problem_type=None if with_sentiment else "multi_label_classification",
        ignore_mismatched_sizes=with_sentiment,
        use_safetensors=True,
    ).to(device)
    limit = min(model.config.max_position_embeddings, tokenizer.model_max_length)
    if not 4 <= args.max_length <= limit:
        raise ValueError(f"--max-length 必须介于 4 和 {limit} 之间")

    def collate(batch):
        texts, labels = zip(*batch)
        return texts, np.stack(labels)

    loader = DataLoader(
        list(zip(train.texts, train.polarities)),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    positives = train.labels.sum(axis=0)
    negatives = len(train.labels) - positives
    weights = np.divide(
        negatives, positives, out=np.ones(len(ASPECT_KEYS)), where=(positives > 0) & (negatives > 0)
    )
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    mixed_precision = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    score_reviews = partial(
        _score_reviews,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        with_sentiment=with_sentiment,
    )
    best_f1 = -1.0
    history = []
    print(f"正在使用 {device} 训练 BERT…", file=sys.stderr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        for step, (texts, batch_labels) in enumerate(loader, start=1):
            inputs = _encode(texts, tokenizer, args.max_length, device)
            polarities = torch.as_tensor(batch_labels, dtype=torch.long, device=device)
            mentions = (polarities != UNMENTIONED).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=mixed_precision):
                logits = model(**inputs).logits
                loss = loss_fn(logits[:, :N_ASPECTS], mentions)
                if with_sentiment:
                    polarity_logits = logits[:, N_ASPECTS:].reshape(-1, N_ASPECTS, N_SENTIMENTS)
                    loss = loss + sentiment_loss(polarity_logits, polarities)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach() * len(texts)
            if step % 500 == 0:
                print(f"epoch {epoch}/{args.epochs} · batch {step}/{len(loader)}", file=sys.stderr)

        result = score_reviews(dev.texts)
        scores, polarity_scores = result if with_sentiment else (result, None)
        thresholds = fit_thresholds(dev.labels, scores)
        macro_f1 = evaluate(dev.labels, scores >= thresholds)["macro_f1"]
        mean_loss = float(total_loss.item() / len(train.labels))
        record = {"epoch": epoch, "train_loss": mean_loss, "dev_macro_f1": macro_f1}
        selection_f1 = macro_f1
        summary = f"epoch {epoch} · loss {mean_loss:.4f} · dev 方面 Macro-F1 {macro_f1:.4f}"
        if with_sentiment:
            predicted = np.asarray(SENTIMENT_VALUES)[polarity_scores.argmax(axis=-1)]
            emotion_f1 = evaluate_sentiment(dev.polarities, predicted)["macro_f1"]
            record["dev_sentiment_macro_f1"] = emotion_f1
            if emotion_f1 is not None:
                selection_f1 = emotion_f1
                summary += f" · dev 情感 Macro-F1 {emotion_f1:.4f}"
        history.append(record)
        print(summary, file=sys.stderr)
        if selection_f1 > best_f1:
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            best_f1 = selection_f1
            best_epoch = epoch
    return {
        "source_checkpoint": args.checkpoint,
        "device": str(device),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "best_epoch": best_epoch,
        "history": history,
    }

