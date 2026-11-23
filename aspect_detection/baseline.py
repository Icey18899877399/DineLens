import logging

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from .data import ASPECTS, SENTIMENTS, UNMENTIONED


def tokenize(review):
    return [token for token in jieba.lcut(review) if any(char.isalnum() for char in token)]


def _vectorizer():
    jieba.setLogLevel(logging.WARNING)
    return TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
    )


def _classifier(seed):
    return OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear", class_weight="balanced", max_iter=1000, random_state=seed
        )
    )


def train_baseline(train, seed):
    pipeline = Pipeline([("tfidf", _vectorizer()), ("classifier", _classifier(seed))])
    return pipeline.fit(train.texts, train.labels)


class SentimentBaseline:
    def __init__(self, mention_pipeline, sentiment_models, sentiment_priors):
        self.mention_pipeline = mention_pipeline
        self.sentiment_models = sentiment_models
        self.sentiment_priors = sentiment_priors

    def score_reviews(self, texts):
        features = self.mention_pipeline["tfidf"].transform(texts)
        mentions = self.mention_pipeline["classifier"].predict_proba(features)
        sentiments = np.zeros((len(texts), len(ASPECTS), len(SENTIMENTS)))
        for index, model in enumerate(self.sentiment_models):
            if model is None:
                sentiments[:, index, :] = self.sentiment_priors[index]
            else:
                aspect_scores = sentiments[:, index, :]
                aspect_scores[:, model.classes_ + 1] = model.predict_proba(features)
        return mentions, sentiments


def train_sentiment_baseline(train, seed):
    vectorizer = _vectorizer()
    features = vectorizer.fit_transform(train.texts)
    mention_classifier = _classifier(seed).fit(features, train.labels)
    mention_pipeline = Pipeline([("tfidf", vectorizer), ("classifier", mention_classifier)])

    gold = train.polarities[train.polarities != UNMENTIONED]
    if not gold.size:
        raise ValueError("训练集没有已提及方面的人工情感标签")
    global_counts = np.bincount(gold + 1, minlength=len(SENTIMENTS))
    global_prior = global_counts / global_counts.sum()
    models, priors = [], []
    for column in train.polarities.T:
        selected = column != UNMENTIONED
        labels = column[selected]
        counts = np.bincount(labels + 1, minlength=len(SENTIMENTS))
        model = None
        if np.count_nonzero(counts) >= 2:
            model = _classifier(seed).fit(features[selected], labels)
        models.append(model)
        priors.append(counts / counts.sum() if labels.size else global_prior)
    return SentimentBaseline(mention_pipeline, models, priors)

