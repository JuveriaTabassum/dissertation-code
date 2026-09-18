
"""Runs grouped classifier experiments for encoder, architecture and feature selection."""

from __future__ import annotations

import gc
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, normalize

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

#Same split, features and labels as the final classifier
import neural_classifier as current

RANDOM_SEED = current.RANDOM_SEED
BENIGN_LABEL = current.BENIGN_LABEL
N_CV_FOLDS = 4
ENCODERS = {
    "TF-IDF": "tfidf",
    "MiniLM": current.MINILM_MODEL_NAME,
    "BGE-small": "BAAI/bge-small-en-v1.5",
}
_MODEL_CACHE: dict[str, object] = {}
_EMBEDDING_CACHE: dict[tuple[str, str], np.ndarray] = {}


@dataclass
class ExperimentResult:
    name: str
    true_labels: np.ndarray
    predictions: np.ndarray
    fold_metrics: list[dict[str, float]]
    seconds: float


def set_reproducibility() -> None:
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    try:
        import torch
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)
    except ImportError:
        pass


def make_grouped_folds(development: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    #Hold out one group per pattern family to prevent event related leakage
    family_groups: dict[str, list[str]] = {}
    for group_id in sorted(development["split_group_id"].unique()):
        family = current.get_pattern_family(group_id)
        family_groups.setdefault(family, []).append(group_id)
    for family, groups in family_groups.items():
        if len(groups) != N_CV_FOLDS:
            raise RuntimeError(f"{family} has {len(groups)} Development groups; expected four.")
    folds = []
    all_validation_ids = []
    for fold_index in range(N_CV_FOLDS):
        validation_groups = {groups[fold_index] for groups in family_groups.values()}
        validation_mask = development["split_group_id"].isin(validation_groups).to_numpy()
        train_indices = np.flatnonzero(~validation_mask)
        validation_indices = np.flatnonzero(validation_mask)
        folds.append((train_indices, validation_indices))
        all_validation_ids.extend(development.iloc[validation_indices]["event_id"].tolist())
    if len(all_validation_ids) != len(set(all_validation_ids)):
        raise RuntimeError("A Development event appears in Validation more than once.")
    if set(all_validation_ids) != set(development["event_id"]):
        raise RuntimeError("Every Development event must be Validation exactly once.")
    return folds


def load_sentence_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            from huggingface_hub import logging as hf_logging
        except ImportError as error:
            raise ImportError("Transformer encoders require sentence-transformers.") from error
        hf_logging.set_verbosity_error()
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def encode_texts(model_name: str, values, role: str = "document") -> np.ndarray:
    #Cache embeddings because the same texts are used across several folds
    texts = [str(value) if value is not None else "" for value in values]
    if model_name == ENCODERS["BGE-small"] and role == "query":
        texts = ["Represent this sentence for searching relevant passages: " + text for text in texts]
    missing = [text for text in dict.fromkeys(texts) if (model_name, text) not in _EMBEDDING_CACHE]
    if missing:
        vectors = load_sentence_model(model_name).encode(
            missing,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for text, vector in zip(missing, vectors):
            _EMBEDDING_CACHE[(model_name, text)] = np.asarray(vector, dtype=np.float32)
    if not texts:
        dimensions = load_sentence_model(model_name).get_sentence_embedding_dimension()
        return np.empty((0, dimensions), dtype=np.float32)
    result = np.vstack([_EMBEDDING_CACHE[(model_name, text)] for text in texts])
    if not np.isfinite(result).all():
        raise RuntimeError(f"{model_name} returned non-finite embeddings.")
    return result


class TextEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_name: str):
        self.encoder_name = encoder_name

    def fit(self, values, labels=None):
        texts = pd.Series(values).fillna("").astype(str).tolist()
        if self.encoder_name == "tfidf":
            self.vectorizer_ = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=20_000,
                sublinear_tf=True,
            )
            self.vectorizer_.fit(texts)
        else:
            encode_texts(self.encoder_name, texts)
        return self

    def transform(self, values):
        texts = pd.Series(values).fillna("").astype(str).tolist()
        if self.encoder_name == "tfidf":
            return self.vectorizer_.transform(texts)
        return encode_texts(self.encoder_name, texts)


class RetrievalSimilarity(BaseEstimator, TransformerMixin):
    #Compare each prompt with its retrieved content as an extra feature
    def __init__(self, encoder_name: str):
        self.encoder_name = encoder_name

    def fit(self, values, labels=None):
        frame = pd.DataFrame(np.asarray(values), columns=["prompt", "retrieved"])
        if self.encoder_name == "tfidf":
            corpus = pd.concat([frame["prompt"], frame["retrieved"]]).fillna("").astype(str)
            self.vectorizer_ = TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=10_000, sublinear_tf=True
            )
            self.vectorizer_.fit(corpus)
        else:
            encode_texts(self.encoder_name, frame["prompt"].fillna("").astype(str).tolist())
            retrieved = frame.loc[frame["retrieved"].fillna("") != "", "retrieved"].astype(str)
            if len(retrieved):
                encode_texts(self.encoder_name, retrieved.tolist())
        return self

    def transform(self, values):
        frame = pd.DataFrame(np.asarray(values), columns=["prompt", "retrieved"])
        prompts = frame["prompt"].fillna("").astype(str).tolist()
        retrieved = frame["retrieved"].fillna("").astype(str).tolist()
        exists = np.asarray([bool(text) for text in retrieved], dtype=bool)
        similarities = np.zeros(len(frame), dtype=np.float32)
        if exists.any():
            active = np.flatnonzero(exists)
            if self.encoder_name == "tfidf":
                prompt_vectors = normalize(self.vectorizer_.transform([prompts[index] for index in active]))
                retrieved_vectors = normalize(self.vectorizer_.transform([retrieved[index] for index in active]))
                similarities[active] = np.asarray(prompt_vectors.multiply(retrieved_vectors).sum(axis=1)).ravel()
            else:
                prompt_vectors = encode_texts(self.encoder_name, [prompts[index] for index in active], "query")
                retrieved_vectors = encode_texts(self.encoder_name, [retrieved[index] for index in active])
                similarities[active] = np.sum(prompt_vectors * retrieved_vectors, axis=1)
        return np.column_stack([similarities, exists.astype(np.float32)])


class RegularizedMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state

    def fit(self, features, labels):
        self.label_encoder_ = LabelEncoder().fit(labels)
        encoded = self.label_encoder_.transform(labels)
        self.classes_ = self.label_encoder_.classes_
        self.model_ = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
            random_state=self.random_state,
        )
        self.model_.fit(features, encoded)
        return self

    def predict(self, features):
        return self.label_encoder_.inverse_transform(self.model_.predict(features).astype(int))


def build_preprocessor(encoder_name: str, feature_set: str) -> ColumnTransformer:
    transformers = [("text", TextEncoder(encoder_name), "text")]
    if feature_set == "all features":
        transformers.extend([
            (
                "retrieval_similarity",
                RetrievalSimilarity(encoder_name),
                ["prompt_only", "retrieved_content_only"],
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                current.CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), current.NUMERIC_FEATURES),
        ])
    elif feature_set != "text only":
        raise ValueError(f"Unknown feature set: {feature_set}")
    return ColumnTransformer(transformers=transformers)


def build_classifier(model_name: str):
    if model_name == "LR":
        return LogisticRegression(
            max_iter=2_000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )
    if model_name == "MLP":
        return RegularizedMLPClassifier()
    raise ValueError(f"Unknown classifier: {model_name}")


def build_flat_model(encoder_name: str, model_name: str, feature_set: str) -> Pipeline:
    return Pipeline([
        ("features", build_preprocessor(encoder_name, feature_set)),
        ("classifier", build_classifier(model_name)),
    ])


class TwoStageClassifier:
    #First detect misuse, then predict the specific misuse category
    def __init__(self, encoder_name: str, stage_1: str, stage_2: str, feature_set: str):
        self.stage_1_model = build_flat_model(encoder_name, stage_1, feature_set)
        self.stage_2_model = build_flat_model(encoder_name, stage_2, feature_set)

    def fit(self, features: pd.DataFrame, labels):
        labels = np.asarray(labels)
        binary_labels = np.where(labels == BENIGN_LABEL, BENIGN_LABEL, "misuse")
        benign_indices = np.flatnonzero(binary_labels == BENIGN_LABEL)
        misuse_indices = np.flatnonzero(binary_labels == "misuse")
        generator = np.random.default_rng(RANDOM_SEED)
        selected_misuse = generator.choice(misuse_indices, size=len(benign_indices), replace=False)
        selected = np.concatenate([benign_indices, selected_misuse])
        generator.shuffle(selected)
        self.stage_1_model.fit(features.iloc[selected].reset_index(drop=True), binary_labels[selected])
        misuse_rows = labels != BENIGN_LABEL
        self.stage_2_model.fit(features.loc[misuse_rows], labels[misuse_rows])
        return self

    def predict(self, features: pd.DataFrame):
        stage_1_predictions = self.stage_1_model.predict(features)
        predictions = np.full(len(features), BENIGN_LABEL, dtype=object)
        misuse_rows = stage_1_predictions == "misuse"
        if misuse_rows.any():
            predictions[misuse_rows] = self.stage_2_model.predict(features.loc[misuse_rows])
        return predictions


def calculate_metrics(true_labels, predictions) -> dict[str, float]:
    true_labels = np.asarray(true_labels)
    predictions = np.asarray(predictions)
    true_binary = true_labels != BENIGN_LABEL
    predicted_binary = predictions != BENIGN_LABEL
    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predictions),
        "macro_f1": f1_score(true_labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(true_labels, predictions, average="weighted", zero_division=0),
        "benign_recall": recall_score(~true_binary, ~predicted_binary, zero_division=0),
        "misuse_recall": recall_score(true_binary, predicted_binary, zero_division=0),
    }


def cross_validate(name: str, model_factory, features: pd.DataFrame, labels: np.ndarray, folds) -> ExperimentResult:
    #Train a fresh model on each grouped fold and combine its validation results
    fold_metrics = []
    all_true = []
    all_predictions = []
    started = time.perf_counter()
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        print(f"{name}: fold {fold_number} of {N_CV_FOLDS}", flush=True)
        model = model_factory()
        train_x = features.iloc[train_indices].reset_index(drop=True)
        validation_x = features.iloc[validation_indices].reset_index(drop=True)
        train_y = labels[train_indices]
        validation_y = labels[validation_indices]
        model.fit(train_x, train_y)
        predictions = np.asarray(model.predict(validation_x))
        fold_metrics.append(calculate_metrics(validation_y, predictions))
        all_true.append(validation_y)
        all_predictions.append(predictions)
        del model
        gc.collect()
    return ExperimentResult(
        name=name,
        true_labels=np.concatenate(all_true),
        predictions=np.concatenate(all_predictions),
        fold_metrics=fold_metrics,
        seconds=time.perf_counter() - started,
    )


def result_table(results: list[ExperimentResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"design": result.name}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "benign_recall", "misuse_recall"):
            values = [fold[metric] for fold in result.fold_metrics]
            row[metric] = np.mean(values)
            row[f"{metric}_sd"] = np.std(values, ddof=1)
        row["seconds"] = result.seconds
        rows.append(row)
    columns = [
        "design",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_f1_sd",
        "weighted_f1",
        "benign_recall",
        "misuse_recall",
        "seconds",
    ]
    return pd.DataFrame(rows)[columns]


def print_section(title: str, results: list[ExperimentResult]) -> None:
    print()
    print(title)
    print(result_table(results).round(4).to_string(index=False))


def best_result(results: list[ExperimentResult]) -> ExperimentResult:
    return max(
        results,
        key=lambda result: (
            np.mean([fold["macro_f1"] for fold in result.fold_metrics]),
            np.mean([fold["balanced_accuracy"] for fold in result.fold_metrics]),
        ),
    )


def rename_result(result: ExperimentResult, name: str) -> ExperimentResult:
    return ExperimentResult(
        name=name,
        true_labels=result.true_labels,
        predictions=result.predictions,
        fold_metrics=result.fold_metrics,
        seconds=result.seconds,
    )


def print_confusion_matrix(result: ExperimentResult) -> None:
    labels = current.CLASS_NAMES
    matrix = confusion_matrix(result.true_labels, result.predictions, labels=labels)
    print()
    print("Multiclass confusion matrix for the selected final design")
    print(pd.DataFrame(matrix, index=labels, columns=labels).to_string())


def main() -> None:
    set_reproducibility()
    print("Loading dataset")
    dataframe = current.load_dataset()
    parts = current.make_grouped_partitions(dataframe)
    development = parts.development.reset_index(drop=True)
    folds = make_grouped_folds(development)
    features = current.build_model_features(development)
    labels = development["misuse_category"].astype(str).to_numpy()
    print(f"Development events: {len(development)}")
    print(f"Reserved Test events: {len(parts.test)}")

    #Compare encoders while keeping the MLP and features fixed
    encoder_results = []
    for display_name, encoder_name in ENCODERS.items():
        encoder_results.append(cross_validate(
            display_name,
            lambda encoder_name=encoder_name: build_flat_model(encoder_name, "MLP", "all features"),
            features,
            labels,
            folds,
        ))
    print_section("Section 1: Encoder selection", encoder_results)

    minilm_result = next(result for result in encoder_results if result.name == "MiniLM")

    #Compare flat classification with different two-stage designs
    architecture_designs = [
        ("Flat 9-class LR", lambda: build_flat_model(current.MINILM_MODEL_NAME, "LR", "all features")),
        ("Two-stage LR then LR", lambda: TwoStageClassifier(current.MINILM_MODEL_NAME, "LR", "LR", "all features")),
        ("Two-stage MLP then MLP", lambda: TwoStageClassifier(current.MINILM_MODEL_NAME, "MLP", "MLP", "all features")),
        ("Two-stage LR then MLP", lambda: TwoStageClassifier(current.MINILM_MODEL_NAME, "LR", "MLP", "all features")),
    ]
    architecture_results = [rename_result(minilm_result, "Flat 9-class MLP")] + [
        cross_validate(name, factory, features, labels, folds)
        for name, factory in architecture_designs
    ]
    architecture_order = ["Flat 9-class LR", "Flat 9-class MLP", "Two-stage LR then LR", "Two-stage MLP then MLP", "Two-stage LR then MLP"]
    architecture_results.sort(key=lambda result: architecture_order.index(result.name))
    print_section("Section 2: Model and architecture selection", architecture_results)

    #Check whether the structured features improve over text alone
    feature_results = [
        cross_validate(
            "Flat MLP with text only",
            lambda: build_flat_model(current.MINILM_MODEL_NAME, "MLP", "text only"),
            features,
            labels,
            folds,
        ),
        rename_result(minilm_result, "Flat MLP with all features"),
    ]
    print_section("Section 3: Final MLP feature selection", feature_results)
    selected = best_result(feature_results)
    print_confusion_matrix(selected)
    print()
    print(f"Selected final feature design: {selected.name}")
    print("The reserved Test set was not evaluated.")


if __name__ == "__main__":
    main()
