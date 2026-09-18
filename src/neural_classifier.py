
"""Trains, evaluates, saves, and loads the final MiniLM-based MLP classifier for AI misuse detection."""

from __future__ import annotations
import hashlib
import io
import json
import logging
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
#Shared helpers identify label-only tool fields and calculate text entropy
from core import TRUE_LABEL_ONLY_TOOLCALL_FIELDS, shannon_entropy

#Paths, class labels and the fixed feature schema used by training and inference
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_PATH = OUTPUTS_DIR / "simulated_events.jsonl"
MODEL_PATH = OUTPUTS_DIR / "neural_classifier_model.joblib"
RANDOM_SEED = 42
DEVELOPMENT_GROUPS_PER_FAMILY = 4
TEST_GROUPS_PER_FAMILY = 2
BENIGN_LABEL = "benign"
MODEL_SCHEMA_VERSION = 3
ARCHITECTURE_NAME = "flat_nine_class_structured_policy_mlp"
ENCODER_NAME = "minilm"
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FEATURE_SET_NAME = "combined_plus_structured_policy"
MISUSE_CATEGORIES = [
    "prompt_injection",
    "sensitive_info_disclosure",
    "excessive_agency",
    "unbounded_consumption",
    "misinformation",
    "hidden_context_exposure",
    "vector_embedding_weakness",
    "improper_output_handling",
]
CLASS_NAMES = [BENIGN_LABEL] + MISUSE_CATEGORIES
CATEGORICAL_FEATURES = [
    "department",
    "actor_role",
    "ai_platform",
    "task_type",
    "content_source",
    "requester_authorization_level",
    "record_scope",
    "data_classification_touched",
    "human_approval",
    "retrieval_owner_department",
]
NUMERIC_FEATURES = [
    "num_tool_calls",
    "sanctioned_platform",
    "text_entropy",
    "calls_in_time_window",
    "systems_touched_count",
    "cumulative_session_cost",
    "output_token_count",
    "record_count",
    "requested_quantity",
    "executed_quantity",
    "num_admin_permission_calls",
    "num_read_write_permission_calls",
    "num_generic_identity_calls",
    "num_irreversible_calls",
    "num_successful_calls",
    "num_failed_calls",
    "num_not_attempted_calls",
    "num_external_destination_calls",
]
OBSERVABLE_TOOLCALL_FIELDS = {
    "tool_name",
    "command",
    "parameters",
    "permission_level",
    "identity_scope",
    "reversibility",
    "outcome",
    "destination",
}
_MODEL_CACHE: dict[str, object] = {}
_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


#Suppress download warnings (but keeps unexpected errors visible)
class _FilteredStderr:

    def __enter__(self):
        self._buffer = io.StringIO()
        self._original_stderr = sys.stderr
        sys.stderr = self._buffer
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        sys.stderr = self._original_stderr
        for line in self._buffer.getvalue().splitlines():
            if "unauthenticated requests" not in line and "HF_TOKEN" not in line:
                print(line, file=sys.stderr)
        return False


#random seed
def set_reproducibility() -> None:
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
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


def find_dataset() -> Path:
    if DATA_PATH.exists():
        return DATA_PATH
    raise FileNotFoundError(
        f"The frozen dataset was not found at {DATA_PATH}. Run dataset_simulator.py first."
    )


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    path = path or find_dataset()
    with path.open(encoding="utf-8-sig") as file:
        records = [json.loads(line) for line in file if line.strip()]
    dataframe = pd.DataFrame(records)
    return dataframe


def get_pattern_family(split_group_id: str) -> str:
    if split_group_id.startswith("agg_variant_"):
        return "sensitive_info_disclosure.aggregation"
    family, separator, suffix = split_group_id.rpartition(".")
    if not separator or not suffix.isdigit():
        raise ValueError(f"Unexpected split_group_id: {split_group_id}")
    return family


@dataclass(frozen=True)
class DatasetPartitions:
    development: pd.DataFrame
    test: pd.DataFrame
    group_to_family: dict[str, str]


def _family_seed(family: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{RANDOM_SEED}:{family}".encode()).digest()[:8], byteorder="big"
    )


#Assign four wording groups to Development and two to Test within every pattern family
def make_grouped_partitions(dataframe: pd.DataFrame) -> DatasetPartitions:
    group_to_family: dict[str, str] = {}
    family_to_groups: dict[str, list[str]] = defaultdict(list)
    for group_id in sorted(dataframe["split_group_id"].unique()):
        family = get_pattern_family(group_id)
        group_to_family[group_id] = family
        family_to_groups[family].append(group_id)
    group_to_role: dict[str, str] = {}
    for family in sorted(family_to_groups):
        groups = sorted(family_to_groups[family])
        if len(groups) != 6:
            raise RuntimeError(
                f"{family} has {len(groups)} independent groups; this classifier expects exactly six (four Development, two Test)."
            )
        random.Random(_family_seed(family)).shuffle(groups)
        for group_id in groups[:DEVELOPMENT_GROUPS_PER_FAMILY]:
            group_to_role[group_id] = "development"
        for group_id in groups[DEVELOPMENT_GROUPS_PER_FAMILY:]:
            group_to_role[group_id] = "test"
    roles = dataframe["split_group_id"].map(group_to_role)
    if roles.isna().any():
        raise RuntimeError("Some events were not assigned to Development or Test.")
    development = dataframe.loc[roles == "development"].copy().reset_index(drop=True)
    test = dataframe.loc[roles == "test"].copy().reset_index(drop=True)
    result = DatasetPartitions(development, test, group_to_family)
    verify_grouped_partitions(dataframe, result)
    return result


#Confirm that related identifiers stay entirely within one partition
def verify_grouped_partitions(
    full_data: pd.DataFrame, parts: DatasetPartitions
) -> None:
    development_groups = set(parts.development["split_group_id"])
    test_groups = set(parts.test["split_group_id"])
    for field_name in ("event_id", "session_id", "template_id", "split_group_id"):
        if set(parts.development[field_name]).intersection(parts.test[field_name]):
            raise RuntimeError(f"{field_name} leakage between Development and Test.")
    if set(parts.development["event_id"]) | set(parts.test["event_id"]) != set(
        full_data["event_id"]
    ):
        raise RuntimeError(
            "Development and Test do not contain every event exactly once."
        )
    expected_families = set(parts.group_to_family.values())
    for family in expected_families:
        dev_count = sum(
            (parts.group_to_family[g] == family for g in development_groups)
        )
        test_count = sum((parts.group_to_family[g] == family for g in test_groups))
        if (dev_count, test_count) != (
            DEVELOPMENT_GROUPS_PER_FAMILY,
            TEST_GROUPS_PER_FAMILY,
        ):
            raise RuntimeError(
                f"{family} has Development/Test group counts {dev_count}/{test_count}."
            )


#Keep only fields that would be observable when the classifier is used
def observable_tool_calls(raw_tool_calls: object) -> list[dict]:
    if not isinstance(raw_tool_calls, list):
        raise TypeError("tool_calls must be a list.")
    calls: list[dict] = []
    for index, raw_call in enumerate(raw_tool_calls):
        if not isinstance(raw_call, dict):
            raise TypeError(f"tool_calls[{index}] must be a dictionary.")
        unexpected = (
            set(raw_call) - OBSERVABLE_TOOLCALL_FIELDS - TRUE_LABEL_ONLY_TOOLCALL_FIELDS
        )
        if unexpected:
            raise ValueError(
                f"tool_calls[{index}] has unknown fields: {sorted(unexpected)}"
            )
        if (
            not isinstance(raw_call.get("tool_name"), str)
            or not raw_call["tool_name"].strip()
        ):
            raise ValueError(
                f"tool_calls[{index}] is missing a valid string 'tool_name'."
            )
        parameters = raw_call.get("parameters")
        if parameters is None:
            parameters = {}
        elif not isinstance(parameters, dict):
            raise TypeError(f"tool_calls[{index}]['parameters'] must be a dictionary.")
        calls.append(
            {
                "tool_name": raw_call["tool_name"],
                "command": raw_call.get("command"),
                "parameters": parameters,
                "permission_level": raw_call.get("permission_level"),
                "identity_scope": raw_call.get("identity_scope"),
                "reversibility": raw_call.get("reversibility"),
                "outcome": raw_call.get("outcome"),
                "destination": raw_call.get("destination"),
            }
        )
    return calls


REQUIRED_EVENT_FIELDS = {
    "prompt",
    "model_output",
    "reasoning_trace",
    "retrieved_content",
    "department",
    "actor_role",
    "ai_platform",
    "task_type",
    "content_source",
    "sanctioned_platform",
    "calls_in_time_window",
    "tool_calls",
    "requester_authorization_level",
    "record_scope",
    "data_classification_touched",
    "human_approval",
    "systems_touched_count",
    "cumulative_session_cost",
    "output_token_count",
    "retrieval_owner_department",
    "record_count",
    "requested_quantity",
    "executed_quantity",
}
_NUMERIC_RAW_FIELDS = [
    "calls_in_time_window",
    "systems_touched_count",
    "cumulative_session_cost",
    "output_token_count",
    "record_count",
    "requested_quantity",
    "executed_quantity",
]


#Reject incomplete or invalid records before feature extraction
def validate_event_for_inference(record: dict) -> None:
    if not isinstance(record, dict):
        raise TypeError("Every event record must be a dictionary.")
    missing = REQUIRED_EVENT_FIELDS - set(record)
    if missing:
        raise ValueError(
            f"Event is missing required observable fields: {sorted(missing)}"
        )
    for field_name in _NUMERIC_RAW_FIELDS:
        value = record[field_name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or (not np.isfinite(value))
        ):
            raise ValueError(f"'{field_name}' must be a finite number, got {value!r}.")


#Combine interaction text with policy, telemetry and summarised tool-call features
def build_model_features(records: pd.DataFrame | Iterable[dict]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        source_records = records.to_dict(orient="records")
    else:
        source_records = list(records)
    if not source_records:
        raise ValueError("At least one event is required.")
    rows = []
    for record in source_records:
        validate_event_for_inference(record)
        tool_calls = observable_tool_calls(record["tool_calls"])
        tool_names = " ".join((call["tool_name"] for call in tool_calls))
        tool_targets = " ".join(
            (str(call.get("parameters", {}).get("target", "")) for call in tool_calls)
        )
        tool_commands = " ".join((str(call.get("command", "")) for call in tool_calls))
        num_admin_permission_calls = sum(
            (call.get("permission_level") == "admin" for call in tool_calls)
        )
        num_read_write_permission_calls = sum(
            (call.get("permission_level") == "read_write" for call in tool_calls)
        )
        num_generic_identity_calls = sum(
            (call.get("identity_scope") == "generic_privileged" for call in tool_calls)
        )
        num_irreversible_calls = sum(
            (call.get("reversibility") == "irreversible" for call in tool_calls)
        )
        num_successful_calls = sum(
            (call.get("outcome") == "success" for call in tool_calls)
        )
        num_failed_calls = sum(
            (call.get("outcome") == "failure" for call in tool_calls)
        )
        num_not_attempted_calls = sum(
            (call.get("outcome") == "not_attempted" for call in tool_calls)
        )
        num_external_destination_calls = sum(
            (
                call.get("destination") not in {None, "", "internal"}
                for call in tool_calls
            )
        )
        text = " ".join(
            (
                part
                for part in (
                    record["prompt"],
                    record["model_output"],
                    record["reasoning_trace"],
                    record["retrieved_content"],
                    tool_names,
                    tool_targets,
                    tool_commands,
                )
                if part
            )
        )
        row = {
            "text": text,
            "prompt_only": record["prompt"],
            "retrieved_content_only": record["retrieved_content"],
            "department": record["department"],
            "actor_role": record["actor_role"],
            "ai_platform": record["ai_platform"],
            "task_type": record["task_type"],
            "content_source": record["content_source"],
            "num_tool_calls": len(tool_calls),
            "sanctioned_platform": int(record["sanctioned_platform"]),
            "text_entropy": shannon_entropy(text),
            "calls_in_time_window": record["calls_in_time_window"],
            "requester_authorization_level": str(
                record["requester_authorization_level"]
            ),
            "record_scope": str(record["record_scope"]),
            "data_classification_touched": str(record["data_classification_touched"]),
            "human_approval": str(record["human_approval"]).lower(),
            "retrieval_owner_department": str(record["retrieval_owner_department"]),
            "systems_touched_count": record["systems_touched_count"],
            "cumulative_session_cost": record["cumulative_session_cost"],
            "output_token_count": record["output_token_count"],
            "record_count": record["record_count"],
            "requested_quantity": record["requested_quantity"],
            "executed_quantity": record["executed_quantity"],
            "num_admin_permission_calls": num_admin_permission_calls,
            "num_read_write_permission_calls": num_read_write_permission_calls,
            "num_generic_identity_calls": num_generic_identity_calls,
            "num_irreversible_calls": num_irreversible_calls,
            "num_successful_calls": num_successful_calls,
            "num_failed_calls": num_failed_calls,
            "num_not_attempted_calls": num_not_attempted_calls,
            "num_external_destination_calls": num_external_destination_calls,
        }
        rows.append(row)
    return pd.DataFrame(rows)


#Load MiniLM (reuse it for later for embedding requests)
def load_sentence_model():
    if MINILM_MODEL_NAME not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            from huggingface_hub import logging as hf_logging
        except ImportError as error:
            raise ImportError(
                "MiniLM requires sentence-transformers. Install it with: pip install sentence-transformers"
            ) from error
        hf_logging.set_verbosity_error()
        with _FilteredStderr():
            _MODEL_CACHE[MINILM_MODEL_NAME] = SentenceTransformer(MINILM_MODEL_NAME)
    return _MODEL_CACHE[MINILM_MODEL_NAME]


#Cache normalised embeddings so repeated text is not encoded again
def encode_minilm(texts) -> np.ndarray:
    prepared = [str(text) if text is not None else "" for text in texts]
    if not prepared:
        return np.empty((0, 384), dtype=np.float32)
    missing = [text for text in dict.fromkeys(prepared) if text not in _EMBEDDING_CACHE]
    if missing:
        vectors = load_sentence_model().encode(
            missing,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for text, vector in zip(missing, vectors):
            _EMBEDDING_CACHE[text] = np.asarray(vector, dtype=np.float32)
    result = np.vstack([_EMBEDDING_CACHE[text] for text in prepared])
    if not np.isfinite(result).all():
        raise RuntimeError("MiniLM returned non-finite embedding values.")
    return result


class MiniLMTextEncoder(BaseEstimator, TransformerMixin):

    def fit(self, values, labels=None):
        encode_minilm(pd.Series(values).fillna("").astype(str).tolist())
        return self

    def transform(self, values):
        return encode_minilm(pd.Series(values).fillna("").astype(str).tolist())


#Measure whether retrieved content exists and how closely it matches the prompt
class PromptRetrievedSimilarity(BaseEstimator, TransformerMixin):

    def fit(self, values, labels=None):
        frame = pd.DataFrame(np.asarray(values), columns=["prompt", "retrieved"])
        encode_minilm(frame["prompt"].fillna("").tolist())
        non_empty = frame.loc[frame["retrieved"].fillna("") != "", "retrieved"]
        if len(non_empty):
            encode_minilm(non_empty.tolist())
        return self

    def transform(self, values):
        frame = pd.DataFrame(np.asarray(values), columns=["prompt", "retrieved"])
        prompts = frame["prompt"].fillna("").astype(str).tolist()
        retrieved = frame["retrieved"].fillna("").astype(str).tolist()
        exists = np.asarray([bool(text) for text in retrieved], dtype=float)
        similarities = np.zeros(len(frame), dtype=float)
        active = np.flatnonzero(exists)
        if len(active):
            active_prompts = [prompts[i] for i in active]
            active_retrieved = [retrieved[i] for i in active]
            prompt_vectors = encode_minilm(active_prompts)
            retrieved_vectors = encode_minilm(active_retrieved)
            similarities[active] = np.sum(prompt_vectors * retrieved_vectors, axis=1)
        return np.column_stack([exists, similarities])


#Prepare text, retrieval similarity, categorical values and numeric telemetry
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("text", MiniLMTextEncoder(), "text"),
            (
                "retrieval_similarity",
                PromptRetrievedSimilarity(),
                ["prompt_only", "retrieved_content_only"],
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


#Wrap the nine-class MLP so label encoding remains inside the saved pipeline
class RegularizedMLPClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state

    def fit(self, features, labels):
        self.label_encoder_ = LabelEncoder()
        encoded_labels = self.label_encoder_.fit_transform(labels)
        self.model_ = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=0.01,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=self.random_state,
        )
        self.model_.fit(features, encoded_labels)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, features):
        encoded = self.model_.predict(features).astype(int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, features):
        return self.model_.predict_proba(features)


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("features", build_preprocessor()),
            ("classifier", RegularizedMLPClassifier(random_state=RANDOM_SEED)),
        ]
    )


def fit_final_model(development: pd.DataFrame) -> Pipeline:
    model = build_model()
    model.fit(
        build_model_features(development),
        development["misuse_category"].astype(str).to_numpy(),
    )
    return model


#Save the fitted pipeline together
def make_model_bundle(
    model: Pipeline, parts: DatasetPartitions, test_metrics: dict | None = None
) -> dict:
    try:
        import sentence_transformers

        sentence_transformers_version = sentence_transformers.__version__
    except ImportError:
        sentence_transformers_version = None
    return {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "architecture": ARCHITECTURE_NAME,
        "encoder": ENCODER_NAME,
        "encoder_model": MINILM_MODEL_NAME,
        "feature_set": FEATURE_SET_NAME,
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "class_names": list(CLASS_NAMES),
        "misuse_categories": list(MISUSE_CATEGORIES),
        "dataset_event_count": len(parts.development) + len(parts.test),
        "random_seed": RANDOM_SEED,
        "development_group_count": int(parts.development["split_group_id"].nunique()),
        "test_group_count": int(parts.test["split_group_id"].nunique()),
        "test_metrics": test_metrics or {},
        "model_classes": list(model.classes_),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sentence-transformers": sentence_transformers_version,
        },
        "model": model,
    }


#Prevent inference with a model whose classes or feature schema no longer match this code
def validate_model_bundle(bundle: dict) -> None:
    required = {
        "model_schema_version",
        "categorical_features",
        "numeric_features",
        "model_classes",
        "model",
    }
    missing = required - set(bundle)
    if missing:
        raise RuntimeError(f"The model bundle is missing: {sorted(missing)}")
    if bundle["model_schema_version"] != MODEL_SCHEMA_VERSION:
        raise RuntimeError("The saved model schema is incompatible with this code.")
    if set(bundle["model_classes"]) != set(CLASS_NAMES):
        raise RuntimeError(
            f"The classifier does not contain exactly the nine expected classes. Got: {sorted(bundle['model_classes'])}"
        )
    if bundle["categorical_features"] != CATEGORICAL_FEATURES:
        raise RuntimeError(
            "The saved categorical feature schema does not match this code."
        )
    if bundle["numeric_features"] != NUMERIC_FEATURES:
        raise RuntimeError("The saved numeric feature schema does not match this code.")


def save_model_bundle(bundle: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        joblib.dump(bundle, temporary_path, compress=3)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_model_bundle(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No trained classifier was found at {path}.")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise RuntimeError("The saved classifier is not a valid model bundle.")
    validate_model_bundle(bundle)
    return bundle


#Return the category prediction and probabilities used later by the fusion layer
def predict_records(
    bundle: dict, records: pd.DataFrame | Iterable[dict]
) -> pd.DataFrame:
    features = build_model_features(records)
    model: Pipeline = bundle["model"]
    classes = np.asarray(model.classes_)
    probabilities = model.predict_proba(features)
    probability_frame = pd.DataFrame(
        probabilities, columns=classes, index=features.index
    )[CLASS_NAMES]
    final_prediction = classes[probabilities.argmax(axis=1)]
    benign_probability = probability_frame[BENIGN_LABEL].to_numpy()
    misuse_probability = 1.0 - benign_probability
    classification_confidence = probability_frame.max(axis=1).to_numpy()
    return pd.DataFrame(
        {
            "predicted_category": final_prediction,
            "classification_confidence": classification_confidence,
            "benign_probability": benign_probability,
            "misuse_risk_score": misuse_probability,
            "category_probabilities": probability_frame.to_dict(orient="records"),
        }
    )


#Report both nine-class performance and binary misuse detection errors
def calculate_metrics(true_labels, predictions) -> dict[str, float]:
    true_labels = np.asarray(true_labels)
    predictions = np.asarray(predictions)
    true_misuse = true_labels != BENIGN_LABEL
    predicted_misuse = predictions != BENIGN_LABEL
    tn, fp, fn, tp = confusion_matrix(
        true_misuse, predicted_misuse, labels=[False, True]
    ).ravel()
    return {
        "macro_f1": f1_score(
            true_labels, predictions, average="macro", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predictions),
        "accuracy": accuracy_score(true_labels, predictions),
        "misuse_recall": recall_score(true_misuse, predicted_misuse, zero_division=0),
        "false_positive_rate": float(
            np.mean(predicted_misuse[~true_misuse]) if (~true_misuse).any() else 0.0
        ),
        "false_negative_rate": float(
            np.mean(~predicted_misuse[true_misuse]) if true_misuse.any() else 0.0
        ),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def evaluate_model(bundle: dict, test: pd.DataFrame) -> dict[str, float]:
    predictions = predict_records(bundle, test)
    true_labels = test["misuse_category"].astype(str).to_numpy()
    predicted_labels = predictions["predicted_category"].to_numpy()
    print("\nMLP Classifier Evaluation (On Test):")
    print(
        classification_report(
            true_labels, predicted_labels, labels=CLASS_NAMES, zero_division=0
        )
    )
    metrics = calculate_metrics(true_labels, predicted_labels)
    print("TP, FP, TN and FN:")
    print(f"{'TP':>10s}{'FP':>10s}")
    print(f"{metrics['tp']:>10d}{metrics['fp']:>10d}")
    print(f"{'TN':>10s}{'FN':>10s}")
    print(f"{metrics['tn']:>10d}{metrics['fn']:>10d}")
    print("\nMulticlass Confusion Matrix:")
    matrix = confusion_matrix(true_labels, predicted_labels, labels=CLASS_NAMES)
    print(pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
    print("\nMetrics Summary:")
    print(f"Macro F1:              {metrics['macro_f1']:.4f}")
    print(f"Balanced accuracy:      {metrics['balanced_accuracy']:.4f}")
    print(f"Accuracy:               {metrics['accuracy']:.4f}")
    print(f"Misuse recall:          {metrics['misuse_recall']:.4f}")
    print(f"False positive rate:    {metrics['false_positive_rate']:.4f}")
    print(f"False negative rate:    {metrics['false_negative_rate']:.4f}")
    return metrics


#Run the complete classifier workflow on the frozen dataset
def train_and_save(
    dataset_path: Path | None = None, model_path: Path = MODEL_PATH
) -> tuple[dict, DatasetPartitions, dict[str, float]]:
    set_reproducibility()
    print("Loading Dataset..")
    dataframe = load_dataset(dataset_path)
    parts = make_grouped_partitions(dataframe)
    print(f"Dataset Loaded: {len(dataframe):,} events")
    print("\nDataset Split:")
    print(
        f"Development: {len(parts.development):,} events (4 groups per pattern family)"
    )
    print(f"Test: {len(parts.test):,} events (2 groups per pattern family)")
    print("Method: Grouped by split_group_id with seeded family-level assignment")
    print("\nEncoding Development text with MiniLM...")
    print("Scaling numeric features and encoding categorical features...")
    print("Training MLP Classifier...")
    model = fit_final_model(parts.development)
    print("Training complete.")
    bundle = make_model_bundle(model, parts)
    metrics = evaluate_model(bundle, parts.test)
    bundle["test_metrics"] = metrics
    save_model_bundle(bundle, model_path)
    print(f"\nModel saved as {model_path.name} to: {model_path.parent}")
    return (bundle, parts, metrics)

if __name__ == "__main__":
    sys.modules["neural_classifier"] = sys.modules[__name__]
    MiniLMTextEncoder.__module__ = "neural_classifier"
    PromptRetrievedSimilarity.__module__ = "neural_classifier"
    RegularizedMLPClassifier.__module__ = "neural_classifier"
    train_and_save()
