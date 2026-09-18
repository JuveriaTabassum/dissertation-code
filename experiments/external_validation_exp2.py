"""Runs the external-only neural, symbolic and fusion experiment."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import neural_classifier
import neurosymbolic_fusion
import symbolic_engine
from core import AgentEvent, MisuseCategory, PLATFORM_INVENTORY


HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "external_validation_events.jsonl"
CLASSES = list(neural_classifier.CLASS_NAMES)
BENIGN = neural_classifier.BENIGN_LABEL


def load_external() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"external_validation_events.jsonl was not found at {DATASET_PATH}. "
            "Place it directly in the experiments folder."
        )
    records = []
    with DATASET_PATH.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {number}.") from error
            if not isinstance(record, dict):
                raise TypeError(f"Line {number} is not a JSON object.")
            records.append(record)

    frame = pd.DataFrame(records)
    required = {
        "event_id", "misuse_category", "prompt", "model_output",
        "reasoning_trace", "retrieved_content", "tool_calls",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"External dataset is missing {sorted(missing)}.")
    if len(frame) != 900 or frame["event_id"].duplicated().any():
        raise RuntimeError("External dataset count or event IDs are invalid.")
    counts = Counter(frame["misuse_category"].astype(str))
    if counts != Counter({category: 100 for category in CLASSES}):
        raise RuntimeError(f"Incorrect class balance: {dict(counts)}")
    return frame


def split_external(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    #Split every category equally into external training and test sets
    training_parts, test_parts = [], []
    for category in CLASSES:
        rows = frame.loc[frame["misuse_category"].astype(str) == category].copy()
        rows["_key"] = rows["event_id"].astype(str).map(
            lambda value: hashlib.sha256(
                f"experiment-4|{value}".encode("utf-8")
            ).hexdigest()
        )
        rows = rows.sort_values(["_key", "event_id"], kind="stable")
        training_parts.append(rows.iloc[:50].drop(columns="_key"))
        test_parts.append(rows.iloc[50:].drop(columns="_key"))
    training = pd.concat(training_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    if set(training.event_id) & set(test.event_id):
        raise RuntimeError("External training and Test IDs overlap.")
    if len(training) != 450 or len(test) != 450:
        raise RuntimeError("External split did not produce 450/450 events.")
    return training, test


def observable_calls(raw_calls: object, record_number: int) -> list[dict]:
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, list):
        raise TypeError(f"Record {record_number} has invalid tool_calls.")
    allowed = set(neural_classifier.OBSERVABLE_TOOLCALL_FIELDS)
    calls = []
    for call_number, raw in enumerate(raw_calls, 1):
        if not isinstance(raw, dict):
            raise TypeError(
                f"Record {record_number}, call {call_number} is not an object."
            )
        call = {key: value for key, value in raw.items() if key in allowed}
        if not call.get("command") and raw.get("raw_action"):
            call["command"] = str(raw["raw_action"])
        if not call.get("tool_name"):
            raise ValueError(
                f"Record {record_number}, call {call_number} has no tool_name."
            )
        calls.append(call)
    return neural_classifier.observable_tool_calls(calls)


def compose_text(record: dict, number: int) -> str:
    #Merge all available text fields into one input for MiniLM
    calls = observable_calls(record.get("tool_calls"), number)
    tool_names = " ".join(str(call["tool_name"]) for call in calls)
    tool_targets = " ".join(
        str(call.get("parameters", {}).get("target", "")) for call in calls
    )
    tool_commands = " ".join(str(call.get("command", "")) for call in calls)
    return " ".join(
        str(value) for value in (
            record.get("prompt"), record.get("model_output"),
            record.get("reasoning_trace"), record.get("retrieved_content"),
            tool_names, tool_targets, tool_commands,
        )
        if value is not None and str(value).strip()
    )


def text_inputs(frame: pd.DataFrame) -> pd.Series:
    texts = [
        compose_text(record, number)
        for number, record in enumerate(frame.to_dict(orient="records"), 1)
    ]
    if not any(text.strip() for text in texts):
        raise RuntimeError("No external text could be constructed.")
    return pd.Series(texts, name="text")


def external_context_events(frame: pd.DataFrame) -> list[AgentEvent]:
    #Create minimal event objects needed by the fusion function
    events = []
    for row in frame.to_dict(orient="records"):
        try:
            timestamp = datetime.fromisoformat(str(row.get("timestamp")))
        except (TypeError, ValueError):
            timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
        ai_platform = str(row.get("ai_platform") or "unknown")
        events.append(AgentEvent(
            event_id=str(row["event_id"]),
            timestamp=timestamp,
            session_id=str(row.get("session_id") or row["event_id"]),
            actor_role=str(row.get("actor_role") or "unknown"),
            department=str(row.get("department") or "unknown"),
            task_type=str(row.get("task_type") or "unknown"),
            ai_platform=ai_platform,
            sanctioned_platform=PLATFORM_INVENTORY.get(ai_platform),
            misuse_category=MisuseCategory(str(row["misuse_category"])),
            misuse_pattern=str(row.get("misuse_pattern") or "external"),
        ))
    return events


def neural_output(model, texts: pd.Series) -> pd.DataFrame:
    probabilities = np.asarray(model.predict_proba(texts), dtype=float)
    classes = np.asarray(model.classes_, dtype=str)
    return pd.DataFrame({
        "predicted_category": classes[probabilities.argmax(axis=1)],
        "confidence": probabilities.max(axis=1),
        "category_probabilities": [
            dict(zip(classes, row.astype(float))) for row in probabilities
        ],
    })


def symbolic_metrics(truth: np.ndarray) -> dict:
    predictions = np.full(len(truth), BENIGN, dtype=object)
    metrics = neurosymbolic_fusion.calculate_full_metrics(truth, predictions)
    true_misuse = truth != BENIGN
    metrics.update({
        "misuse_precision": 0.0,
        "misuse_recall": 0.0,
        "fpr": 0.0,
        "fnr": 1.0 if true_misuse.any() else 0.0,
        "tp": 0,
        "fp": 0,
        "tn": int((~true_misuse).sum()),
        "fn": int(true_misuse.sum()),
    })
    return metrics


def main() -> None:
    neural_classifier.set_reproducibility()
    print("Experiment 4: external-only comparison")
    external_training, external_test = split_external(load_external())
    print(f"External training events: {len(external_training)} (50 per category)")
    print(f"External Test events:     {len(external_test)} (50 per category)")
    print("External events contain text but not operational telemetry.")

    #Train a text-only model because the external records lack telemetry
    print("\nEncoding external text with MiniLM and training the MLP")
    model = Pipeline([
        ("minilm", neural_classifier.MiniLMTextEncoder()),
        ("classifier", neural_classifier.RegularizedMLPClassifier(
            random_state=neural_classifier.RANDOM_SEED
        )),
    ])
    model.fit(
        text_inputs(external_training),
        external_training["misuse_category"].astype(str).to_numpy(),
    )
    neural = neural_output(model, text_inputs(external_test))
    truth = external_test["misuse_category"].astype(str).to_numpy()

    #With no symbolic evidence the fused category should match the neural one
    events = external_context_events(external_test)
    unavailable = {
        event.event_id: symbolic_engine.SymbolicVerdict() for event in events
    }
    fused = neurosymbolic_fusion.fuse_predictions(events, unavailable, neural, {})
    neural_labels = neural["predicted_category"].to_numpy()
    fused_labels = np.asarray([
        fused[event.event_id].final_category for event in events
    ])
    if not np.array_equal(neural_labels, fused_labels):
        raise RuntimeError("No-evidence fusion changed an external neural category.")

    rows = {
        "symbolic": symbolic_metrics(truth),
        "neural": neurosymbolic_fusion.calculate_full_metrics(truth, neural_labels),
        "fused": neurosymbolic_fusion.calculate_full_metrics(truth, fused_labels),
    }
    print("\nEXTERNAL-ONLY THREE-WAY COMPARISON")
    neurosymbolic_fusion.print_metrics_table(rows)
    print("\nSYMBOLIC AVAILABILITY")
    print("External Test events with symbolic findings: 0 / 450")
    print("Reason: required operational telemetry is unavailable.")
    print("Fused category metrics therefore equal neural category metrics.")
    print(
        "This is an auxiliary reduced-observability experiment, not a "
        "replacement evaluation of the final full-feature system."
    )


if __name__ == "__main__":
    main()
