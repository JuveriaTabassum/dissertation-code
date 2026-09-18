"""Runs the combined simulated and external-data neural and fusion experiment."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import neural_classifier
import neurosymbolic_fusion
import symbolic_engine
from core import AgentEvent, MisuseCategory, PLATFORM_INVENTORY


HERE = Path(__file__).resolve().parent
EXTERNAL_PATH = HERE / "external_validation_events.jsonl"
CLASSES = list(neural_classifier.CLASS_NAMES)
RAW_NUMERIC = (
    "sanctioned_platform", "calls_in_time_window", "systems_touched_count",
    "cumulative_session_cost", "output_token_count", "record_count",
    "requested_quantity", "executed_quantity",
)


def load_external() -> pd.DataFrame:
    if not EXTERNAL_PATH.exists():
        raise FileNotFoundError(
            f"external_validation_events.jsonl was not found at {EXTERNAL_PATH}. "
            "Place it directly in the experiments folder."
        )
    records = []
    with EXTERNAL_PATH.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on external line {number}.") from error
            if not isinstance(record, dict):
                raise TypeError(f"External line {number} is not an object.")
            records.append(record)
    frame = pd.DataFrame(records)
    if len(frame) != 900:
        raise RuntimeError(f"Expected 900 external events, found {len(frame)}.")
    required = set(neural_classifier.REQUIRED_EVENT_FIELDS) | {"event_id", "misuse_category"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"External dataset is missing {sorted(missing)}.")
    if frame["event_id"].duplicated().any():
        raise RuntimeError("External event IDs are not unique.")
    counts = Counter(frame["misuse_category"].astype(str))
    if counts != Counter({category: 100 for category in CLASSES}):
        raise RuntimeError(f"Incorrect external class balance: {dict(counts)}")
    return frame


def split_external(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    #Split each external category equally for adaptation and testing
    adaptation_parts, test_parts = [], []
    for category in CLASSES:
        rows = frame.loc[frame["misuse_category"].astype(str) == category].copy()
        rows["_key"] = rows["event_id"].astype(str).map(
            lambda value: hashlib.sha256(
                f"experiment-3|{value}".encode("utf-8")
            ).hexdigest()
        )
        rows = rows.sort_values(["_key", "event_id"], kind="stable")
        adaptation_parts.append(rows.iloc[:50].drop(columns="_key"))
        test_parts.append(rows.iloc[50:].drop(columns="_key"))
    adaptation = pd.concat(adaptation_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    if set(adaptation.event_id) & set(test.event_id):
        raise RuntimeError("External adaptation and Test IDs overlap.")
    if len(adaptation) != 450 or len(test) != 450:
        raise RuntimeError("External split did not produce 450/450 events.")
    return adaptation, test


def adapted_external_records(frame: pd.DataFrame) -> list[dict]:
    records = frame.to_dict(orient="records")
    allowed = set(neural_classifier.OBSERVABLE_TOOLCALL_FIELDS)
    for record_number, record in enumerate(records, 1):
        raw_calls = record.get("tool_calls")
        if raw_calls is None:
            record["tool_calls"] = []
            continue
        if not isinstance(raw_calls, list):
            raise TypeError(f"External record {record_number} has invalid tool_calls.")
        calls = []
        for call_number, raw in enumerate(raw_calls, 1):
            if not isinstance(raw, dict):
                raise TypeError(
                    f"External record {record_number}, call {call_number} is invalid."
                )
            call = {key: value for key, value in raw.items() if key in allowed}
            if not call.get("command") and raw.get("raw_action"):
                call["command"] = str(raw["raw_action"])
            if not call.get("tool_name"):
                raise ValueError(
                    f"External record {record_number}, call {call_number} has no tool_name."
                )
            calls.append(call)
        record["tool_calls"] = calls
    return records


def external_features(
    frame: pd.DataFrame, development_means: dict[str, float]
) -> tuple[pd.DataFrame, Counter]:
    #Use simulated Development means where external telemetry is unavailable
    records = adapted_external_records(frame)
    masks, counts = {}, Counter()
    for field in RAW_NUMERIC:
        mask = np.asarray([
            value is None or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not np.isfinite(value)
            for value in (record.get(field) for record in records)
        ], dtype=bool)
        masks[field] = mask
        counts[field] = int(mask.sum())
        for index in np.flatnonzero(mask):
            records[index][field] = 0
    features = neural_classifier.build_model_features(records).reset_index(drop=True)
    for field, mask in masks.items():
        if mask.any():
            features[field] = features[field].astype(float)
            features.loc[mask, field] = development_means[field]
    return features, counts


def predict(model, features: pd.DataFrame) -> pd.DataFrame:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_, dtype=str)
    return pd.DataFrame({
        "predicted_category": classes[probabilities.argmax(axis=1)],
        "confidence": probabilities.max(axis=1),
        "category_probabilities": [
            dict(zip(classes, row.astype(float))) for row in probabilities
        ],
    })


def load_simulated_test_events(test_frame: pd.DataFrame) -> list[AgentEvent]:
    all_events = symbolic_engine._load_events_for_report()
    by_id = {event.event_id: event for event in all_events}
    ordered = []
    for event_id in test_frame["event_id"].astype(str):
        if event_id not in by_id:
            raise RuntimeError(f"Could not reconstruct simulated event {event_id}.")
        ordered.append(by_id[event_id])
    return ordered


def external_context_events(frame: pd.DataFrame) -> list[AgentEvent]:
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


def print_system_result(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print(f"Accuracy:             {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy:    {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:             {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:          {metrics['weighted_f1']:.4f}")
    print(f"Misuse recall:        {metrics['misuse_recall']:.4f}")
    print(f"Misuse precision:     {metrics['misuse_precision']:.4f}")
    print(f"False positive rate:  {metrics['fpr']:.4f}")
    print(f"False negative rate:  {metrics['fnr']:.4f}")


def main() -> None:
    neural_classifier.set_reproducibility()
    print("Experiment 3: combined neural and fusion")
    print("Loading and verifying both datasets")
    simulated = neural_classifier.load_dataset()
    partitions = neural_classifier.make_grouped_partitions(simulated)
    development = partitions.development.reset_index(drop=True)
    simulated_test = partitions.test.reset_index(drop=True)
    external_adaptation, external_test = split_external(load_external())

    print(f"Simulated Development:       {len(development):,}")
    print(f"External adaptation:         {len(external_adaptation):,} (50 per class)")
    print(f"Combined neural training:    {len(development) + len(external_adaptation):,}")
    print(f"Simulated Test:              {len(simulated_test):,}")
    print(f"External Test:               {len(external_test):,} (50 per class)")

    #Combine simulated and external records for neural training
    print("\nBuilding the full neural feature set")
    development_x = neural_classifier.build_model_features(development)
    development_means = {
        field: float(development_x[field].astype(float).mean())
        for field in RAW_NUMERIC
    }
    external_adaptation_x, _ = external_features(
        external_adaptation, development_means
    )
    external_test_x, _ = external_features(
        external_test, development_means
    )
    simulated_test_x = neural_classifier.build_model_features(simulated_test)
    training_x = pd.concat(
        [development_x, external_adaptation_x], ignore_index=True
    )
    training_y = np.concatenate([
        development["misuse_category"].astype(str).to_numpy(),
        external_adaptation["misuse_category"].astype(str).to_numpy(),
    ])

    print("Training the combined-data neural classifier")
    model = neural_classifier.build_model()
    model.fit(training_x, training_y)
    print("Training complete")

    simulated_neural = predict(model, simulated_test_x)
    external_neural = predict(model, external_test_x)
    simulated_truth = simulated_test["misuse_category"].astype(str).to_numpy()
    external_truth = external_test["misuse_category"].astype(str).to_numpy()

    #Symbolic rules run only where the required simulated telemetry exists
    print("\nRunning symbolic evaluation on simulated Test events")
    reliability_events = load_simulated_test_events(development)
    rule_reliability = neurosymbolic_fusion.calculate_rule_reliability(reliability_events)
    simulated_events = load_simulated_test_events(simulated_test)
    simulated_verdicts = symbolic_engine.evaluate_symbolic(simulated_events)
    simulated_fused = neurosymbolic_fusion.fuse_predictions(
        simulated_events, simulated_verdicts, simulated_neural, rule_reliability
    )
    simulated_fused_labels = np.asarray([
        simulated_fused[event.event_id].final_category for event in simulated_events
    ])
    print("Applying neural fallback where external telemetry is unavailable")
    external_events = external_context_events(external_test)
    unavailable_verdicts = {
        event.event_id: symbolic_engine.SymbolicVerdict() for event in external_events
    }
    external_fused = neurosymbolic_fusion.fuse_predictions(
        external_events, unavailable_verdicts, external_neural, {}
    )
    external_fused_labels = np.asarray([
        external_fused[event.event_id].final_category for event in external_events
    ])
    if not np.array_equal(
        external_fused_labels, external_neural.predicted_category.to_numpy()
    ):
        raise RuntimeError("No-evidence external fusion changed a neural category.")
    rule_events = sum(verdict.fired for verdict in simulated_verdicts.values())

    #Join both test portions for one final comparison
    combined_truth = np.concatenate([simulated_truth, external_truth])
    combined_neural_labels = np.concatenate([
        simulated_neural.predicted_category.to_numpy(),
        external_neural.predicted_category.to_numpy(),
    ])
    combined_fused_labels = np.concatenate([
        simulated_fused_labels, external_fused_labels,
    ])
    simulated_symbolic_labels = np.asarray([
        simulated_verdicts[event.event_id].predicted_category
        for event in simulated_events
    ])
    combined_symbolic_labels = np.concatenate([
        simulated_symbolic_labels,
        np.full(len(external_truth), neural_classifier.BENIGN_LABEL, dtype=object),
    ])
    symbolic_metrics = neurosymbolic_fusion.calculate_full_metrics(
        combined_truth, combined_symbolic_labels
    )
    symbolic_binary_predictions = np.concatenate([
        np.asarray([
            simulated_verdicts[event.event_id].fired
            for event in simulated_events
        ], dtype=bool),
        np.zeros(len(external_truth), dtype=bool),
    ])
    true_misuse = combined_truth != neural_classifier.BENIGN_LABEL
    tp = int(np.sum(symbolic_binary_predictions & true_misuse))
    fp = int(np.sum(symbolic_binary_predictions & ~true_misuse))
    tn = int(np.sum(~symbolic_binary_predictions & ~true_misuse))
    fn = int(np.sum(~symbolic_binary_predictions & true_misuse))
    symbolic_metrics.update({
        "misuse_precision": tp / (tp + fp) if tp + fp else 0.0,
        "misuse_recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    })
    neural_metrics = neurosymbolic_fusion.calculate_full_metrics(
        combined_truth, combined_neural_labels
    )
    fused_metrics = neurosymbolic_fusion.calculate_full_metrics(
        combined_truth, combined_fused_labels
    )

    print("\nCOMBINED DATASET DESIGN")
    print(f"Training: {len(training_y):,} events (simulated + external)")
    print(f"Testing:  {len(combined_truth):,} events (simulated + external)")
    print_system_result(
        "1. COMBINED-DATA NEURAL CLASSIFIER", neural_metrics
    )
    print_system_result(
        "2. COMBINED-DATA FUSED SYSTEM", fused_metrics
    )

    rows = []
    for label, key in (
        ("Accuracy", "accuracy"), ("Balanced accuracy", "balanced_accuracy"),
        ("Macro F1", "macro_f1"), ("Weighted F1", "weighted_f1"),
        ("Misuse precision", "misuse_precision"),
        ("Misuse recall", "misuse_recall"),
        ("False positive rate", "fpr"), ("False negative rate", "fnr"),
    ):
        rows.append({
            "Metric": label,
            "Combined-data symbolic": f"{symbolic_metrics[key]:.4f}",
            "Combined-data neural classifier": f"{neural_metrics[key]:.4f}",
            "Combined-data fused system": f"{fused_metrics[key]:.4f}",
        })
    print("\nTHREE-WAY COMPARISON ON THE SAME COMBINED TEST SET")
    print(pd.DataFrame(rows).set_index("Metric").to_string())
    print("\nSYMBOLIC AVAILABILITY")
    print(
        f"Simulated Test events with symbolic findings: "
        f"{rule_events} / {len(simulated_events)}"
    )
    print(
        "External Test events with symbolic findings: 0 / 450 "
        "(telemetry unavailable; neural fallback used)"
    )
    print(
        "The symbolic column therefore treats external Test events as having "
        "no symbolic finding; it is a telemetry-availability diagnostic."
    )
    print(
        "Interpretation: symbolic rules contribute to simulated events but not "
        "to the external portion. On external events, fused category results "
        "equal neural category results because telemetry is unavailable."
    )


if __name__ == "__main__":
    main()
