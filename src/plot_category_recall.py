
"""Creates Figure: per-category recall comparison for the final held-out Test set."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score

import neural_classifier
import symbolic_engine
import neurosymbolic_fusion

OUTPUT_PATH = Path(__file__).resolve().parent / "category_recall_comparison.png"

CATEGORY_LABELS = {
    "prompt_injection": "Prompt\ninjection",
    "sensitive_info_disclosure": "Sensitive info\ndisclosure",
    "excessive_agency": "Excessive\nagency",
    "unbounded_consumption": "Unbounded\nconsumption",
    "misinformation": "Misinformation",
    "hidden_context_exposure": "Hidden context\nexposure",
    "vector_embedding_weakness": "Vector / embedding\nweakness",
    "improper_output_handling": "Improper output\nhandling",
}

def main():
    all_events = symbolic_engine._load_events_for_report()
    test_ids = symbolic_engine._select_test_event_ids(all_events)
    development_events = [e for e in all_events if e.event_id not in test_ids]
    test_events = sorted(
        [e for e in all_events if e.event_id in test_ids],
        key=lambda e: e.event_id,
    )

    symbolic_verdicts = symbolic_engine.evaluate_symbolic(test_events)
    rule_reliability = neurosymbolic_fusion.calculate_rule_reliability(development_events)

    bundle = neural_classifier.load_model_bundle()
    records = neural_classifier.pd.DataFrame([e.to_record() for e in test_events])
    neural_predictions = neural_classifier.predict_records(bundle, records)
    neural_predictions.index = range(len(test_events))

    fused = neurosymbolic_fusion.fuse_predictions(
        test_events, symbolic_verdicts, neural_predictions, rule_reliability
    )

    true_labels = np.array([e.misuse_category.value for e in test_events])
    symbolic_predictions = np.array(
        [symbolic_verdicts[e.event_id].predicted_category for e in test_events]
    )
    neural_predictions_array = neural_predictions["predicted_category"].to_numpy()
    fused_predictions = np.array([fused[e.event_id].final_category for e in test_events])

    categories = list(CATEGORY_LABELS)
    symbolic_recall = recall_score(
        true_labels, symbolic_predictions, labels=categories, average=None, zero_division=0
    )
    neural_recall = recall_score(
        true_labels, neural_predictions_array, labels=categories, average=None, zero_division=0
    )
    fused_recall = recall_score(
        true_labels, fused_predictions, labels=categories, average=None, zero_division=0
    )

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, symbolic_recall, width, label="Symbolic", color="#9EC5E6")
    ax.bar(x, neural_recall, width, label="Neural", color="#F2BE86")
    ax.bar(x + width, fused_recall, width, label="Neurosymbolic", color="#9FCFA8")

    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in categories], fontsize=8)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Chart saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
