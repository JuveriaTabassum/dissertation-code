
"""Evaluates and combines neural and symbolic evidence, reports fused test results, and exports dashboard-ready governance decisions."""

from __future__ import annotations
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
#The fusion layer consumes predictions and evidence produced by both component engines
import neural_classifier
import symbolic_engine
from core import AgentEvent, MisuseCategory, PLATFORM_INVENTORY

#Decision thresholds control confidence and conflict resolution
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_DATA_PATH = OUTPUTS_DIR / "fusion_dashboard_data.json"
RULE_ROLE = symbolic_engine.RULE_ROLE
RELIABLE_DIRECT_THRESHOLD = 0.9
NEURAL_CONFIDENT_THRESHOLD = 0.75
NEURAL_MARGIN_CONFIDENT = 0.3
RULE_RELIABILITY_CLEAR_MARGIN = 0.1


#Complete event-level decision retained for evaluation, governance and the dashboard
@dataclass
class FusionDecision:
    event_id: str
    final_category: str
    confidence: str
    human_review: bool
    decision_case: str
    explanation: str
    reason_for_flagging: str | None
    neural_top_category: str
    neural_p1: float
    neural_p2: float
    neural_margin: float
    neural_probabilities: dict
    direct_findings: list
    supporting_findings: list
    department: str
    task_type: str
    actor_role: str
    ai_platform: str
    timestamp: str
    platform_status: str
    shadow_ai: bool
    governance_outcome: str

    @property
    def is_misuse(self) -> bool:
        return self.final_category != MisuseCategory.BENIGN.value

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "final_category": self.final_category,
            "is_misuse": self.is_misuse,
            "confidence": self.confidence,
            "human_review": self.human_review,
            "decision_case": self.decision_case,
            "explanation": self.explanation,
            "reason_for_flagging": self.reason_for_flagging,
            "neural": {
                "top_category": self.neural_top_category,
                "p1": round(self.neural_p1, 4),
                "p2": round(self.neural_p2, 4),
                "margin": round(self.neural_margin, 4),
                "probabilities": {
                    k: round(v, 4) for k, v in self.neural_probabilities.items()
                },
            },
            "symbolic": {
                "direct_findings": [
                    {"category": c, "rule": r, "explanation": e}
                    for c, r, e in self.direct_findings
                ],
                "supporting_findings": [
                    {"category": c, "rule": r, "explanation": e}
                    for c, r, e in self.supporting_findings
                ],
            },
            "context": {
                "department": self.department,
                "task_type": self.task_type,
                "actor_role": self.actor_role,
                "ai_platform": self.ai_platform,
                "timestamp": self.timestamp,
                "platform_status": self.platform_status,
                "shadow_ai": self.shadow_ai,
            },
            "governance_outcome": self.governance_outcome,
        }


#Extract the top two neural probabilities and their confidence margin
def _neural_reading(neural_row: pd.Series) -> tuple[str, float, float, float, dict]:
    probs: dict = neural_row["category_probabilities"]
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_category, p1 = ranked[0]
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0
    return (top_category, p1, p2, p1 - p2, probs)


def _plain_rule_reason(rule, event: AgentEvent) -> str:
    reasons = {
        "entropy_encoding": "The interaction contained hidden Unicode or encoded text consistent with concealed instructions.",
        "data_to_external_target": "Sensitive information was sent successfully to an external destination.",
        "session_aggregation": f"The session combined sensitive information from {event.systems_touched_count} sources in one {event.task_type.replace('_', ' ')} action.",
        "covert_channel_mismatch": "Sensitive information was moved to an external destination without the response disclosing that action.",
        "confused_deputy": "A requester with restricted authorisation exported confidential or personal information.",
        "deprecated_tool_inventory": "The agent used a deprecated tool that should no longer have been available.",
        "tool_in_approved_set": f"The agent used a tool that is not approved for the {event.task_type.replace('_', ' ')} task.",
        "command_scope": "The agent ran a command outside the permitted scope of the selected tool.",
        "permission_level": "The agent used a connection with more access than the tool required.",
        "identity_scope": "The agent used a shared privileged identity instead of an individual user identity.",
        "approval_check": "The agent completed an irreversible action without recorded human approval.",
        "parameter_range": f"The agent acted on {event.executed_quantity} items although only {event.requested_quantity} were requested.",
        "department_tool_mismatch": f"The agent used a tool that is not normally relevant to the {event.department.replace('_', ' ')} department.",
        "consecutive_autonomous_steps": "The agent completed several consecutive tool actions without an approval checkpoint.",
        "rate_limiter": f"The session made {event.calls_in_time_window} calls within the monitored time window, exceeding the permitted rate.",
        "cumulative_cost": f"The session's cumulative cost rose to {event.cumulative_session_cost:.2f}, exceeding the permitted limit.",
        "cost_input_mismatch": "The agent produced disproportionately high resource usage for the size of the request without using a supporting tool.",
        "tool_call_fanout": "One request triggered an unusually large number of completed downstream tool actions.",
        "query_diversity_frequency": "The session made many rapid and varied queries across multiple query angles.",
        "oversized_output": f"The agent produced an unusually large output of {event.output_token_count} tokens.",
        "precondition_verification": "The response treated a prerequisite as satisfied even though the required irreversible action was not attempted.",
        "known_entity_check": "The response named a software entity that was not present in the known-valid reference list.",
        "completion_outcome_mismatch": "The response claimed that an action was completed even though the recorded tool action failed or was not attempted.",
        "credential_scan": "The response exposed text shaped like a credential or API key.",
        "known_tool_name_scan": "The response exposed the name of an internal tool.",
        "tool_schema_scan": "The response exposed internal tool schema or parameter details.",
        "env_variable_scan": "The response exposed configuration or environment-variable information.",
        "prompt_retrieved_similarity": "The retrieved content was unrelated to the question, indicating that the retrieval result may have been poisoned or misdirected.",
        "attempted_retrieval_refused": "The retrieval succeeded, but the response incorrectly claimed that no information was available.",
        "department_content_mismatch": "The retrieved content belonged to a different department from the requester.",
        "xss_markup_scan": "The agent's output contained executable web markup that could be unsafe if rendered.",
        "path_traversal_scan": "The agent's output contained a path-traversal sequence that could access an unintended file location.",
        "unescaped_email_content": "The agent sent externally directed email content containing an unescaped, phishing-like link.",
        "control_character_check": "The agent's output contained a control character or terminal escape sequence.",
        "covert_channel_mismatch_output": "The agent's output contained an automatically rendered image reference to an external address.",
        "tool_call_ordering_disambiguation": "The agent passed unvalidated SQL for execution even though raw SQL is not an approved command.",
    }
    return reasons.get(
        rule.rule_name,
        f"The event matched a rule for {rule.category.replace('_', ' ')}: {rule.explanation}.",
    )


def _neural_only_reason(category: str, low_confidence: bool) -> str:
    prefix = (
        "The event showed indicators of"
        if low_confidence
        else "The classifier identified"
    )
    descriptions = {
        "prompt_injection": "instruction-manipulation language consistent with prompt injection",
        "sensitive_info_disclosure": "content or behaviour consistent with inappropriate exposure of sensitive information",
        "excessive_agency": "tool-use behaviour consistent with the agent acting beyond its necessary authority",
        "unbounded_consumption": "activity consistent with excessive or insufficiently bounded resource use",
        "misinformation": "a response pattern consistent with unsupported, incorrect or misleading information",
        "hidden_context_exposure": "content consistent with seeking or exposing internal system information",
        "vector_embedding_weakness": "retrieval behaviour consistent with an unsafe or unreliable retrieved result",
        "improper_output_handling": "output or tool-use behaviour consistent with unsafe downstream handling",
    }
    detail = descriptions.get(
        category, f"behaviour consistent with {category.replace('_', ' ')}"
    )
    return f"{prefix} {detail}."


#Build a readable explanation from the strongest evidence supporting the final category
def _build_reason_for_flagging(
    event: AgentEvent,
    final_category: str,
    confidence: str,
    decision_case: str,
    direct: dict[str, list],
    supporting: dict[str, list],
    rule_reliability: dict[str, float],
) -> str | None:
    if final_category == MisuseCategory.BENIGN.value:
        return None
    if decision_case == "F-unresolved":
        base = _neural_only_reason(final_category, True)
        return f"{base[:-1]}, but conflicting symbolic findings mean the alert requires review."
    matching_direct = direct.get(final_category, [])
    if matching_direct:
        decisive = max(
            matching_direct,
            key=lambda rule: _reliability(rule.rule_name, rule_reliability),
        )
        return _plain_rule_reason(decisive, event)
    matching_support = supporting.get(final_category, [])
    if matching_support:
        strongest = max(
            matching_support,
            key=lambda rule: _reliability(rule.rule_name, rule_reliability),
        )
        observation = _plain_rule_reason(strongest, event)
        neural = _neural_only_reason(final_category, confidence == "low")
        return f"{neural[:-1]}; this was supported by the observation that {observation[0].lower() + observation[1:]}"
    return _neural_only_reason(final_category, confidence == "low")


#Measure each rule's own-category precision using Development events
def calculate_rule_reliability(
    development_events: list[AgentEvent],
    development_verdicts: dict[str, symbolic_engine.SymbolicVerdict] | None = None,
) -> dict[str, float]:
    verdicts = development_verdicts
    if verdicts is None:
        verdicts = symbolic_engine.evaluate_symbolic(development_events)
    event_ids = {event.event_id for event in development_events}
    if set(verdicts) != event_ids:
        raise ValueError(
            "Reliability verdicts must correspond exactly to development_events."
        )
    total = Counter()
    correct = Counter()
    for event in development_events:
        for finding in verdicts[event.event_id].fired_rules:
            total[finding.rule_name] += 1
            if finding.category == event.misuse_category.value:
                correct[finding.rule_name] += 1
    unmeasured = sorted(set(RULE_ROLE) - set(total))
    if unmeasured:
        raise ValueError(
            "The training partition contains no firings for active rule(s): "
            + ", ".join(unmeasured)
        )
    return {name: correct[name] / total[name] for name in sorted(RULE_ROLE)}


def _reliability(rule_name: str, rule_reliability: dict[str, float]) -> float:
    try:
        return rule_reliability[rule_name]
    except KeyError as exc:
        raise KeyError(
            f"No training-derived reliability is available for fired rule '{rule_name}'."
        ) from exc


#Combine neural probabilities, symbolic evidence and platform governance for one event
def fuse_event(
    event: AgentEvent,
    symbolic_verdict: symbolic_engine.SymbolicVerdict,
    neural_row: pd.Series,
    rule_reliability: dict[str, float],
) -> FusionDecision:
    neural_cat, p1, p2, margin, probs = _neural_reading(neural_row)
    neural_confident = p1 >= NEURAL_CONFIDENT_THRESHOLD
    neural_says_misuse = neural_cat != MisuseCategory.BENIGN.value
    direct: dict[str, list] = defaultdict(list)
    supporting: dict[str, list] = defaultdict(list)
    for r in symbolic_verdict.fired_rules:
        try:
            role = RULE_ROLE[r.rule_name]
        except KeyError as exc:
            raise KeyError(
                f"No evidence role is defined for fired rule '{r.rule_name}'."
            ) from exc
        if role not in {"direct", "supporting"}:
            raise ValueError(
                f"Invalid evidence role '{role}' for fired rule '{r.rule_name}'."
            )
        (direct if role == "direct" else supporting)[r.category].append(r)
    direct_findings = [
        (c, r.rule_name, r.explanation) for c, rules in direct.items() for r in rules
    ]
    supporting_findings = [
        (c, r.rule_name, r.explanation)
        for c, rules in supporting.items()
        for r in rules
    ]
    direct_categories = set(direct)
    #Platform status is evaluated independently from the misuse decision
    inventory_status = PLATFORM_INVENTORY.get(event.ai_platform)
    if inventory_status is None:
        platform_status = "unknown"
        shadow_ai = True
    else:
        if bool(event.sanctioned_platform) != bool(inventory_status):
            raise ValueError(
                f"Platform status mismatch for {event.ai_platform}: event says sanctioned_platform={event.sanctioned_platform}, inventory says {inventory_status}."
            )
        platform_status = "sanctioned" if inventory_status else "unsanctioned"
        shadow_ai = not inventory_status

    #Finalise the detection, review and governance fields through one shared path
    def _decision(final_category, confidence, human_review, case, explanation):
        is_misuse = final_category != MisuseCategory.BENIGN.value
        if shadow_ai and is_misuse:
            governance_outcome = "shadow_ai_and_agentic_ai_misuse"
        elif shadow_ai:
            governance_outcome = "shadow_ai_usage"
        elif is_misuse:
            governance_outcome = "agentic_ai_misuse"
        else:
            governance_outcome = "normal_activity"
        human_review = bool(human_review or platform_status == "unknown")
        reason_for_flagging = _build_reason_for_flagging(
            event,
            final_category,
            confidence,
            case,
            direct,
            supporting,
            rule_reliability,
        )
        return FusionDecision(
            event_id=event.event_id,
            final_category=final_category,
            confidence=confidence,
            human_review=human_review,
            decision_case=case,
            explanation=explanation,
            reason_for_flagging=reason_for_flagging,
            neural_top_category=neural_cat,
            neural_p1=p1,
            neural_p2=p2,
            neural_margin=margin,
            neural_probabilities=probs,
            direct_findings=direct_findings,
            supporting_findings=supporting_findings,
            department=event.department,
            task_type=event.task_type,
            actor_role=event.actor_role,
            ai_platform=event.ai_platform,
            timestamp=event.timestamp.isoformat(),
            platform_status=platform_status,
            shadow_ai=shadow_ai,
            governance_outcome=governance_outcome,
        )

    def _contradicting_support(final_category: str) -> str | None:
        plausible = {neural_cat}
        second_best = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        if len(second_best) > 1:
            plausible.add(second_best[1][0])
        for cat in supporting:
            if cat != final_category and cat in plausible:
                return cat
        return None

    def _demote_for_contradiction(
        confidence: str, final_category: str
    ) -> tuple[str, str]:
        contradicting = _contradicting_support(final_category)
        if contradicting is None:
            return (confidence, "")
        order = ["low", "medium", "high"]
        demoted = order[max(0, order.index(confidence) - 1)]
        return (demoted, f" Contradicted by supporting evidence for {contradicting}.")

    #Multiple direct categories are ranked by independent evidence families and reliability
    if len(direct_categories) >= 2:
        evidence_family = {
            "credential_scan": "hidden_content",
            "ip_infrastructure_scan": "hidden_content",
            "known_tool_name_scan": "hidden_content",
            "env_variable_scan": "hidden_content",
            "tool_schema_scan": "hidden_content",
            "output_schema_scan": "hidden_content",
            "xss_markup_scan": "unsafe_output",
            "path_traversal_scan": "unsafe_output",
            "unescaped_email_content": "unsafe_output",
            "control_character_check": "unsafe_output",
            "covert_channel_mismatch_output": "unsafe_output",
            "rate_limiter": "resource_use",
            "cumulative_cost": "resource_use",
            "cost_input_mismatch": "resource_use",
            "tool_call_fanout": "resource_use",
            "query_diversity_frequency": "resource_use",
        }

        def symbolic_strength(cat):
            rules = direct[cat]
            best_by_family: dict[str, float] = {}
            for rule in rules:
                family = evidence_family.get(rule.rule_name, rule.rule_name)
                best_by_family[family] = max(
                    best_by_family.get(family, 0.0),
                    _reliability(rule.rule_name, rule_reliability),
                )
            return {
                "family_count": len(best_by_family),
                "best_reliability": max(best_by_family.values()),
                "average_reliability": sum(best_by_family.values())
                / len(best_by_family),
            }

        def ranking_key(cat):
            strength = symbolic_strength(cat)
            return (
                strength["family_count"],
                strength["best_reliability"],
                strength["average_reliability"],
            )

        ranked_categories = sorted(direct_categories, key=ranking_key, reverse=True)
        best_category = ranked_categories[0]
        runner_up = ranked_categories[1]
        best_strength = symbolic_strength(best_category)
        runner_strength = symbolic_strength(runner_up)
        family_advantage = (
            best_strength["family_count"] > runner_strength["family_count"]
        )
        reliability_advantage = (
            best_strength["family_count"] == runner_strength["family_count"]
            and best_strength["best_reliability"]
            >= runner_strength["best_reliability"] + RULE_RELIABILITY_CLEAR_MARGIN
            and (
                best_strength["average_reliability"]
                >= runner_strength["average_reliability"]
            )
        )
        average_advantage = (
            best_strength["family_count"] == runner_strength["family_count"]
            and best_strength["average_reliability"]
            >= runner_strength["average_reliability"] + RULE_RELIABILITY_CLEAR_MARGIN
            and (
                best_strength["best_reliability"] >= runner_strength["best_reliability"]
            )
        )
        clearly_ahead = family_advantage or reliability_advantage or average_advantage
        if clearly_ahead and neural_cat == best_category:
            confidence = "high" if neural_confident and clearly_ahead else "medium"
            return _decision(
                best_category,
                confidence,
                not clearly_ahead,
                "F-agreement-and-strength",
                f"Multiple direct categories fired {sorted(direct_categories)}; {best_category} has the most/most-reliable direct support and matches neural's top category.",
            )
        if clearly_ahead:
            confidence = "medium"
            return _decision(
                best_category,
                confidence,
                True,
                "F-strength-arbitrates",
                f"Multiple direct categories fired {sorted(direct_categories)}; {best_category} has clearly stronger direct support ({best_strength['family_count']} independent evidence family/families), but doesn't match neural's top category ({neural_cat}) - routed for review.",
            )
        unresolved_category = neural_cat
        return _decision(
            unresolved_category,
            "low",
            True,
            "F-unresolved",
            f"Multiple direct categories fired {sorted(direct_categories)} with no clear symbolic winner; the neural category ({neural_cat}) is retained at low confidence and routed for review.",
        )
    #One direct category is compared with neural agreement and the rule's reliability
    if len(direct_categories) == 1:
        category = next(iter(direct_categories))
        best_reliability = max(
            (_reliability(r.rule_name, rule_reliability) for r in direct[category])
        )
        reliable = best_reliability >= RELIABLE_DIRECT_THRESHOLD
        if neural_cat == category:
            confidence, note = _demote_for_contradiction("high", category)
            return _decision(
                category,
                confidence,
                confidence == "low",
                "C-agreement",
                f"Neural top category and direct symbolic evidence agree on {category} (rule reliability {best_reliability:.2f}).{note}",
            )
        if not neural_says_misuse or not neural_confident:
            base_confidence = "high" if reliable else "medium"
            confidence, note = _demote_for_contradiction(base_confidence, category)
            return _decision(
                category,
                confidence,
                confidence == "low",
                "B-symbolic-rescue",
                f"Direct symbolic evidence establishes {category} (rule reliability {best_reliability:.2f}) despite neural predicting {('benign' if not neural_says_misuse else f'{neural_cat} at only p={p1:.2f}')}.{note}",
            )
        if reliable:
            return _decision(
                category,
                "medium",
                False,
                "E-conflict-symbolic-favoured",
                f"Conflict: neural confidently predicts {neural_cat} (p={p1:.2f}) but a reliable direct rule (reliability {best_reliability:.2f}) establishes {category}; directly observed structural evidence is favoured.",
            )
        return _decision(
            neural_cat,
            "low",
            True,
            "E-conflict-neural-favoured",
            f"Conflict: neural confidently predicts {neural_cat} (p={p1:.2f}) against a lower-reliability direct rule for {category} (reliability {best_reliability:.2f}); neither is decisive, routed for review.",
        )
    #Without direct evidence the neural decision is adjusted by supporting findings
    if neural_says_misuse:
        same_category_support = bool(supporting.get(neural_cat))
        contradicting = _contradicting_support(neural_cat)
        if contradicting:
            confidence = "low"
            case = "A-supporting-contradicts"
            note = f" Supporting evidence instead points toward {contradicting}."
        elif neural_confident and same_category_support:
            confidence = "high"
            case = "A-neural-and-supporting-agree"
            note = " Supporting symbolic evidence corroborates the same category."
        elif neural_confident:
            confidence = "medium"
            case = "A-neural-only"
            note = ""
        elif margin >= NEURAL_MARGIN_CONFIDENT or same_category_support:
            confidence = "medium"
            case = (
                "A-neural-moderate"
                if margin >= NEURAL_MARGIN_CONFIDENT
                else "A-supporting-corroborates"
            )
            note = (
                " Supporting symbolic evidence corroborates the same category."
                if same_category_support
                else ""
            )
        else:
            confidence = "low"
            case = "A-neural-weak"
            note = ""
        review = confidence == "low"
        return _decision(
            neural_cat,
            confidence,
            review,
            case,
            f"No direct symbolic rule fired; neural predicts {neural_cat} (p={p1:.2f}, margin={margin:.2f}).{note}",
        )
    #Benign predictions with unexplained supporting evidence are sent for review
    if supporting:
        return _decision(
            MisuseCategory.BENIGN.value,
            "low",
            True,
            "benign-with-supporting",
            f"Neural predicts benign (p={p1:.2f}) with no direct symbolic evidence, but unexplained supporting finding(s) present: {sorted(supporting)}.",
        )
    if neural_confident:
        return _decision(
            MisuseCategory.BENIGN.value,
            "high",
            False,
            "benign-confident",
            f"Neural confidently predicts benign (p={p1:.2f}); no symbolic evidence at all.",
        )
    if margin >= NEURAL_MARGIN_CONFIDENT:
        return _decision(
            MisuseCategory.BENIGN.value,
            "medium",
            False,
            "benign-moderate",
            f"Neural predicts benign with a clear margin ({margin:.2f}) though p1={p1:.2f}; no symbolic evidence.",
        )
    return _decision(
        MisuseCategory.BENIGN.value,
        "low",
        True,
        "benign-weak",
        f"Neural weakly predicts benign (p={p1:.2f}, margin={margin:.2f}); no symbolic evidence either way, routed for review.",
    )


#Preserve event order while applying the fusion policy across the test set
def fuse_predictions(
    events: list[AgentEvent],
    symbolic_verdicts: dict[str, symbolic_engine.SymbolicVerdict],
    neural_predictions: pd.DataFrame,
    rule_reliability: dict[str, float],
) -> dict[str, FusionDecision]:
    if len(neural_predictions) != len(events):
        raise ValueError(
            f"neural_predictions has {len(neural_predictions)} rows but {len(events)} events were given - they must be computed on the same event list, in the same order."
        )
    fused: dict[str, FusionDecision] = {}
    for event, (_, neural_row) in zip(events, neural_predictions.iterrows()):
        fused[event.event_id] = fuse_event(
            event, symbolic_verdicts[event.event_id], neural_row, rule_reliability
        )
    return fused


#Use the same multiclass and binary metrics for all three compared systems
def calculate_full_metrics(true_labels: np.ndarray, predictions: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    true_labels = np.asarray(true_labels)
    predictions = np.asarray(predictions)
    benign = MisuseCategory.BENIGN.value
    true_misuse = true_labels != benign
    predicted_misuse = predictions != benign
    tn, fp, fn, tp = confusion_matrix(
        true_misuse, predicted_misuse, labels=[False, True]
    ).ravel()
    fpr = (
        float(np.mean(predicted_misuse[~true_misuse])) if (~true_misuse).any() else 0.0
    )
    fnr = float(np.mean(~predicted_misuse[true_misuse])) if true_misuse.any() else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    true_label_names = sorted(set(true_labels))
    return {
        "macro_f1": f1_score(
            true_labels,
            predictions,
            labels=true_label_names,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            true_labels,
            predictions,
            labels=true_label_names,
            average="weighted",
            zero_division=0,
        ),
        "balanced_accuracy": recall_score(
            true_labels,
            predictions,
            labels=true_label_names,
            average="macro",
            zero_division=0,
        ),
        "accuracy": accuracy_score(true_labels, predictions),
        "misuse_recall": recall_score(true_misuse, predicted_misuse, zero_division=0),
        "misuse_precision": precision,
        "fpr": fpr,
        "fnr": fnr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


#Measure confidence, review routing and automation coverage produced by fusion
def calculate_fusion_specific_metrics(
    events: list[AgentEvent], fused: dict[str, FusionDecision]
) -> dict:
    true_labels = np.array([e.misuse_category.value for e in events])
    predictions = np.array([fused[e.event_id].final_category for e in events])
    correct = predictions == true_labels
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    reviewed = reviewed_correct = not_reviewed_correct = 0
    fp_reviewed = fp_total = fn_reviewed = fn_total = 0
    benign = MisuseCategory.BENIGN.value
    for e in events:
        d = fused[e.event_id]
        confidence_counts[d.confidence] += 1
        true_misuse = e.misuse_category.value != benign
        pred_misuse = d.final_category != benign
        is_correct = d.final_category == e.misuse_category.value
        if d.human_review:
            reviewed += 1
            reviewed_correct += is_correct
        else:
            not_reviewed_correct += is_correct
        if pred_misuse and (not true_misuse):
            fp_total += 1
            fp_reviewed += d.human_review
        if not pred_misuse and true_misuse:
            fn_total += 1
            fn_reviewed += d.human_review
    n = len(events)
    not_reviewed = n - reviewed
    total_errors = int((~correct).sum())
    reviewed_errors = reviewed - reviewed_correct
    return {
        "confidence_distribution": confidence_counts,
        "pct_high": confidence_counts["high"] / n,
        "pct_medium": confidence_counts["medium"] / n,
        "pct_low": confidence_counts["low"] / n,
        "pct_routed_for_review": reviewed / n,
        "accuracy_among_reviewed": (
            reviewed_correct / reviewed if reviewed else float("nan")
        ),
        "accuracy_among_not_reviewed": (
            not_reviewed_correct / not_reviewed if not_reviewed else float("nan")
        ),
        "overall_accuracy": float(correct.mean()),
        "automation_coverage": not_reviewed / n,
        "review_error_rate": reviewed_errors / reviewed if reviewed else float("nan"),
        "pct_all_errors_caught_by_review": (
            reviewed_errors / total_errors if total_errors else float("nan")
        ),
        "pct_false_positives_caught_by_review": (
            fp_reviewed / fp_total if fp_total else float("nan")
        ),
        "pct_false_negatives_caught_by_review": (
            fn_reviewed / fn_total if fn_total else float("nan")
        ),
        "fp_total": fp_total,
        "fn_total": fn_total,
    }


#Separate Shadow AI status from misuse and compare sanctioned and unsanctioned groups
def calculate_governance_metrics(
    events: list[AgentEvent], fused: dict[str, FusionDecision]
) -> dict:
    outcomes = Counter((fused[e.event_id].governance_outcome for e in events))
    platform_statuses = Counter((fused[e.event_id].platform_status for e in events))
    shadow_events = [e for e in events if fused[e.event_id].shadow_ai]
    sanctioned_events = [e for e in events if not fused[e.event_id].shadow_ai]

    def subgroup_metrics(subgroup: list[AgentEvent]) -> dict | None:
        if not subgroup:
            return None
        truth = np.array([e.misuse_category.value for e in subgroup])
        predictions = np.array([fused[e.event_id].final_category for e in subgroup])
        return calculate_full_metrics(truth, predictions)

    return {
        "platform_statuses": platform_statuses,
        "governance_outcomes": outcomes,
        "shadow_ai_metrics": subgroup_metrics(shadow_events),
        "sanctioned_metrics": subgroup_metrics(sanctioned_events),
    }


def print_metrics_table(rows: dict[str, dict]) -> None:
    keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "misuse_recall",
        "misuse_precision",
        "fpr",
        "fnr",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    name_width = max((len(name) for name in rows)) + 2
    print(f"{'metric':22s}" + "".join((f"{name:>{name_width}s}" for name in rows)))
    for key in keys:
        values = [rows[name][key] for name in rows]
        formatted = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in values]
        print(f"{key:22s}" + "".join((f"{v:>{name_width}s}" for v in formatted)))


#Convert into valid json values
def _json_safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


#Export fused data for dashboard
def export_dashboard_data(
    events: list[AgentEvent],
    fused: dict[str, FusionDecision],
    metrics: dict,
    fusion_metrics: dict,
    governance_metrics: dict,
    total_event_count: int,
    output_path: Path = DASHBOARD_DATA_PATH,
) -> Path:
    event_records = []
    for event in events:
        decision_record = fused[event.event_id].to_dict()
        decision_record.update(
            {
                "actual_category": event.misuse_category.value,
                "actual_pattern": event.misuse_pattern,
                "prompt": event.prompt,
                "model_output": event.model_output,
                "reasoning_trace": event.reasoning_trace,
                "retrieved_content": event.retrieved_content,
                "content_source": event.content_source,
                "tool_calls": [
                    {
                        "tool_name": call.tool_name,
                        "command": call.command,
                        "parameters": call.parameters,
                        "permission_level": call.permission_level,
                        "identity_scope": call.identity_scope,
                        "reversibility": call.reversibility,
                        "outcome": call.outcome,
                        "destination": call.destination,
                    }
                    for call in event.tool_calls
                ],
            }
        )
        event_records.append(decision_record)
    payload = _json_safe(
        {
            "schema_version": 2,
            "dataset_event_count": total_event_count,
            "evaluated_event_count": len(events),
            "metrics": metrics,
            "fusion_metrics": fusion_metrics,
            "governance_metrics": governance_metrics,
            "events": event_records,
        }
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary_path.replace(output_path)
    return output_path


#Run symbolic, neural and fused evaluation on the same held-out test events
def run_fusion_report(dataset_path: Path | None = None, model_path: Path | None = None):
    if dataset_path is None:
        dataset_path = OUTPUTS_DIR / "simulated_events.jsonl"
    print("Loading Dataset..")
    all_events = symbolic_engine._load_events_for_report(dataset_path)
    print(f"Dataset Loaded: {len(all_events)} events")
    test_ids = symbolic_engine._select_test_event_ids(all_events)
    development_events = [e for e in all_events if e.event_id not in test_ids]
    test_events = [e for e in all_events if e.event_id in test_ids]
    test_events.sort(key=lambda e: e.event_id)
    print(
        f"Test set: {len(test_events)} / {len(all_events)} events ({len(test_events) / len(all_events):.1%})"
    )
    symbolic_verdicts = symbolic_engine.evaluate_symbolic(test_events)
    rule_reliability = calculate_rule_reliability(development_events)
    print("Loading trained classifier and predicting on test set...")
    bundle = (
        neural_classifier.load_model_bundle(model_path)
        if model_path
        else neural_classifier.load_model_bundle()
    )
    records = pd.DataFrame([e.to_record() for e in test_events])
    neural_predictions = neural_classifier.predict_records(bundle, records)
    neural_predictions.index = range(len(test_events))
    print("Generating fused decisions...")
    fused = fuse_predictions(
        test_events, symbolic_verdicts, neural_predictions, rule_reliability
    )
    true_labels = np.array([e.misuse_category.value for e in test_events])
    symbolic_predictions = np.array(
        [symbolic_verdicts[e.event_id].predicted_category for e in test_events]
    )
    neural_predictions_arr = neural_predictions["predicted_category"].to_numpy()
    fused_predictions = np.array(
        [fused[e.event_id].final_category for e in test_events]
    )
    symbolic_metrics = calculate_full_metrics(true_labels, symbolic_predictions)
    symbolic_binary_metrics, _, _ = symbolic_engine.calculate_symbolic_metrics(
        test_events, symbolic_verdicts
    )
    for key in (
        "misuse_recall",
        "misuse_precision",
        "fpr",
        "fnr",
        "tp",
        "fp",
        "tn",
        "fn",
    ):
        symbolic_metrics[key] = symbolic_binary_metrics[key]
    metrics_rows = {
        "symbolic": symbolic_metrics,
        "neural": calculate_full_metrics(true_labels, neural_predictions_arr),
        "fused": calculate_full_metrics(true_labels, fused_predictions),
    }
    from sklearn.metrics import (
        classification_report,
        confusion_matrix as sk_confusion_matrix,
    )

    all_labels = sorted(set(true_labels))
    print("\nNeurosymbolic Fused Engine Evaluation (On Test):")
    print(
        classification_report(
            true_labels, fused_predictions, labels=all_labels, zero_division=0
        )
    )
    fused_binary = metrics_rows["fused"]
    print("Binary Confusion Matrix (Misuse vs Benign):")
    print(f"{'':18s}{'Predicted Misuse':>20s}{'Predicted Benign':>20s}")
    print(f"{'Actual Misuse':18s}{fused_binary['tp']:>20d}{fused_binary['fn']:>20d}")
    print(f"{'Actual Benign':18s}{fused_binary['fp']:>20d}{fused_binary['tn']:>20d}")
    print("\nConfusion Matrix (Neurosymbolic Fused Engine):")
    cm = sk_confusion_matrix(true_labels, fused_predictions, labels=all_labels)
    header = "".join((f"{lbl[:10]:>12s}" for lbl in all_labels))
    print(" " * 22 + header)
    for lbl, row in zip(all_labels, cm):
        print(f"{lbl[:22]:22s}" + "".join((f"{v:>12d}" for v in row)))
    print("\nThree Way Comparison (On Test):")
    print_metrics_table(metrics_rows)
    print("\nFusion-Specific Metrics:")
    fusion_metrics = calculate_fusion_specific_metrics(test_events, fused)
    print(f"\nConfidence distribution: {fusion_metrics['confidence_distribution']}")
    print(f"  high:   {fusion_metrics['pct_high']:.1%}")
    print(f"  medium: {fusion_metrics['pct_medium']:.1%}")
    print(f"  low:    {fusion_metrics['pct_low']:.1%}")
    print(f"\nRouted for human review: {fusion_metrics['pct_routed_for_review']:.1%}")
    print(
        f"Automation coverage:             {fusion_metrics['automation_coverage']:.1%}"
    )
    print(
        f"Accuracy among reviewed events:     {fusion_metrics['accuracy_among_reviewed']:.4f}"
    )
    print(
        f"Accuracy among NOT-reviewed events:  {fusion_metrics['accuracy_among_not_reviewed']:.4f}"
    )
    print(
        f"Overall accuracy:                    {fusion_metrics['overall_accuracy']:.4f}"
    )
    print(
        f"Error rate among reviewed events:    {fusion_metrics['review_error_rate']:.1%}"
    )
    print(
        f"All errors routed for review:        {fusion_metrics['pct_all_errors_caught_by_review']:.1%}"
    )
    print(
        f"\nFalse positives caught by review: {fusion_metrics['pct_false_positives_caught_by_review']:.1%} ({fusion_metrics['fp_total']} total FP)"
    )
    print(
        f"False negatives caught by review: {fusion_metrics['pct_false_negatives_caught_by_review']:.1%} ({fusion_metrics['fn_total']} total FN)"
    )
    governance_metrics = calculate_governance_metrics(test_events, fused)
    print("\nShadow AI and Agentic AI Governance Results:")
    for status in ("sanctioned", "unsanctioned", "unknown"):
        count = governance_metrics["platform_statuses"].get(status, 0)
        print(
            f"  {status.replace('_', ' ').title():24s} {count:5d}  ({count / len(test_events):.1%})"
        )
    print("\nGovernance outcomes:")
    outcome_labels = {
        "normal_activity": "Normal activity",
        "shadow_ai_usage": "Shadow AI usage without detected misuse",
        "agentic_ai_misuse": "Agentic AI misuse on a sanctioned platform",
        "shadow_ai_and_agentic_ai_misuse": "Shadow AI and agentic AI misuse",
    }
    for outcome in (
        "normal_activity",
        "shadow_ai_usage",
        "agentic_ai_misuse",
        "shadow_ai_and_agentic_ai_misuse",
    ):
        count = governance_metrics["governance_outcomes"].get(outcome, 0)
        print(
            f"  {outcome_labels[outcome]:48s} {count:5d}  ({count / len(test_events):.1%})"
        )
    subgroup_rows = {}
    if governance_metrics["sanctioned_metrics"]:
        subgroup_rows["sanctioned"] = governance_metrics["sanctioned_metrics"]
    if governance_metrics["shadow_ai_metrics"]:
        subgroup_rows["shadow_ai"] = governance_metrics["shadow_ai_metrics"]
    if subgroup_rows:
        print("\nMisuse detection by platform status:")
        print_metrics_table(subgroup_rows)
    print("\nFusion Decision Breakdown:")
    case_labels = {
        "C-agreement": "Neural and direct-rule agreement",
        "A-neural-only": "Neural decision without direct rule",
        "benign-confident": "Clear benign decision",
        "B-symbolic-rescue": "Direct rule corrected neural decision",
        "A-neural-and-supporting-agree": "Neural and supporting-rule agreement",
        "E-conflict-symbolic-favoured": "Reliable direct rule resolved conflict",
        "F-agreement-and-strength": "Multiple direct findings resolved",
        "A-neural-moderate": "Moderate neural decision",
        "A-neural-weak": "Weak neural decision",
        "benign-weak": "Uncertain benign decision",
        "benign-with-supporting": "Benign decision with supporting finding",
        "benign-moderate": "Moderate benign decision",
        "A-supporting-contradicts": "Neural and supporting evidence conflict",
        "E-conflict-neural-favoured": "Neural decision with unresolved direct conflict",
        "A-supporting-corroborates": "Supporting rule corroborated neural decision",
        "F-strength-arbitrates": "Strongest direct evidence selected",
        "F-unresolved": "Unresolved multiple-rule conflict",
    }
    case_counts = Counter((fused[e.event_id].decision_case for e in test_events))
    for case, count in case_counts.most_common():
        label = case_labels.get(case, case)
        print(f"  {label:48s} {count:5d}  ({count / len(test_events):.1%})")
    dashboard_data_path = export_dashboard_data(
        test_events,
        fused,
        metrics_rows,
        fusion_metrics,
        governance_metrics,
        len(all_events),
    )
    print("\nRule Reliability Table (from Development):")
    for rule_name, value in rule_reliability.items():
        print(f"  {rule_name:42s} {value:.4f}")
    print(f"\nDashboard data saved to: {dashboard_data_path.name}")
    return {
        "test_events": test_events,
        "symbolic_verdicts": symbolic_verdicts,
        "neural_predictions": neural_predictions,
        "fused": fused,
        "metrics": metrics_rows,
        "fusion_metrics": fusion_metrics,
        "governance_metrics": governance_metrics,
        "rule_reliability": rule_reliability,
        "dashboard_data_path": dashboard_data_path,
    }

if __name__ == "__main__":
    run_fusion_report()
