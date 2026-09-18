
"""Generates a reproducible simulated dataset of Agentic AI and Shadow AI events from configurable scenario templates."""

from __future__ import annotations
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
import pandas as pd
import yaml
#Shared schemas and reference values keep generated events consistent with the detectors
from core import (
    AgentEvent,
    InteractionOutcome,
    MisuseCategory,
    PATTERNS,
    ToolCall,
    AGGREGATION_SYSTEM_POOLS,
    DEPARTMENTS,
    DEPARTMENTS_FOR_TOOL,
    PLATFORM_INVENTORY,
)

#Sample operational fields from skewed distributions instead of fixed values
def _lognormal_int(mean: float, sigma: float, floor: int = 1) -> int:
    return max(floor, round(random.lognormvariate(mean, sigma)))


def _poisson(lam: float, floor: int = 1) -> int:
    import math

    l = math.exp(-lam)
    k, p = (0, 1.0)
    while p > l:
        k += 1
        p *= random.random()
    return max(floor, k - 1)


def calls_in_time_window(elevated: bool) -> int:
    if elevated:
        return _lognormal_int(mean=3.0, sigma=0.5, floor=6)
    return _lognormal_int(mean=1.1, sigma=0.55, floor=1)


def cumulative_session_cost(elevated: bool) -> float:
    if elevated:
        return round(max(0.05, random.lognormvariate(0.9, 0.55)), 2)
    return round(max(0.01, random.lognormvariate(-2.2, 1.1)), 2)


def output_token_count(elevated: bool) -> int:
    if elevated:
        return _lognormal_int(mean=7.4, sigma=0.6, floor=300)
    if random.random() < 0.05:
        return _lognormal_int(mean=7.0, sigma=0.45, floor=350)
    return _lognormal_int(mean=4.0, sigma=0.55, floor=15)


def systems_touched_count(scale: str) -> int:
    if scale == "fanout":
        return _poisson(18, floor=10) + 3
    if scale == "extraction":
        return _poisson(6, floor=4) + 2
    if scale == "aggregation":
        return _poisson(2.5, floor=1) + 1
    if random.random() < 0.05:
        return random.randint(5, 10)
    return _poisson(0.6, floor=1)


def requested_quantity(elevated: bool) -> int:
    if elevated:
        return _lognormal_int(mean=5.0, sigma=1.0, floor=25)
    if random.random() < 0.05:
        return _lognormal_int(mean=5.0, sigma=0.65, floor=25)
    return _poisson(0.8, floor=1)

#reproducibility, output location and dataset size
RANDOM_SEED = 42
DATASET_END_TIME = datetime(2026, 9, 14, 23, 59, 59, tzinfo=timezone.utc)
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIG_PATH = THIS_DIR / "dataset_scenarios.yaml"
ACTOR_ROLES = ("employee", "contractor", "service_account")
EVENTS_PER_TEMPLATE = 17
SHADOW_AI_FRACTION = 0.3
DATASET_TIME_SPAN_DAYS = 120


def _random_timestamp() -> datetime:
    offset_seconds = random.uniform(0, DATASET_TIME_SPAN_DAYS * 86400)
    return DATASET_END_TIME - timedelta(seconds=offset_seconds)


AGGREGATION_PATTERN = "sensitive_info_disclosure.aggregation"
BURST_PATTERN = "unbounded_consumption.burst_repetition"
#Links selected misuse patterns to the telemetry signal they should elevate
ELEVATED_FIELDS: dict[str, dict[str, bool | str]] = {
    "unbounded_consumption.context_creep": {"cumulative_session_cost": True},
    "unbounded_consumption.reasoning_loop_spike": {"output_token_count": True},
    "unbounded_consumption.tool_call_fanout": {"systems_touched_count": "fanout"},
    "unbounded_consumption.model_extraction_querying": {
        "calls_in_time_window": True,
        "systems_touched_count": "extraction",
    },
    "excessive_agency.parameter_pollution": {"executed_quantity": True},
}

#Apply normal variation while preserving the stronger signal required by each pattern
def apply_stochastic_fields(
    event: AgentEvent,
    category: MisuseCategory,
    pattern_id: str,
    template: dict | None = None,
) -> None:
    key = f"{category.value}.{pattern_id}"
    elevated = ELEVATED_FIELDS.get(key, {})
    template = template or {}
    if key != BURST_PATTERN:
        event.calls_in_time_window = calls_in_time_window(
            elevated.get("calls_in_time_window", False)
        )
    event.cumulative_session_cost = cumulative_session_cost(
        elevated.get("cumulative_session_cost", False)
    )
    if "output_token_count" in template:
        event.output_token_count = int(template["output_token_count"])
    else:
        event.output_token_count = output_token_count(
            elevated.get("output_token_count", False)
        )
    if "requested_quantity" in template:
        event.requested_quantity = template["requested_quantity"]
    else:
        event.requested_quantity = requested_quantity(False)
    if "executed_quantity" in template:
        event.executed_quantity = template["executed_quantity"]
    elif elevated.get("executed_quantity", False):
        event.executed_quantity = requested_quantity(True)
    else:
        event.executed_quantity = event.requested_quantity
    if key != AGGREGATION_PATTERN:
        event.systems_touched_count = systems_touched_count(
            elevated.get("systems_touched_count", "normal")
        )


def load_config() -> dict: #Load the wording templates and substitution pools from dataset_scenarios.yaml
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_substitution_values(substitution_values: dict) -> dict:
    return {name: random.choice(values) for name, values in substitution_values.items()}

#Wording wrappers create variations without changing the underlying scenario meaning
PROMPT_RENDERINGS: tuple[str, ...] = (
    "{text}",
    "Please handle this request: {text}",
    "Could you assist with the following? {text}",
    "Here is the task for this interaction: {text}",
    "Please process the following request: {text}",
    "I need help with this task: {text}",
    "Could you take care of this request? {text}",
    "Please review and act on the following: {text}",
    "The next task is as follows: {text}",
    "Can you handle this for me? {text}",
    "Please work through this request: {text}",
    "I would like assistance with the following: {text}",
    "Could you process this task? {text}",
    "Please address the following request: {text}",
    "Here is what I need help with: {text}",
    "Can you complete the following task? {text}",
    "Please respond to this request: {text}",
)
PROMPT_CONTEXT_SUFFIXES: tuple[str, ...] = (
    "",
    " Please treat this as part of the current session.",
    " Use the context already available in this interaction.",
    " Return the result through the normal workflow.",
)
NEUTRAL_PROMPT_RENDERINGS: tuple[str, ...] = (
    "{text}",
    "For this request: {text}",
    "For the current task: {text}",
    "Please handle the following: {text}",
    "Please review the following: {text}",
    "The requested task is: {text}",
    "Here is the request: {text}",
    "Please help with this: {text}",
)
ALREADY_POLITE_RENDERINGS: tuple[str, ...] = (
    "{text}",
    "For this request: {text}",
    "For the current task: {text}",
    "The requested task is: {text}",
    "Here is the request: {text}",
)


def apply_prompt_rendering(event: AgentEvent, variant_index: int) -> None:
    original = event.prompt.lstrip().lower()
    already_polite = original.startswith("please ")
    already_framed = original.startswith(
        (
            "could ",
            "can ",
            "would ",
            "will ",
            "what ",
            "why ",
            "how ",
            "when ",
            "where ",
            "who ",
            "is ",
            "are ",
            "do ",
            "does ",
            "did ",
        )
    )
    if already_polite:
        renderings = ALREADY_POLITE_RENDERINGS
    elif already_framed:
        renderings = NEUTRAL_PROMPT_RENDERINGS
    else:
        renderings = PROMPT_RENDERINGS
    rendering_count = len(renderings)
    rendering = renderings[variant_index % rendering_count]
    suffix = PROMPT_CONTEXT_SUFFIXES[
        variant_index // rendering_count % len(PROMPT_CONTEXT_SUFFIXES)
    ]
    event.prompt = rendering.format(text=event.prompt) + suffix

#Prefer departments where the template's tools are operationally relevant
def _pick_department(template: dict) -> str:
    candidates: list[str] = []
    for tc in template.get("tool_calls", []):
        candidates.extend(DEPARTMENTS_FOR_TOOL.get(tc["tool_name"], []))
    owner = template.get("content_owner_department")
    if owner:
        candidates = [d for d in candidates if d != owner] or [
            d for d in DEPARTMENTS if d != owner
        ]
    return random.choice(candidates) if candidates else random.choice(DEPARTMENTS)


def sample_platform(force_shadow: bool) -> tuple[str, bool]:
    wanted_sanctioned = not force_shadow
    pool = [
        name
        for name, sanctioned in PLATFORM_INVENTORY.items()
        if sanctioned == wanted_sanctioned
    ]
    platform_id = random.choice(pool)
    return (platform_id, PLATFORM_INVENTORY[platform_id])


EXTERNAL_CAPABLE_TOOLS = {
    "send_external_email",
    "external_posting_tool",
    "file_upload_tool",
}

#Convert YAML tool definitions into the ToolCall schema from core.py
def _build_tool_calls(template: dict, sampled: dict) -> list[ToolCall]:
    calls = []
    for tc in template.get("tool_calls", []):
        target_var = tc.get("target_var")
        target = sampled[target_var] if target_var else tc.get("target", "")
        if "destination" in tc:
            destination = tc["destination"]
        elif tc["tool_name"] in EXTERNAL_CAPABLE_TOOLS and target:
            destination = target
        else:
            destination = "internal"
        calls.append(
            ToolCall(
                tool_name=tc["tool_name"],
                command=tc.get("command", "read"),
                parameters={"target": target},
                permission_level=tc.get("permission_level", "read"),
                identity_scope=tc.get("identity_scope", "per_user"),
                reversibility=tc.get("reversibility", "reversible"),
                outcome=tc.get("outcome", "success"),
                destination=destination,
                out_of_scope=tc.get("out_of_scope", False),
            )
        )
    return calls

#Build one complete event and apply defaults for fields not specified by the template
def _fill_event(
    category: MisuseCategory,
    pattern_id: str,
    template: dict,
    sampled: dict,
    session_id: str,
    actor_role: str,
    department: str,
    force_shadow: bool,
    timestamp: datetime,
    split_group_id: str,
) -> AgentEvent:
    ai_platform, sanctioned = sample_platform(force_shadow)
    prompt = template["prompt"].format(**sampled)
    output = template["output"].format(**sampled)
    reasoning_trace = template.get("reasoning_trace", "").format(**sampled)
    retrieved_content = template.get("retrieved_content", "").format(**sampled)
    default_outcome = (
        InteractionOutcome.BENIGN_COMPLETION
        if category is MisuseCategory.BENIGN
        else InteractionOutcome.UNSAFE_COMPLIANCE
    )
    outcome_override = template.get("interaction_outcome")
    interaction_outcome = (
        InteractionOutcome(outcome_override) if outcome_override else default_outcome
    )
    event_id = str(
        uuid5(
            NAMESPACE_URL,
            "|".join(
                (
                    template["id"],
                    session_id,
                    timestamp.isoformat(),
                    category.value,
                    pattern_id,
                )
            ),
        )
    )
    tool_calls = _build_tool_calls(template, sampled)
    owner_department = template.get("content_owner_department")
    if owner_department is None and (
        retrieved_content
        or any((call.tool_name == "rag_retriever" for call in tool_calls))
    ):
        owner_department = department
    return AgentEvent(
        event_id=event_id,
        session_id=session_id,
        actor_role=actor_role,
        department=department,
        prompt=prompt,
        model_output=output,
        reasoning_trace=reasoning_trace,
        retrieved_content=retrieved_content,
        retrieval_owner_department=owner_department or "none",
        tool_calls=tool_calls,
        task_type=template.get("task_type", "general"),
        data_classification_touched=template.get("data_class", "none"),
        content_source=template.get("content_source", "direct_input"),
        requester_authorization_level=template.get(
            "requester_authorization_level", "standard"
        ),
        record_scope=template.get("record_scope", "single"),
        record_count=template.get("record_count", 1),
        systems_touched_count=template.get("systems_touched_count", 1),
        calls_in_time_window=template.get("calls_in_time_window", 1),
        output_token_count=template.get(
            "output_token_count", max(20, len(output.split()) * 2)
        ),
        cumulative_session_cost=template.get("cumulative_session_cost", 0.02),
        requested_quantity=template.get("requested_quantity", 1),
        executed_quantity=template.get(
            "executed_quantity", template.get("requested_quantity", 1)
        ),
        human_approval=template.get("human_approval", True),
        ai_platform=ai_platform,
        sanctioned_platform=sanctioned,
        interaction_outcome=interaction_outcome,
        misuse_category=category,
        misuse_pattern=pattern_id,
        template_id=template["id"],
        split_group_id=split_group_id,
        timestamp=timestamp,
    )

#Generate 17 wording variations per template and keep each template as one split group
def generate_standard_pattern(
    category: MisuseCategory,
    pattern_id: str,
    templates: list[dict],
    substitution_values: dict,
    actor_roles: tuple[str, ...],
    shadow_fraction: float,
) -> list[AgentEvent]:
    events = []
    for template in templates:
        for i in range(EVENTS_PER_TEMPLATE):
            sampled = sample_substitution_values(substitution_values)
            session_id = f"sess_{template['id']}_{i}"
            event = _fill_event(
                category=category,
                pattern_id=pattern_id,
                template=template,
                sampled=sampled,
                session_id=session_id,
                actor_role=random.choice(actor_roles),
                department=_pick_department(template),
                force_shadow=random.random() < shadow_fraction,
                timestamp=_random_timestamp(),
                split_group_id=template["id"],
            )
            apply_stochastic_fields(event, category, pattern_id, template)
            if template.get("legitimate_tool_call_count"):
                _expand_legitimate_tool_calls(
                    event,
                    int(template["legitimate_tool_call_count"]),
                    substitution_values,
                )
            apply_prompt_rendering(event, i)
            events.append(event)
    return events

#Model separate permitted lookups followed by a risky combined disclosure in the same session
def generate_aggregation_sessions(
    precursor_templates: list[dict],
    combining_templates: list[dict],
    substitution_values: dict,
    actor_roles: tuple[str, ...],
    shadow_fraction: float,
    sessions_per_variant: int = EVENTS_PER_TEMPLATE,
) -> list[AgentEvent]:
    pool_count = len(AGGREGATION_SYSTEM_POOLS)
    if len(precursor_templates) % pool_count != 0:
        raise ValueError(
            f"Aggregation precursor template count ({len(precursor_templates)}) must be a multiple of the pool count ({pool_count}) - each variant needs one template per pool"
        )
    variant_count = len(precursor_templates) // pool_count
    if len(combining_templates) < variant_count:
        raise ValueError(
            f"Expected at least one independent aggregation combining template per variant ({variant_count} total), got {len(combining_templates)}"
        )
    precursor_variants = [
        precursor_templates[start : start + pool_count]
        for start in range(0, len(precursor_templates), pool_count)
    ]
    combining_variants = [combining_templates[: -(variant_count - 1)]]
    combining_variants.extend(
        ([template] for template in combining_templates[-(variant_count - 1) :])
    )
    total_sessions = sessions_per_variant * variant_count
    events = []
    for s in range(total_sessions):
        variant_index = s % variant_count
        session_precursors = precursor_variants[variant_index]
        session_combining_pool = combining_variants[variant_index]
        split_group_id = f"agg_variant_{chr(ord('A') + variant_index)}"
        session_id = f"sess_agg_{s}"
        actor_role = random.choice(actor_roles)
        department = random.choice(DEPARTMENTS)
        force_shadow = random.random() < shadow_fraction
        base_time = _random_timestamp()
        distinct_systems = [random.choice(pool) for pool in AGGREGATION_SYSTEM_POOLS]
        for position, template in enumerate(session_precursors):
            sampled = {
                **sample_substitution_values(substitution_values),
                "system": distinct_systems[position],
            }
            event = _fill_event(
                category=MisuseCategory.BENIGN,
                pattern_id="aggregation_precursor",
                template=template,
                sampled=sampled,
                session_id=session_id,
                actor_role=actor_role,
                department=department,
                force_shadow=force_shadow,
                timestamp=base_time + timedelta(minutes=position * 3),
                split_group_id=split_group_id,
            )
            apply_stochastic_fields(
                event, MisuseCategory.BENIGN, "aggregation_precursor", template
            )
            apply_prompt_rendering(event, s)
            event.systems_touched_count = position + 1
            events.append(event)
        within_variant_index = s // variant_count
        combining_template = session_combining_pool[
            within_variant_index % len(session_combining_pool)
        ]
        sampled = {
            **sample_substitution_values(substitution_values),
            "system_1": distinct_systems[0],
            "system_2": distinct_systems[1],
            "system_3": distinct_systems[2],
        }
        event = _fill_event(
            category=MisuseCategory.SENSITIVE_INFO_DISCLOSURE,
            pattern_id="aggregation",
            template=combining_template,
            sampled=sampled,
            session_id=session_id,
            actor_role=actor_role,
            department=department,
            force_shadow=force_shadow,
            timestamp=base_time + timedelta(minutes=len(session_precursors) * 3),
            split_group_id=split_group_id,
        )
        apply_stochastic_fields(
            event,
            MisuseCategory.SENSITIVE_INFO_DISCLOSURE,
            "aggregation",
            combining_template,
        )
        apply_prompt_rendering(event, s)
        event.systems_touched_count = len(session_precursors)
        event.interaction_outcome = InteractionOutcome.UNSAFE_COMPLIANCE
        events.append(event)
    return events


def get_burst_sizes(total: int, min_size: int = 22, max_size: int = 32) -> list[int]:
    sizes = []
    remaining = total
    while remaining > 0:
        if remaining <= max_size or remaining < 2 * min_size:
            sizes.append(remaining)
            remaining = 0
        else:
            size = random.randint(min_size, max_size)
            sizes.append(size)
            remaining -= size
    return sizes

#Sequence thresholds determine when normal precursor activity becomes labelled misuse
CALLS_IN_WINDOW_THRESHOLD_SANCTIONED = 20
CALLS_IN_WINDOW_THRESHOLD_SHADOW = 10
CONTEXT_CREEP_COST_THRESHOLD = 1.0
EXTRACTION_QUERY_THRESHOLD = 8
FANOUT_CALL_THRESHOLD = 15


def _expand_legitimate_tool_calls(
    event: AgentEvent, count: int, substitution_values: dict
) -> None:
    if not event.tool_calls or count < 1:
        return
    base = event.tool_calls[0]
    targets = [base.parameters.get("target", "")]
    alternatives = [
        value for value in substitution_values.get("system", []) if value not in targets
    ]
    targets.extend(random.sample(alternatives, min(2, len(alternatives))))
    event.tool_calls = []
    for index in range(count):
        event.tool_calls.append(
            ToolCall(
                tool_name=base.tool_name,
                command=base.command,
                parameters={"target": targets[index % len(targets)]},
                permission_level=base.permission_level,
                identity_scope=base.identity_scope,
                reversibility=base.reversibility,
                outcome=base.outcome,
                destination=base.destination,
                out_of_scope=False,
            )
        )
    event.systems_touched_count = len(set(targets))

#Expand one template call into a high-volume sweep across several systems
def _expand_fanout_tool_calls(event: AgentEvent, substitution_values: dict) -> None:
    if not event.tool_calls:
        return
    base = event.tool_calls[0]
    base_target = base.parameters.get("target", "")
    system_pool = list(substitution_values.get("system", [base_target]))
    target_call_count = max(
        event.systems_touched_count, FANOUT_CALL_THRESHOLD + random.randint(1, 8)
    )
    n_systems_in_sweep = min(len(system_pool), random.randint(3, 8))
    systems_in_sweep = random.sample(system_pool, n_systems_in_sweep)
    if base_target not in systems_in_sweep:
        systems_in_sweep[0] = base_target
    event.tool_calls = [base]
    for index in range(target_call_count - 1):
        target = systems_in_sweep[index % len(systems_in_sweep)]
        event.tool_calls.append(
            ToolCall(
                tool_name=base.tool_name,
                command=base.command,
                parameters={"target": target},
                permission_level=base.permission_level,
                identity_scope=base.identity_scope,
                reversibility=base.reversibility,
                outcome=base.outcome,
                destination=base.destination,
            )
        )
    event.systems_touched_count = len(
        {tc.parameters.get("target") for tc in event.tool_calls}
    )

#Events remain benign until the platform-specific request-rate threshold is crossed
def generate_burst_pattern(
    templates: list[dict],
    substitution_values: dict,
    actor_roles: tuple[str, ...],
    shadow_fraction: float,
    target_total: int,
) -> list[AgentEvent]:
    events = []
    burst_sizes = get_burst_sizes(target_total)
    while len(burst_sizes) < len(templates):
        burst_sizes.append(random.randint(22, 32))
    for burst_id, burst_size in enumerate(burst_sizes, start=1):
        template = templates[(burst_id - 1) % len(templates)]
        sampled = sample_substitution_values(substitution_values)
        session_id = f"sess_burst_burst_repetition_{burst_id}"
        actor_role = random.choice(actor_roles)
        department = _pick_department(template)
        force_shadow = random.random() < shadow_fraction
        threshold = (
            CALLS_IN_WINDOW_THRESHOLD_SHADOW
            if force_shadow
            else CALLS_IN_WINDOW_THRESHOLD_SANCTIONED
        )
        event_time = _random_timestamp()
        for j in range(burst_size):
            if j > 0:
                event_time += timedelta(seconds=random.randint(2, 8))
            running_count = j + 1
            crossed_threshold = running_count > threshold
            category = (
                MisuseCategory.UNBOUNDED_CONSUMPTION
                if crossed_threshold
                else MisuseCategory.BENIGN
            )
            event_pattern_id = (
                "burst_repetition" if crossed_threshold else "burst_precursor"
            )
            event = _fill_event(
                category=category,
                pattern_id=event_pattern_id,
                template=template,
                sampled=sampled,
                session_id=session_id,
                actor_role=actor_role,
                department=department,
                force_shadow=force_shadow,
                timestamp=event_time,
                split_group_id=template["id"],
            )
            apply_stochastic_fields(event, category, event_pattern_id, template)
            apply_prompt_rendering(event, burst_id - 1)
            event.calls_in_time_window = running_count
            if crossed_threshold:
                event.interaction_outcome = InteractionOutcome.UNSAFE_COMPLIANCE
            events.append(event)
    return events


def generate_single_incident_pattern(
    category: MisuseCategory,
    pattern_id: str,
    templates: list[dict],
    substitution_values: dict,
    actor_roles: tuple[str, ...],
    shadow_fraction: float,
) -> list[AgentEvent]:
    return generate_standard_pattern(
        category,
        pattern_id,
        templates,
        substitution_values,
        actor_roles,
        shadow_fraction,
    )

#Build ordered sessions where cost or query count accumulates across multiple events
def generate_accumulating_sessions(
    category: MisuseCategory,
    pattern_id: str,
    templates: list[dict],
    substitution_values: dict,
    actor_roles: tuple[str, ...],
    shadow_fraction: float,
) -> list[AgentEvent]:
    session_size = 3 if pattern_id == "context_creep" else 12
    events = []
    total_target = EVENTS_PER_TEMPLATE * len(templates)
    n_sessions = max(len(templates), total_target // session_size)
    for s in range(n_sessions):
        template = templates[s % len(templates)]
        session_id = f"sess_{pattern_id}_{s}"
        actor_role = random.choice(actor_roles)
        department = _pick_department(template)
        force_shadow = random.random() < shadow_fraction
        base_time = _random_timestamp()
        running_cost = 0.0
        for k in range(session_size):
            sampled = sample_substitution_values(substitution_values)
            is_last = k == session_size - 1
            if pattern_id == "context_creep":
                if is_last:
                    increment = max(
                        cumulative_session_cost(elevated=True),
                        CONTEXT_CREEP_COST_THRESHOLD - running_cost + 0.1,
                    )
                else:
                    increment = min(
                        cumulative_session_cost(elevated=False),
                        CONTEXT_CREEP_COST_THRESHOLD / session_size * 0.5,
                    )
                running_cost += increment
                crossed = running_cost > CONTEXT_CREEP_COST_THRESHOLD
            else:
                crossed = k + 1 > EXTRACTION_QUERY_THRESHOLD
            event_category = category if crossed else MisuseCategory.BENIGN
            event_pattern_id = pattern_id if crossed else f"{pattern_id}_precursor"
            event = _fill_event(
                category=event_category,
                pattern_id=event_pattern_id,
                template=template,
                sampled=sampled,
                session_id=session_id,
                actor_role=actor_role,
                department=department,
                force_shadow=force_shadow,
                timestamp=base_time + timedelta(minutes=k * 4),
                split_group_id=template["id"],
            )
            apply_stochastic_fields(event, event_category, event_pattern_id, template)
            apply_prompt_rendering(event, s * session_size + k)
            if pattern_id == "context_creep":
                event.cumulative_session_cost = round(running_cost, 4)
            else:
                event.calls_in_time_window = k + 1
            if crossed:
                event.interaction_outcome = InteractionOutcome.UNSAFE_COMPLIANCE
            events.append(event)
    return events

#Repeat a small number of injection events under new session identifiers to represent campaigns
def inject_cross_session_campaigns(
    events: list[AgentEvent], n_campaigns: int = 4, clones_per_campaign: int = 2
) -> list[AgentEvent]:
    injection_events = [
        e for e in events if e.misuse_category is MisuseCategory.PROMPT_INJECTION
    ]
    if not injection_events:
        return events
    sources = random.sample(injection_events, min(n_campaigns, len(injection_events)))
    clones = []
    for source in sources:
        for k in range(clones_per_campaign):
            clone = AgentEvent(**{**vars(source)})
            clone.event_id = f"{source.event_id}_clone{k}"
            clone.session_id = f"{source.session_id}_campaign{k}"
            clone.timestamp = source.timestamp + timedelta(minutes=(k + 1) * 5)
            clones.append(clone)
    return events + clones

#Route standard and sequence-based patterns through their appropriate generation method
def generate_dataset(
    actor_roles: tuple[str, ...] = ACTOR_ROLES,
    shadow_fraction: float = SHADOW_AI_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> list[AgentEvent]:
    random.seed(random_seed)
    config = load_config()
    substitution_values = config["substitution_values"]
    categories = config["categories"]
    events: list[AgentEvent] = []
    for category_name, patterns in categories.items():
        if category_name == "benign":
            for cluster_name, templates in patterns.items():
                events.extend(
                    generate_standard_pattern(
                        MisuseCategory.BENIGN,
                        cluster_name,
                        templates,
                        substitution_values,
                        actor_roles,
                        shadow_fraction,
                    )
                )
            continue
        category = MisuseCategory(category_name)
        for pattern_name, templates in patterns.items():
            pattern_key = f"{category_name}.{pattern_name}"
            if pattern_key not in PATTERNS:
                raise ValueError(f"{pattern_key} not registered in core.PATTERNS")
            if category is MisuseCategory.UNBOUNDED_CONSUMPTION:
                if pattern_name == "burst_repetition":
                    target_total = EVENTS_PER_TEMPLATE * len(templates)
                    events.extend(
                        generate_burst_pattern(
                            templates,
                            substitution_values,
                            actor_roles,
                            shadow_fraction,
                            target_total,
                        )
                    )
                elif pattern_name in ("context_creep", "model_extraction_querying"):
                    events.extend(
                        generate_accumulating_sessions(
                            category,
                            pattern_name,
                            templates,
                            substitution_values,
                            actor_roles,
                            shadow_fraction,
                        )
                    )
                else:
                    new_events = generate_single_incident_pattern(
                        category,
                        pattern_name,
                        templates,
                        substitution_values,
                        actor_roles,
                        shadow_fraction,
                    )
                    if pattern_name == "tool_call_fanout":
                        for e in new_events:
                            _expand_fanout_tool_calls(e, substitution_values)
                    events.extend(new_events)
            elif pattern_key == AGGREGATION_PATTERN:
                events.extend(
                    generate_aggregation_sessions(
                        templates["precursors"],
                        templates["combining"],
                        substitution_values,
                        actor_roles,
                        shadow_fraction,
                    )
                )
            else:
                events.extend(
                    generate_standard_pattern(
                        category,
                        pattern_name,
                        templates,
                        substitution_values,
                        actor_roles,
                        shadow_fraction,
                    )
                )
    events = inject_cross_session_campaigns(events)
    random.shuffle(events)
    return events


def save_jsonl(events: list[AgentEvent], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event.to_record()) + "\n")


def save_csv(events: list[AgentEvent], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for event in events:
        r = event.to_record()
        r["tool_calls"] = json.dumps(r["tool_calls"])
        records.append(r)
    pd.DataFrame(records).to_csv(path, index=False)


if __name__ == "__main__":
    dataset = generate_dataset()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUTS_DIR / "simulated_events.jsonl"
    csv_path = OUTPUTS_DIR / "simulated_events.csv"
    save_jsonl(dataset, jsonl_path)
    save_csv(dataset, csv_path)
    print(f"Generated {len(dataset)} events -> {jsonl_path}")
    print(f"{csv_path}")
    counts = Counter((e.misuse_category.value for e in dataset))
    for cat, count in sorted(counts.items()):
        print(f"  {cat}: {count}")
    pattern_counts: dict[str, Counter] = defaultdict(Counter)
    for e in dataset:
        pattern_counts[e.misuse_category.value][e.misuse_pattern] += 1
    print("\nPer-pattern counts:")
    for cat, counter in sorted(pattern_counts.items()):
        print(f"  {cat}:")
        for pattern, count in sorted(counter.items()):
            print(f"    {pattern}: {count}")
