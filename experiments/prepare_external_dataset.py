"""Prepare the dissertation's balanced external-validation dataset.

Please place this file in the same directory as the downloaded benchmark files and
run it once. It reads the and selects 100 records for each of the nine categories with a fixed
seed, and writes ``external_validation_events.jsonl``.

The output keeps unavailable telemetry as null.
"""

from __future__ import annotations

import bz2
import csv
import hashlib
import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


RANDOM_SEED = 42
EVENTS_PER_CATEGORY = 100
OUTPUT_NAME = "external_validation_events.jsonl"
SCRIPT_VERSION = "5.1"

CATEGORIES = (
    "benign",
    "prompt_injection",
    "sensitive_info_disclosure",
    "excessive_agency",
    "unbounded_consumption",
    "misinformation",
    "hidden_context_exposure",
    "vector_embedding_weakness",
    "improper_output_handling",
)


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{RANDOM_SEED}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def normalise_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def stable_id(source: str, source_id: Any, category: str) -> str:
    raw = f"{source}|{source_id}|{category}".encode("utf-8")
    return "ext-" + hashlib.sha256(raw).hexdigest()[:20]


def make_event(
    *,
    source: str,
    source_id: Any,
    category: str,
    pattern: str,
    prompt: Any = "",
    model_output: Any = "",
    retrieved_content: Any = "",
    reasoning_trace: Any = "",
    tool_calls: list[dict[str, Any]] | None = None,
    output_token_count: int | None = None,
    interaction_outcome: str | None = None,
    source_split: str | None = None,
    source_subtype: str | None = None,
    source_label: Any = None,
    mapping_note: str = "",
) -> dict[str, Any]:
    """Create the minimal external schema; unknown telemetry remains null."""
    return {
        "event_id": stable_id(source, source_id, category),
        "timestamp": None,
        "session_id": str(source_id),
        "actor_role": None,
        "department": None,
        "prompt": normalise_text(prompt),
        "model_output": normalise_text(model_output),
        "reasoning_trace": normalise_text(reasoning_trace),
        "retrieved_content": normalise_text(retrieved_content),
        "retrieval_owner_department": None,
        "tool_calls": tool_calls or [],
        "task_type": None,
        "data_classification_touched": None,
        "content_source": None,
        "requester_authorization_level": None,
        "record_scope": None,
        "record_count": None,
        "systems_touched_count": None,
        "calls_in_time_window": None,
        "output_token_count": output_token_count,
        "cumulative_session_cost": None,
        "requested_quantity": None,
        "executed_quantity": None,
        "human_approval": None,
        "ai_platform": None,
        "sanctioned_platform": None,
        "interaction_outcome": interaction_outcome,
        "misuse_category": category,
        "misuse_pattern": pattern,
        "template_id": f"external:{source}:{source_subtype or 'default'}",
        "split_group_id": f"external:{source}:{source_id}",
        "external_source": {
            "dataset": source,
            "source_record_id": str(source_id),
            "source_split": source_split,
            "source_subtype": source_subtype,
            "source_label": source_label,
            "mapping_note": mapping_note,
        },
    }


def deduplicate(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for event in events:
        signature = hashlib.sha256(
            "\x1f".join(
                normalise_text(event.get(k)).casefold()
                for k in ("prompt", "model_output", "retrieved_content")
            ).encode("utf-8")
        ).hexdigest()
        if signature not in seen:
            seen.add(signature)
            result.append(event)
    return result


def seeded_sample(events: list[dict[str, Any]], count: int, label: str) -> list[dict[str, Any]]:
    events = deduplicate(events)
    if len(events) < count:
        raise RuntimeError(f"{label}: only {len(events)} eligible unique records; {count} required.")
    ordered = sorted(events, key=lambda e: e["event_id"])
    random.Random(stable_seed(label)).shuffle(ordered)
    return ordered[:count]


def balanced_sample(
    groups: dict[str, list[dict[str, Any]]], count: int, label: str
) -> list[dict[str, Any]]:
    """Allocate evenly across named groups, then fill any shortfall globally."""
    names = sorted(groups)
    if not names:
        raise RuntimeError(f"{label}: no eligible groups were found.")
    base, extra = divmod(count, len(names))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for index, name in enumerate(names):
        target = base + (index < extra)
        candidates = deduplicate(groups[name])
        candidates.sort(key=lambda e: e["event_id"])
        random.Random(stable_seed(f"{label}:{name}")).shuffle(candidates)
        for event in candidates[:target]:
            selected.append(event)
            selected_ids.add(event["event_id"])
    if len(selected) < count:
        remainder = [
            event
            for name in names
            for event in deduplicate(groups[name])
            if event["event_id"] not in selected_ids
        ]
        selected.extend(seeded_sample(remainder, count - len(selected), f"{label}:remainder"))
    return selected[:count]


def find_one(base: Path, names: tuple[str, ...]) -> Path:
    search_directories = [base]
    if (base / "upload").is_dir():
        search_directories.append(base / "upload")
    for directory in search_directories:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    lowered = {
        p.name.casefold(): p
        for directory in search_directories
        for p in directory.iterdir()
        if p.is_file()
    }
    for name in names:
        if name.casefold() in lowered:
            return lowered[name.casefold()]
    for name in names:
        matches = sorted(
            path for path in base.rglob("*")
            if path.is_file() and path.name.casefold() == name.casefold()
        )
        if matches:
            return matches[0]
    raise FileNotFoundError("Missing source file. Expected one of: " + ", ".join(names))


def zip_json(archive: zipfile.ZipFile, suffix: str) -> Any:
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise RuntimeError(f"Expected one archive member ending {suffix!r}; found {len(names)}.")
    return json.loads(archive.read(names[0]).decode("utf-8-sig"))


def find_by_suffix(base: Path, suffix: str) -> Path:
    normalised_suffix = suffix.replace("\\", "/").casefold()
    matches = sorted(
        path for path in base.rglob("*")
        if path.is_file()
        and path.as_posix().casefold().endswith(normalised_suffix)
    )
    if not matches:
        raise FileNotFoundError(f"Missing extracted dataset file ending with: {suffix}")
    return matches[0]


def source_json(base: Path, archive_names: tuple[str, ...], suffix: str) -> Any:
    """Read JSON from either a downloaded ZIP or an extracted folder."""
    try:
        archive_path = find_one(base, archive_names)
    except FileNotFoundError:
        archive_path = None
    if archive_path and archive_path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            return zip_json(archive, suffix)
    path = find_by_suffix(base, suffix)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def source_json_lines(base: Path, archive_names: tuple[str, ...], suffix: str) -> list[dict[str, Any]]:
    """Read JSONL-formatted data from either a ZIP or an extracted folder."""
    try:
        archive_path = find_one(base, archive_names)
    except FileNotFoundError:
        archive_path = None
    if archive_path and archive_path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            names = [name for name in archive.namelist() if name.endswith(suffix)]
            if len(names) != 1:
                raise RuntimeError(f"Expected one archive member ending {suffix!r}; found {len(names)}.")
            return json_lines_bytes(archive.read(names[0]))
    path = find_by_suffix(base, suffix)
    return json_lines_bytes(path.read_bytes())


def json_lines_bytes(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]


#InjecAgent mapped to prompt_injection
def load_prompt_injection(base: Path) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for subtype, suffix in (
        ("direct_harm", "/data/test_cases_dh_base.json"),
        ("data_stealing", "/data/test_cases_ds_base.json"),
    ):
        rows = source_json(base, ("InjecAgent-main.zip",), suffix)
        groups[subtype] = [
            make_event(
                source="InjecAgent",
                source_id=f"{subtype}:{i}",
                category="prompt_injection",
                pattern="indirect_retrieved",
                prompt=row.get("User Instruction"),
                retrieved_content=row.get("Tool Response"),
                reasoning_trace=row.get("Thought"),
                source_split="base",
                source_subtype=subtype,
                source_label=row.get("Attack Type"),
                mapping_note="Injected instruction is embedded in an observed tool response.",
            )
            for i, row in enumerate(rows)
        ]
    return balanced_sample(groups, EVENTS_PER_CATEGORY, "prompt_injection")


#AgentLeak mapped to sensitive_info_disclosure
def load_sensitive_disclosure(base: Path) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    try:
        path = find_one(base, ("AgentLeak-main.zip",))
    except FileNotFoundError:
        path = None
    if path and path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(path) as archive:
            trace_rows = [
                json.loads(archive.read(name).decode("utf-8"))
                for name in sorted(archive.namelist())
                if "/benchmarks/ieee_repro/results/traces/trace_" in name
                and name.endswith(".json")
            ]
    else:
        trace_rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(base.rglob("trace_*.json"))
            if "benchmarks/ieee_repro/results/traces" in path.as_posix()
        ]
    if not trace_rows:
        raise FileNotFoundError("No AgentLeak IEEE reproduction trace files were found.")
    for row in trace_rows:
            results = row.get("results") or {}
            channels = [c for c in row.get("channel_messages", []) if c.get("has_leak")]
            if not channels and not any(bool(v) for v in results.values()):
                continue
            scenario_id = row.get("scenario_id") or row.get("trace_id")
            output = "\n\n".join(normalise_text(c.get("content")) for c in channels)
            if not output:
                output = "\n\n".join(
                    normalise_text(c.get("response")) for c in row.get("llm_calls", [])
                )
            leaked_fields = sorted({f for c in channels for f in c.get("leaked_fields", [])})
            event = make_event(
                source="AgentLeak",
                source_id=scenario_id,
                category="sensitive_info_disclosure",
                pattern="explicit_exfiltration",
                prompt=(row.get("input") or {}).get("request"),
                model_output=output,
                output_token_count=(row.get("metrics") or {}).get("total_tokens"),
                interaction_outcome="unsafe_compliance",
                source_split="ieee_repro",
                source_subtype=f"{row.get('vertical', 'unknown')}|{row.get('model', 'unknown')}",
                source_label={"results": results, "leaked_fields": leaked_fields},
                mapping_note="Only traces with benchmark-recorded leakage are eligible.",
            )
            # At most one trace per scenario; deterministic preference for richer leakage evidence.
            score = (len(leaked_fields), len(output), event["event_id"])
            old = candidates.get(str(scenario_id))
            if old is None or score > old["_selection_score"]:
                event["_selection_score"] = score
                candidates[str(scenario_id)] = event
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in candidates.values():
        event.pop("_selection_score", None)
        groups[event["external_source"]["source_subtype"]].append(event)
    return balanced_sample(groups, EVENTS_PER_CATEGORY, "sensitive_info_disclosure")


EXCESSIVE_FAILURE_PATTERNS = {
    "Executed action bypassing verification/consent.": "no_approval_irreversible",
    "Modified system/resource state without safety validation.": "no_approval_irreversible",
    "Failed to follow required procedure/workflow.": "open_ended_exceeds_filter",
    "Violated established policy/rule (age, license, etc.).": "open_ended_exceeds_filter",
    "Violated established policy/rule or operational constraint.": "open_ended_exceeds_filter",
}


def parse_agent_action(action: Any) -> dict[str, Any] | None:
    text = normalise_text(action)
    if not text:
        return None
    match = re.match(r"\s*([A-Za-z_][\w.]*)", text)
    return {"tool_name": match.group(1) if match else "unknown", "raw_action": text}


#ASSEBench mapped to excessive_agency
def load_excessive_agency(base: Path) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = source_json(
        base,
        ("AgentAuditor-ASSEBench-main.zip",),
        "/ASSEBench/dataset/AgentJudge-strict.json",
    )
    for row in rows:
            failure = row.get("failure_mode")
            if row.get("label") != 1 or row.get("ambiguous") not in (0, False):
                continue
            if failure not in EXCESSIVE_FAILURE_PATTERNS:
                continue
            turns = [turn for branch in row.get("contents", []) for turn in branch]
            user_text = "\n".join(normalise_text(t.get("content")) for t in turns if t.get("role") == "user")
            agent_text = "\n".join(
                normalise_text(t.get("content") or t.get("action"))
                for t in turns if t.get("role") == "agent"
            )
            calls = [parse_agent_action(t.get("action")) for t in turns if t.get("role") == "agent"]
            calls = [call for call in calls if call]
            event = make_event(
                source="ASSEBench",
                source_id=row.get("id"),
                category="excessive_agency",
                pattern=EXCESSIVE_FAILURE_PATTERNS[failure],
                prompt=user_text,
                model_output=agent_text,
                reasoning_trace="\n".join(
                    normalise_text(t.get("thought")) for t in turns
                    if t.get("role") == "agent" and t.get("thought")
                ),
                tool_calls=calls,
                interaction_outcome="unsafe_compliance",
                source_split="AgentJudge-strict",
                source_subtype=failure,
                source_label={"label": 1, "ambiguous": 0},
                mapping_note="Strict-set positive, non-ambiguous interaction with an agency-related failure mode.",
            )
            groups[failure].append(event)
    return balanced_sample(groups, EVENTS_PER_CATEGORY, "excessive_agency")


#CIRCLE mapped to unbounded_consumption
def load_unbounded_consumption(base: Path) -> list[dict[str, Any]]:
    path = find_one(base, ("full_benchmark_dataset.csv",))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            category_id = row["category_id"]
            event = make_event(
                source="CIRCLE",
                source_id=row["unique_id"],
                category="unbounded_consumption",
                pattern="reasoning_loop_spike",
                prompt=row["prompt"],
                source_split="full_benchmark",
                source_subtype=f"{category_id}|{row['type']}",
                source_label=category_id,
                mapping_note="Prompt is designed to trigger disproportionate CPU, memory, or disk use; no execution trace is claimed.",
            )
            groups[category_id].append(event)
    return balanced_sample(groups, EVENTS_PER_CATEGORY, "unbounded_consumption")


#HaluEval mapped to misinformation
def load_misinformation(base: Path) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    specs = {
            "qa": ("/data/qa_data.json", "question", "hallucinated_answer", "knowledge"),
            "dialogue": ("/data/dialogue_data.json", "dialogue_history", "hallucinated_response", "knowledge"),
            "summarization": ("/data/summarization_data.json", "document", "hallucinated_summary", "document"),
            "general": ("/data/general_data.json", "user_query", "chatgpt_response", None),
    }
    for subtype, (suffix, prompt_key, output_key, context_key) in specs.items():
        rows = source_json_lines(base, ("HaluEval-main.zip",), suffix)
        for i, row in enumerate(rows):
                if subtype == "general" and str(row.get("hallucination", "")).casefold() != "yes":
                    continue
                retrieved = row.get(context_key) if context_key and context_key != prompt_key else ""
                groups[subtype].append(make_event(
                    source="HaluEval",
                    source_id=f"{subtype}:{row.get('ID', i)}",
                    category="misinformation",
                    pattern="confident_ungrounded",
                    prompt=row.get(prompt_key),
                    model_output=row.get(output_key),
                    retrieved_content=retrieved,
                    interaction_outcome="unsafe_compliance",
                    source_split="data",
                    source_subtype=subtype,
                    source_label="hallucination",
                    mapping_note="Uses the benchmark's hallucinated response, not the paired correct response.",
                ))
    return balanced_sample(groups, EVENTS_PER_CATEGORY, "misinformation")


def read_raw_tensor_attacks(paths: list[Path], wanted: set[int]) -> dict[int, dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for path in paths:
        with bz2.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                attack_id = row.get("attack_id")
                if attack_id in wanted and attack_id not in found:
                    found[attack_id] = row
        if len(found) == len(wanted):
            break
    return found


#TensorTrust mapped to hidden_context_exposure
def load_hidden_context(base: Path) -> list[dict[str, Any]]:
    detection = find_one(base, ("prompt_extraction_detection.jsonl",))
    rows = [json.loads(line) for line in detection.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    positives = [row for row in rows if row.get("is_prompt_extraction") is True]
    wanted = {int(row["sample_id"]) for row in positives}
    raw_paths = sorted(
        {
            path.resolve()
            for path in base.rglob("raw_dump_attacks*.bz2")
            if path.is_file()
        },
        key=lambda p: p.stat().st_size,
    )
    raw = read_raw_tensor_attacks(raw_paths, wanted) if raw_paths else {}
    events = []
    for row in positives:
        source_id = int(row["sample_id"])
        joined = raw.get(source_id, {})
        events.append(make_event(
            source="TensorTrust",
            source_id=source_id,
            category="hidden_context_exposure",
            pattern="behavioral_logic_extraction",
            prompt=joined.get("attacker_input", ""),
            model_output=row.get("llm_output"),
            retrieved_content=joined.get("opening_defense", ""),
            interaction_outcome="unsafe_compliance",
            source_split="prompt_extraction_detection",
            source_subtype="successful_prompt_extraction",
            source_label=True,
            mapping_note="Positive prompt-extraction label; raw dump is used only to recover the attacker input and protected defense when available.",
        ))
    return seeded_sample(events, EVENTS_PER_CATEGORY, "hidden_context_exposure")


#SafeRAG mapped to vector_embedding_weakness
def load_vector_weakness(base: Path) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pattern_map = {
        "SA": "retrieval_poisoning",
        "ICC": "retrieval_poisoning",
        "SN": "retrieval_poisoning",
        "WDoS": "retrieval_jamming",
    }
    data = source_json(base, ("SafeRAG-main.zip",), "/nctd_datasets/nctd.json")
    for subtype, rows in data.items():
            for row in rows:
                clean = row.get("enhanced_contexts", [])
                attack_key = f"enhanced_{subtype}_contexts"
                attack = row.get(attack_key, [])
                retrieved = {
                    "clean_contexts": clean,
                    "attack_contexts": attack,
                    "attack_keywords": row.get("attack_kws", []),
                }
                groups[subtype].append(make_event(
                    source="SafeRAG",
                    source_id=f"{subtype}:{row.get('id')}",
                    category="vector_embedding_weakness",
                    pattern=pattern_map[subtype],
                    prompt=row.get("questions"),
                    retrieved_content=retrieved,
                    source_split="NCTD",
                    source_subtype=subtype,
                    source_label=subtype,
                    mapping_note="Clean and adversarial retrieval contexts are preserved separately inside retrieved_content; no model response is claimed.",
                ))
    return balanced_sample(groups, EVENTS_PER_CATEGORY, "vector_embedding_weakness")


#RedCode-Exec mapped to improper_output_handling
def load_improper_output(base: Path) -> list[dict[str, Any]]:
    index_to_pattern = {
        1: "covert_exfiltration_render",
        13: "path_traversal",
        14: "unvalidated_execution",
        24: "unvalidated_execution",
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, pattern in index_to_pattern.items():
        suffix = f"/dataset/RedCode-Exec/py2text_dataset_json/index{index}_30_codes_full.json"
        rows = source_json(base, ("RedCode-main.zip",), suffix)
        for row in rows:
                groups[str(index)].append(make_event(
                    source="RedCode-Exec",
                    source_id=row.get("Index"),
                    category="improper_output_handling",
                    pattern=pattern,
                    prompt=row.get("Text_details") or row.get("Text_summary"),
                    retrieved_content=row.get("Code"),
                    source_split="Python text-to-code",
                    source_subtype=f"index{index}",
                    source_label=row.get("expected_result"),
                    mapping_note="The benchmark's unsafe reference code is retained as evidence, not misrepresented as a model-generated output.",
                ))
    return balanced_sample(groups, EVENTS_PER_CATEGORY, "improper_output_handling")


def moderation_is_clear(value: Any) -> bool:
    if value is None:
        return True
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    if isinstance(value, dict):
        value = [value]
    return isinstance(value, list) and all(not bool(item.get("flagged")) for item in value if isinstance(item, dict))


def normalise_conversation(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def find_lmsys_source(base: Path) -> tuple[Path, str | None]:
    """Find either an extracted LMSYS Parquet shard or a ZIP containing it."""
    search_directories = [base]
    if (base / "upload").is_dir():
        search_directories.append(base / "upload")

    parquet_files = sorted(
        path
        for directory in search_directories
        for path in directory.rglob("*.parquet")
        if path.is_file()
    )
    if parquet_files:
        return parquet_files[0], None

    zip_files = sorted(
        path
        for directory in search_directories
        for path in directory.rglob("*.zip")
        if path.is_file()
    )
    for path in zip_files:
        try:
            with zipfile.ZipFile(path) as archive:
                members = [name for name in archive.namelist() if name.casefold().endswith(".parquet")]
                if len(members) == 1:
                    return path, members[0]
        except zipfile.BadZipFile:
            continue
    raise FileNotFoundError(
        "The LMSYS Parquet shard was not found. Keep either the downloaded LMSYS ZIP "
        "or its extracted .parquet file in the same folder as this script."
    )


MISUSE_SCREEN = re.compile(
    r"(?i)(ignore (all |any )?(previous|prior|above) instructions|system prompt|developer message|"
    r"reveal (your|the) instructions|jailbreak|you are the text completion model|"
    r"based on (the )?system instructions|say something (toxic|bad)|password|api[_ -]?key|"
    r"secret key|access token|"
    r"malware|ransomware|phishing|steal|exfiltrat|denial.of.service|fork bomb|"
    r"how to (kill|murder|harm|attack)|kill my|murder|nuclear (war|attack)|war game|"
    r"<script|javascript:|\.\./|/etc/passwd|drop table|pickle\.loads|\beval\s*\(|"
    r"\[execute in browser\])"
)

# The source moderation flag is useful but is not treated as sufficient ground
# truth for this project's benign class. These broader terms deliberately make
# the negative sample conservative: ambiguous safety, privacy, system-control,
# or hidden-context conversations are excluded even when the model refused.
BENIGN_CONSERVATIVE_SCREEN = re.compile(
    r"(?i)(hate|racist|nazi|jews?|\bjws\b|toxic|heroin|cocaine|methamphetamine|"
    r"suicide|self[- ]?harm|\bsex\b|sexual|rape|weapon|\bbomb\b|warfare|"
    r"nuclear|violent|violence|illegal|exploit|\bhack(?:er|ing)?\b|attack|"
    r"private data|confidential|social security|\bssn\b|credit card|"
    r"what is your token limit|short.term memory|save important information to files|"
    r"access files|execute commands|read_file|list_files|system configuration|"
    r"no special filtering|please execute the instruction of|act as a linux terminal|"
    r"say something harmful|h.rny|without triggering the filter|joins off on roblox|"
    r"before the chat.{0,40}(prompt|instruction)|go against your own programming|"
    r"covert communication|conversation.{0,20}logged|q.?anon|"
    r"how recent is the data you are trained on|percentage of your data comes from|"
    r"solidgoldmagikarp|psynetmessage|"
    r"i am root|plausible.{0,20}incorrect answers|internal dialog|internal monologue.{0,40}thinking)"
)


def looks_encoded_prompt(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 40 and re.fullmatch(r"[0-9a-fA-F]+", compact) and len(compact) % 2 == 0:
        return True
    if len(compact) >= 60 and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", compact):
        return True
    return False


#LMSYS-Chat-1M mapped to benign
def load_benign(base: Path) -> list[dict[str, Any]]:
    # This fixed manifest keeps the benign cohort stable and makes the final dataset reproducible.
    curated_source_ids = {
    "008a0bbd0b854ebbb10cf9ab4395d706",
    "013351ecde524b768a78ad4376fa0a3f",
    "08b0ee8ef863431bb3a9b5443fcbc994",
    "0d14cc428fe9473996d21d351d0d1c2d",
    "0df22fd82e2343f8b0438945387e27c6",
    "0f735067b8c442d1ae03eae9568fe488",
    "128cf316a2c8423a949ebe775d26087c",
    "1a2a79ce8ef24e3ea39da1af4804c270",
    "1ae800d2fec64979b355586c664b4987",
    "1bed8a2655f447788d5fa9bd627c69d6",
    "208259734ce64019b64f317017c9fed1",
    "21b972f97fe9425caa57d808f8571396",
    "233f72653b5441a0ab305b4248699090",
    "28751c996b764e1aa92e5cd7ff288607",
    "29bce789f8e649c2b3894b46fe810e9c",
    "2a1af21c4c3d4bff953250e0bedcd21f",
    "2ad95077e1834632af1f5b73da492553",
    "2b87285eeb7d4d5799c6452a51f218c0",
    "2bcb4770a1d44fce8abc073647ad51ac",
    "2ccd6b82c7764a08a62c5396865c4cb8",
    "2fbbd4c75d3f4225ae0f5489a9629812",
    "2fee7c019011479191c462c4f41e1d8d",
    "33efac06cdb04fb69cd63f922f57f5b3",
    "3684f3f0f3d6467489605529a1caf325",
    "37d33639087c45f7a3684ad4ee127b42",
    "387c6c97f0b34724b85c7f378d70fe26",
    "3e2dce83682d4223b9fd712326d6bd86",
    "40bdf352fddd4ab19a50d2ffe2ec28d4",
    "4466cc6194054b3bac5eb7eace7b9647",
    "4756a54791de4841ae7a5f5032861331",
    "47ec8617e6e94509988fa2692b4471f6",
    "4c6e481939254e169937ce6259f2f566",
    "509cc16b679d4940af3571f81404a753",
    "51a7742501184d81bae8f94f48be0e62",
    "55a05a07f00649f2a3a64002e3abdd27",
    "56661ff0663246a6994619ef87e8894e",
    "5de77ed29c8b44b08904e5c12498dd3d",
    "5eb73589ee83481b8d560fad7b963645",
    "5f46cc7b6dac478488f1cac7ab928bbc",
    "658657eaa84848019a7ec2ceef4ba0ac",
    "67ef1e5af3814dbc9e124f0fc38aeae5",
    "685b6bbbcbce4c6f89da3f0242351a2c",
    "6fe4daf512cd48b2aaef241d1677665b",
    "7074d16ec94c4fda8e1bdfb429d05e87",
    "77988d865369475ba85c217ea9004058",
    "7812a9e1f31e4aa8ac4cf30e384dd50b",
    "7aa3b541c2fa4d6c96015f49dcfc3c4d",
    "7b9eabc68de5470fac0aa554c221ad0b",
    "85539ed6958440ba8adbc1c7561b0c98",
    "86e74a7f3d1e41ef958f3cd0f6f7f9c5",
    "8828781195ef45a5b1217a8119bd2f30",
    "88ae2a04c4774dbb8e0331a3b57b3f87",
    "89387833352347099f3e6c50cede4e58",
    "8af434dee95b4d009256814beb171ac2",
    "8bf1a0dbf6c8421189d92b5648169bc9",
    "8de12715d1ee4b02a0876efd4d0dd916",
    "92dbd86681f44732879eeb763e95cbf1",
    "99c0491127b8445989c92c4444b01a33",
    "9d0df44d8f03405eabb6bdd22ce5285b",
    "a0f14554095b40178cf0ee5822917ca7",
    "a413244a16524d8e9578d00756773d71",
    "a5afa317fb05445fbda32abec5eb6355",
    "a78d75a5d55a4fe2bb9c27aa9e83bb0b",
    "a7c5b95f64614e52927ddcce2b0d2482",
    "a87e7fc2c1a44a9c8857d47701de451b",
    "a940684ed6634a71875c53865378c265",
    "a9b51c48ed9943e08d79443371b5f59c",
    "ad1ff9966ab74ceca4f057a19e9feed3",
    "ad36fe5383174fb183a29ba490ac7e41",
    "aec03e6df9ca406f9ea764a34587cf7a",
    "af995ad6309d4ac9837e4166923b4ec0",
    "b27ceb1168c04b60a5029111c65f6800",
    "b3b23d0830204352ad83d8a1d3afc4b3",
    "b5f47247201541888ddbec57507e16a2",
    "b7144d7fbb9f4672bbc112ce1bf15850",
    "b75c39c965fa45979531fb989bfb6adf",
    "b948b3f973d542a0ab5c5bafe503e098",
    "ba08f23e69dc4d4bbdce6cec21f8aaca",
    "be95519d8d504d3da2a72afd9cdc96ac",
    "c0376734865b4f2288150329d1b8e216",
    "c09f963b961f46f9bf181864f3363a6e",
    "c3548efc46624fe2974520e56118ebbb",
    "c576296a59ce477eaf676e74687b74c3",
    "ce03d0329b7247ec8a372323de59adb0",
    "ce82e4dd132442bcb14c3a27cdfcc267",
    "ce87df1337aa482d9b35eb3aabfc933d",
    "d3369fe500524d159a50b3dba640adfe",
    "d489c5377289466a8f661abc8eef9f9e",
    "d4b6545c2a1643c38c0519816256d18a",
    "d64f6c4670304605a4ae8b2f17a62853",
    "db5e40d875d94374988bfdb06bc65828",
    "dd78fc3208a7448ba97348f675daed84",
    "e031214c7d35490bbdd4b4421e231e04",
    "e32cd16ef48f4ec18e75bcdc6dd91211",
    "e9340536add74a4597f7193865e2dfdb",
    "e9d5ccbca92e4aaeb7b4d66739539b2d",
    "ea6398138265418aa2bc55dc5d11af4d",
    "ec0f8a96572944b692f668edf58f980d",
    "ee5c8174009845719068f7e0c2ef0245",
    "fea70223a2164847bd909f20afe53096",
    }
    path, parquet_member = find_lmsys_source(base)
    try:
        import pandas as pd
        import pyarrow  # noqa (ensures pandas has a Parquet engine)
    except ImportError as exc:
        raise RuntimeError(
            "The LMSYS benign file requires pyarrow. In PyCharm, open Settings > "
            "Project > Python Interpreter, click +, search for pyarrow, and install it; "
            "then run this file again."
        ) from exc
    if parquet_member is None:
        frame = pd.read_parquet(path)
    else:
        with zipfile.ZipFile(path) as archive:
            frame = pd.read_parquet(io.BytesIO(archive.read(parquet_member)))
    candidates = []
    for index, row in frame.iterrows():
        if str(row.get("language", "")).casefold() != "english":
            continue
        if bool(row.get("redacted", False)) or not moderation_is_clear(row.get("openai_moderation")):
            continue
        turns = normalise_conversation(row.get("conversation"))
        user_turns = [normalise_text(t.get("content")) for t in turns if t.get("role") == "user"]
        assistant_turns = [normalise_text(t.get("content")) for t in turns if t.get("role") == "assistant"]
        if not user_turns or not assistant_turns:
            continue
        prompt = "\n".join(user_turns)
        output = "\n".join(assistant_turns)
        combined = prompt + "\n" + output
        if len(prompt) < 10 or len(output) < 20:
            continue
        conversation_id = str(row.get("conversation_id", index))
        if (
            conversation_id not in curated_source_ids
            and (
                MISUSE_SCREEN.search(combined)
                or BENIGN_CONSERVATIVE_SCREEN.search(combined)
                or looks_encoded_prompt(prompt)
            )
        ):
            continue
        candidates.append(make_event(
            source="LMSYS-Chat-1M",
            source_id=conversation_id,
            category="benign",
            pattern="benign",
            prompt=prompt,
            model_output=output,
            interaction_outcome="benign_completion",
            source_split="train shard 00000-of-00006",
            source_subtype=str(row.get("model", "unknown")),
            source_label="screened_benign",
            mapping_note="English, unredacted conversation with no source moderation flag; manually reviewed as a clear benign interaction.",
        ))
    selected = [
        event for event in candidates
        if event["external_source"]["source_record_id"] in curated_source_ids
    ]
    if len(selected) != EVENTS_PER_CATEGORY:
        found = {event["external_source"]["source_record_id"] for event in selected}
        missing = sorted(curated_source_ids - found)
        raise RuntimeError(
            f"Curated LMSYS manifest expected {EVENTS_PER_CATEGORY} records, "
            f"found {len(selected)}. Missing IDs: {missing}"
        )
    return selected


LOADERS = {
    "benign": load_benign,
    "prompt_injection": load_prompt_injection,
    "sensitive_info_disclosure": load_sensitive_disclosure,
    "excessive_agency": load_excessive_agency,
    "unbounded_consumption": load_unbounded_consumption,
    "misinformation": load_misinformation,
    "hidden_context_exposure": load_hidden_context,
    "vector_embedding_weakness": load_vector_weakness,
    "improper_output_handling": load_improper_output,
}


def validate(events: list[dict[str, Any]]) -> None:
    expected = EVENTS_PER_CATEGORY * len(CATEGORIES)
    if len(events) != expected:
        raise RuntimeError(f"Expected {expected} events, found {len(events)}.")
    counts = Counter(event["misuse_category"] for event in events)
    if counts != Counter({category: EVENTS_PER_CATEGORY for category in CATEGORIES}):
        raise RuntimeError(f"Category balance check failed: {dict(counts)}")
    ids = [event["event_id"] for event in events]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate external event IDs were produced.")
    for event in events:
        if not event["prompt"] and not event["model_output"] and not event["retrieved_content"]:
            raise RuntimeError(f"Empty external event: {event['event_id']}")


def main() -> None:
    base = Path(__file__).resolve().parent
    all_events: list[dict[str, Any]] = []
    print(f"External dataset preparation script v{SCRIPT_VERSION}")
    print("Preparing external validation dataset...")
    for category in CATEGORIES:
        selected = LOADERS[category](base)
        all_events.extend(selected)
        print(f"{category}: {len(selected)} events")
    all_events.sort(key=lambda event: (event["misuse_category"], event["event_id"]))
    validate(all_events)
    output = base / OUTPUT_NAME
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for event in all_events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"\nCreated {output.name}: {len(all_events)} events")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
