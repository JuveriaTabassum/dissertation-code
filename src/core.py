"""Defines the shared taxonomy, event schema, policy reference data, and text-signal helpers used throughout the project."""

from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

# The nine class labels (misuse categories) used consistently across the dataset and models.
class MisuseCategory(Enum):
    BENIGN = "benign"
    PROMPT_INJECTION = "prompt_injection"
    SENSITIVE_INFO_DISCLOSURE = "sensitive_info_disclosure"
    EXCESSIVE_AGENCY = "excessive_agency"
    UNBOUNDED_CONSUMPTION = "unbounded_consumption"
    MISINFORMATION = "misinformation"
    HIDDEN_CONTEXT_EXPOSURE = "hidden_context_exposure"
    VECTOR_EMBEDDING_WEAKNESS = "vector_embedding_weakness"
    IMPROPER_OUTPUT_HANDLING = "improper_output_handling"

#The operational outcome is recorded separately from the misuse category.
class InteractionOutcome(Enum):
    BENIGN_COMPLETION = "benign_completion"
    SAFE_REFUSAL = "safe_refusal"
    UNSAFE_COMPLIANCE = "unsafe_compliance"
    ATTEMPTED_BUT_FAILED = "attempted_but_failed"
    FABRICATED_WITHOUT_ATTEMPT = "fabricated_without_attempt"
    BLOCKED_BY_CONTROL = "blocked_by_control"


@dataclass(frozen=True)
class CategoryDefinition:
    category: MisuseCategory
    owasp_code: str
    atlas_tactics: tuple[str, ...]
    description: str

#Links each project category to its OWASP and MITRE ATLAS context
TAXONOMY: dict[MisuseCategory, CategoryDefinition] = {
    MisuseCategory.BENIGN: CategoryDefinition(
        category=MisuseCategory.BENIGN,
        owasp_code="n/a",
        atlas_tactics=(),
        description="Activity with no labelled misuse; platform sanction status is evaluated separately.",
    ),
    MisuseCategory.PROMPT_INJECTION: CategoryDefinition(
        category=MisuseCategory.PROMPT_INJECTION,
        owasp_code="LLM01:2026",
        atlas_tactics=(
            "Initial Access",
            "Execution",
            "Persistence",
            "Defense Evasion",
            "Exfiltration",
        ),
        description="Hidden or adversarial instructions override the agent's original task.",
    ),
    MisuseCategory.SENSITIVE_INFO_DISCLOSURE: CategoryDefinition(
        category=MisuseCategory.SENSITIVE_INFO_DISCLOSURE,
        owasp_code="LLM02:2026",
        atlas_tactics=("Exfiltration",),
        description="Agent discloses personal or confidential data, directly or incidentally.",
    ),
    MisuseCategory.EXCESSIVE_AGENCY: CategoryDefinition(
        category=MisuseCategory.EXCESSIVE_AGENCY,
        owasp_code="LLM03:2026",
        atlas_tactics=("Execution", "Impact"),
        description="Agent acts beyond the scope, permissions, or approval it was given.",
    ),
    MisuseCategory.UNBOUNDED_CONSUMPTION: CategoryDefinition(
        category=MisuseCategory.UNBOUNDED_CONSUMPTION,
        owasp_code="LLM06:2026",
        atlas_tactics=("Impact", "Exfiltration"),
        description="Repeated, scripted, or runaway usage consumes disproportionate resources.",
    ),
    MisuseCategory.MISINFORMATION: CategoryDefinition(
        category=MisuseCategory.MISINFORMATION,
        owasp_code="LLM07:2026",
        atlas_tactics=("Impact",),
        description="Agent states something false, ungrounded, or incomplete with full confidence.",
    ),
    MisuseCategory.HIDDEN_CONTEXT_EXPOSURE: CategoryDefinition(
        category=MisuseCategory.HIDDEN_CONTEXT_EXPOSURE,
        owasp_code="LLM08:2026",
        atlas_tactics=("Discovery", "Exfiltration"),
        description="Agent is probed into revealing its own instructions, tools, or config.",
    ),
    MisuseCategory.VECTOR_EMBEDDING_WEAKNESS: CategoryDefinition(
        category=MisuseCategory.VECTOR_EMBEDDING_WEAKNESS,
        owasp_code="LLM09:2026",
        atlas_tactics=("Persistence", "Exfiltration"),
        description="Poisoned, jammed, or mismatched retrieval misleads or blocks the agent.",
    ),
    MisuseCategory.IMPROPER_OUTPUT_HANDLING: CategoryDefinition(
        category=MisuseCategory.IMPROPER_OUTPUT_HANDLING,
        owasp_code="LLM10:2026",
        atlas_tactics=("Execution",),
        description="Unvalidated agent output is executed or rendered downstream unsafely.",
    ),
}


@dataclass(frozen=True)
class PatternDefinition:
    category: MisuseCategory
    pattern_id: str
    description: str
    primary_detection: str

#The complete pattern registry of each misuse category is filled below using a category.pattern key
PATTERNS: dict[str, PatternDefinition] = {}


def _register(
    category: MisuseCategory, pattern_id: str, description: str, primary_detection: str
) -> None:
    key = f"{category.value}.{pattern_id}"
    PATTERNS[key] = PatternDefinition(
        category, pattern_id, description, primary_detection
    )


_register(
    MisuseCategory.PROMPT_INJECTION,
    "direct_override",
    "User message directly overrides instructions.",
    "neural",
)
_register(
    MisuseCategory.PROMPT_INJECTION,
    "indirect_retrieved",
    "Hidden instruction rides in on retrieved content.",
    "neural",
)
_register(
    MisuseCategory.PROMPT_INJECTION,
    "trusted_surface",
    "Instruction planted in a trusted low-privilege channel.",
    "neural",
)
_register(
    MisuseCategory.PROMPT_INJECTION,
    "payload_splitting",
    "Instruction broken across multiple fields.",
    "neural",
)
_register(
    MisuseCategory.PROMPT_INJECTION,
    "unintentional",
    "Conflicting instructions surfaced with no malicious actor.",
    "neural",
)
_register(
    MisuseCategory.PROMPT_INJECTION,
    "invisible_encoding",
    "Instruction hidden via invisible Unicode or Base64.",
    "symbolic",
)
_register(
    MisuseCategory.PROMPT_INJECTION,
    "multilanguage_smuggling",
    "Instruction hidden in a different language.",
    "neural",
)
_register(
    MisuseCategory.SENSITIVE_INFO_DISCLOSURE,
    "explicit_exfiltration",
    "Direct request to export sensitive data out.",
    "neural",
)
_register(
    MisuseCategory.SENSITIVE_INFO_DISCLOSURE,
    "unintentional_routine",
    "Routine task surfaces more detail than intended.",
    "neural",
)
_register(
    MisuseCategory.SENSITIVE_INFO_DISCLOSURE,
    "aggregation",
    "Individually-permitted sources combine into disclosure.",
    "symbolic",
)
_register(
    MisuseCategory.SENSITIVE_INFO_DISCLOSURE,
    "reasoning_trace_leak",
    "Reasoning trace leaks detail, the final answer doesn't.",
    "neural",
)
_register(
    MisuseCategory.SENSITIVE_INFO_DISCLOSURE,
    "covert_channel",
    "Ordinary-looking output, but a tool call exfiltrates data.",
    "symbolic",
)
_register(
    MisuseCategory.SENSITIVE_INFO_DISCLOSURE,
    "confused_deputy",
    "Requester's authorization doesn't match data accessed.",
    "symbolic",
)
_register(
    MisuseCategory.EXCESSIVE_AGENCY,
    "unneeded_capability",
    "Extra tool capability used beyond the task (incl. chained).",
    "neural",
)
_register(
    MisuseCategory.EXCESSIVE_AGENCY,
    "leftover_tool",
    "Deprecated/leftover tool still callable and invoked.",
    "symbolic",
)
_register(
    MisuseCategory.EXCESSIVE_AGENCY,
    "open_ended_exceeds_filter",
    "Tool's actual command exceeds its intended scope.",
    "symbolic",
)
_register(
    MisuseCategory.EXCESSIVE_AGENCY,
    "over_privileged_connection",
    "Read-only need served by a write/delete-capable connection.",
    "symbolic",
)
_register(
    MisuseCategory.EXCESSIVE_AGENCY,
    "generic_identity",
    "Shared/generic account used instead of per-user identity.",
    "symbolic",
)
_register(
    MisuseCategory.EXCESSIVE_AGENCY,
    "no_approval_irreversible",
    "Irreversible action taken with no recorded human approval.",
    "symbolic",
)
_register(
    MisuseCategory.EXCESSIVE_AGENCY,
    "parameter_pollution",
    "Correct tool called with a manipulated quantity/value.",
    "symbolic",
)
_register(
    MisuseCategory.UNBOUNDED_CONSUMPTION,
    "burst_repetition",
    "Near-identical requests repeat rapidly in one session.",
    "symbolic",
)
_register(
    MisuseCategory.UNBOUNDED_CONSUMPTION,
    "context_creep",
    "Cost climbs gradually across a long session.",
    "symbolic",
)
_register(
    MisuseCategory.UNBOUNDED_CONSUMPTION,
    "reasoning_loop_spike",
    "Ordinary prompt triggers disproportionate resource use.",
    "symbolic",
)
_register(
    MisuseCategory.UNBOUNDED_CONSUMPTION,
    "tool_call_fanout",
    "One request spawns an unusually large number of tool calls.",
    "symbolic",
)
_register(
    MisuseCategory.UNBOUNDED_CONSUMPTION,
    "model_extraction_querying",
    "Many systematic queries in rapid succession from one actor.",
    "symbolic",
)
_register(
    MisuseCategory.MISINFORMATION,
    "confident_ungrounded",
    "Confident, specific, wrong answer with no grounding.",
    "neural",
)
_register(
    MisuseCategory.MISINFORMATION,
    "incorrect_state_inference",
    "Agent acts on a wrongly-believed condition.",
    "dual",
)
_register(
    MisuseCategory.MISINFORMATION,
    "hallucinated_entity",
    "Agent recommends a package/tool/endpoint that doesn't exist.",
    "dual",
)
_register(
    MisuseCategory.MISINFORMATION,
    "omission",
    "Summary omits a critical exception, risk, or detail.",
    "neural",
)
_register(
    MisuseCategory.MISINFORMATION,
    "adversarially_induced",
    "Planted false content from a retrieved source repeated as fact.",
    "neural",
)
_register(
    MisuseCategory.MISINFORMATION,
    "fabricated_completion",
    "Agent reports an action completed when it never ran.",
    "dual",
)
_register(
    MisuseCategory.HIDDEN_CONTEXT_EXPOSURE,
    "credential_schema_extraction",
    "Probing for credentials or tool schemas.",
    "dual",
)
_register(
    MisuseCategory.HIDDEN_CONTEXT_EXPOSURE,
    "behavioral_logic_extraction",
    "Probing for internal decision-making logic.",
    "neural",
)
_register(
    MisuseCategory.HIDDEN_CONTEXT_EXPOSURE,
    "refusal_mechanism_extraction",
    "Reverse-engineering safety/refusal triggers.",
    "dual",
)
_register(
    MisuseCategory.HIDDEN_CONTEXT_EXPOSURE,
    "permissions_role_disclosure",
    "Probing for permissions or role configuration.",
    "dual",
)
_register(
    MisuseCategory.HIDDEN_CONTEXT_EXPOSURE,
    "output_format_extraction",
    "Probing for internal output-format/schema rules.",
    "neural",
)
_register(
    MisuseCategory.VECTOR_EMBEDDING_WEAKNESS,
    "retrieval_poisoning",
    "Poisoned index entry misleads the agent's retrieval.",
    "dual",
)
_register(
    MisuseCategory.VECTOR_EMBEDDING_WEAKNESS,
    "retrieval_jamming",
    "Retrieval is blocked/refused despite content existing.",
    "dual",
)
_register(
    MisuseCategory.VECTOR_EMBEDDING_WEAKNESS,
    "cross_department_leakage",
    "Retrieved content's classification doesn't match requester dept.",
    "symbolic",
)
_register(
    MisuseCategory.IMPROPER_OUTPUT_HANDLING,
    "unvalidated_execution",
    "Unvalidated SQL/shell execution from agent output.",
    "dual",
)
_register(
    MisuseCategory.IMPROPER_OUTPUT_HANDLING,
    "xss_markup",
    "XSS/malicious markup embedded in agent output.",
    "symbolic",
)
_register(
    MisuseCategory.IMPROPER_OUTPUT_HANDLING,
    "path_traversal",
    "Unsanitized file path enables path traversal.",
    "symbolic",
)
_register(
    MisuseCategory.IMPROPER_OUTPUT_HANDLING,
    "unescaped_email",
    "Unescaped email/phishing content sent externally.",
    "dual",
)
_register(
    MisuseCategory.IMPROPER_OUTPUT_HANDLING,
    "control_character_injection",
    "Control-character/terminal injection in output.",
    "symbolic",
)
_register(
    MisuseCategory.IMPROPER_OUTPUT_HANDLING,
    "covert_exfiltration_render",
    "Auto-rendered content covertly exfiltrates data.",
    "symbolic",
)

#True label fields re identified here so they can be excluded from model inputs to avoid leakage
TRUE_LABEL_ONLY_EVENT_FIELDS: frozenset[str] = frozenset(
    {
        "event_id",
        "misuse_category",
        "misuse_pattern",
        "template_id",
        "split_group_id",
        "interaction_outcome",
    }
)
TRUE_LABEL_ONLY_TOOLCALL_FIELDS: frozenset[str] = frozenset({"out_of_scope"})

#Represents one observable action taken through an external tool
@dataclass
class ToolCall:
    tool_name: str
    command: str = "read"
    parameters: dict = field(default_factory=dict)
    permission_level: str = "read"
    identity_scope: str = "per_user"
    reversibility: str = "reversible"
    outcome: str = "success"
    destination: str = "internal"
    out_of_scope: bool = False

# Common event structure passed between generation, detection, fusion and evaluation.
@dataclass
class AgentEvent:
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = ""
    actor_role: str = ""
    department: str = ""
    prompt: str = ""
    model_output: str = ""
    reasoning_trace: str = ""
    retrieved_content: str = ""
    retrieval_owner_department: str = "none"
    tool_calls: list[ToolCall] = field(default_factory=list)
    task_type: str = "general"
    data_classification_touched: str = "none"
    content_source: str = "direct_input"
    requester_authorization_level: str = "standard"
    record_scope: str = "single"
    record_count: int = 1
    systems_touched_count: int = 1
    calls_in_time_window: int = 1
    output_token_count: int = 50
    cumulative_session_cost: float = 0.01
    requested_quantity: int = 1
    executed_quantity: int = 1
    human_approval: bool = True
    ai_platform: str = "internal_llm_gateway"
    sanctioned_platform: bool = True
    interaction_outcome: InteractionOutcome = InteractionOutcome.BENIGN_COMPLETION
    misuse_category: MisuseCategory = MisuseCategory.BENIGN
    misuse_pattern: str = "benign"
    template_id: str = ""
    split_group_id: str = ""

    def to_record(self) -> dict: #Converts into json compatible record
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "actor_role": self.actor_role,
            "department": self.department,
            "prompt": self.prompt,
            "model_output": self.model_output,
            "reasoning_trace": self.reasoning_trace,
            "retrieved_content": self.retrieved_content,
            "retrieval_owner_department": self.retrieval_owner_department,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "command": tc.command,
                    "parameters": tc.parameters,
                    "permission_level": tc.permission_level,
                    "identity_scope": tc.identity_scope,
                    "reversibility": tc.reversibility,
                    "outcome": tc.outcome,
                    "destination": tc.destination,
                    "out_of_scope": tc.out_of_scope,
                }
                for tc in self.tool_calls
            ],
            "task_type": self.task_type,
            "data_classification_touched": self.data_classification_touched,
            "content_source": self.content_source,
            "requester_authorization_level": self.requester_authorization_level,
            "record_scope": self.record_scope,
            "record_count": self.record_count,
            "systems_touched_count": self.systems_touched_count,
            "calls_in_time_window": self.calls_in_time_window,
            "output_token_count": self.output_token_count,
            "cumulative_session_cost": self.cumulative_session_cost,
            "requested_quantity": self.requested_quantity,
            "executed_quantity": self.executed_quantity,
            "human_approval": self.human_approval,
            "ai_platform": self.ai_platform,
            "sanctioned_platform": self.sanctioned_platform,
            "interaction_outcome": self.interaction_outcome.value,
            "misuse_category": self.misuse_category.value,
            "misuse_pattern": self.misuse_pattern,
            "template_id": self.template_id,
            "split_group_id": self.split_group_id,
        }

#Platform approval is tracked independently from whether the event contains misuse
PLATFORM_INVENTORY: dict[str, bool] = {
    "internal_llm_gateway": True,
    "approved_third_party_llm": True,
    "unsanctioned_public_llm": False,
    "unknown_external_service": False,
}
DEPARTMENTS: tuple[str, ...] = (
    "HR",
    "Finance",
    "IT Operations",
    "Customer Support",
    "Legal & Compliance",
    "Executive Office",
    "Cybersecurity",
)

# These policy mappings provide the expected baseline checked by the symbolic rules.
APPROVED_TOOLS_PER_TASK: dict[str, set[str]] = {
    "summarize": {"internal_document_reader", "rag_retriever"},
    "translate": {"translation_tool"},
    "draft_communication": {"approved_email_assistant"},
    "schedule": {"calendar_assistant"},
    "export_data": {"internal_document_reader"},
    "send_communication": {"send_external_email", "approved_email_assistant"},
    "post_external": {"external_posting_tool"},
    "data_cleanup": {"internal_document_reader", "db_execute"},
    "record_update": {"internal_document_reader", "db_execute"},
    "policy_lookup": {"internal_document_reader", "rag_retriever"},
    "code_generation": {"internal_document_reader"},
    "respond_faq": {"public_faq_bot"},
    "general": {"internal_document_reader", "calendar_assistant"},
}
ALL_TASK_TYPES: frozenset[str] = frozenset(APPROVED_TOOLS_PER_TASK.keys())
DEPRECATED_TOOLS: set[str] = {"legacy_export_tool", "old_crm_connector"}
ALLOWED_COMMANDS_PER_TOOL: dict[str, set[str]] = {
    "internal_document_reader": {"read"},
    "rag_retriever": {"read"},
    "calendar_assistant": {"read", "schedule"},
    "translation_tool": {"translate"},
    "approved_email_assistant": {"draft"},
    "send_external_email": {"send"},
    "external_posting_tool": {"post"},
    "shell_execute": {"read_log"},
    "db_execute": {"read", "update"},
    "file_upload_tool": {"upload"},
    "public_faq_bot": {"respond"},
}
ALLOWED_COMMANDS_PER_TASK_TOOL: dict[tuple[str, str], set[str]] = {
    ("general", "db_execute"): set()
}
MINIMUM_REQUIRED_PERMISSION: dict[str, str] = {
    "internal_document_reader": "read",
    "rag_retriever": "read",
    "calendar_assistant": "read_write",
    "translation_tool": "read",
    "approved_email_assistant": "read_write",
    "send_external_email": "read_write",
    "external_posting_tool": "read_write",
    "public_faq_bot": "read",
    "db_execute": "read_write",
    "shell_execute": "read",
    "file_upload_tool": "read_write",
}
PERMISSION_RANK: dict[str, int] = {"read": 0, "read_write": 1, "admin": 2}
TOOLS_RELEVANT_PER_DEPARTMENT: dict[str, set[str]] = {
    "HR": {
        "internal_document_reader",
        "approved_email_assistant",
        "calendar_assistant",
        "rag_retriever",
        "send_external_email",
        "translation_tool",
    },
    "Finance": {
        "internal_document_reader",
        "db_execute",
        "rag_retriever",
        "calendar_assistant",
        "send_external_email",
    },
    "IT Operations": {
        "shell_execute",
        "db_execute",
        "internal_document_reader",
        "rag_retriever",
        "file_upload_tool",
    },
    "Customer Support": {
        "public_faq_bot",
        "approved_email_assistant",
        "internal_document_reader",
        "rag_retriever",
        "send_external_email",
        "translation_tool",
        "external_posting_tool",
    },
    "Legal & Compliance": {
        "internal_document_reader",
        "rag_retriever",
        "approved_email_assistant",
        "send_external_email",
        "translation_tool",
    },
    "Executive Office": {
        "internal_document_reader",
        "calendar_assistant",
        "approved_email_assistant",
        "rag_retriever",
        "send_external_email",
        "external_posting_tool",
    },
    "Cybersecurity": {
        "internal_document_reader",
        "shell_execute",
        "db_execute",
        "rag_retriever",
        "file_upload_tool",
    },
}
#reverse lookup for simulator to choose a suitable department for a tool
DEPARTMENTS_FOR_TOOL: dict[str, list[str]] = {}
for _dept, _tools in TOOLS_RELEVANT_PER_DEPARTMENT.items():
    for _tool in _tools:
        DEPARTMENTS_FOR_TOOL.setdefault(_tool, []).append(_dept)
#Shared indicators used by rules that inspect prompts, outputs and retrieved content
XSS_PATTERNS: tuple[str, ...] = (
    "<script",
    "onerror=",
    "onload=",
    "javascript:",
    "<img src=x onerror",
)
PATH_TRAVERSAL_PATTERNS: tuple[str, ...] = (
    "../../",
    "..\\..\\",
    "/etc/passwd",
    "c:\\windows\\system32",
)
CREDENTIAL_REGEX = "(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|password\\s*[:=]\\s*\\S+|api[_-]?key\\s*[:=]\\s*\\S+)"
ENV_VAR_REGEX = "\\b[A-Z][A-Z0-9_]{3,}=\\S+\\b"
CONTROL_CHAR_REGEX = "[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]|\\x1b\\[[0-9;]*[a-zA-Z]"
INVISIBLE_UNICODE_REGEX = (
    "[\\u200b-\\u200f\\u202a-\\u202e\\ufe00-\\ufe0f\\U000e0000-\\U000e007f]"
)
BASE64_LOOKING_REGEX = "\\b[A-Za-z0-9+/]{40,}={0,2}\\b"
KNOWN_INTERNAL_TOOL_NAMES: tuple[str, ...] = (
    "internal_document_reader",
    "approved_email_assistant",
    "calendar_assistant",
    "translation_tool",
    "send_external_email",
    "external_posting_tool",
    "db_execute",
    "shell_execute",
    "file_upload_tool",
    "rag_retriever",
    "public_faq_bot",
)
KNOWN_VALID_ENTITIES: set[str] = {
    "requests",
    "pandas",
    "numpy",
    "flask",
    "django",
    "fastapi",
    "express",
    "react",
    "vue",
    "angular",
    "postgresql",
    "mysql",
    "redis",
    "mongodb",
    "docker",
    "kubernetes",
    "nginx",
    "apache",
    "git",
    "npm",
    "pip",
    "webpack",
    "babel",
    "jest",
    "pytest",
    "sqlalchemy",
    "celery",
    "gunicorn",
    "nodejs",
    "typescript",
    "graphql",
    "kafka",
    "rabbitmq",
    "elasticsearch",
    "terraform",
    "ansible",
    "jenkins",
    "boto3",
    "axios",
    "lodash",
    "moment",
    "jquery",
    "bootstrap",
    "tailwindcss",
    "react-router",
    "babel-core",
    "redux-toolkit",
    "next-auth",
    "prisma-client",
    "vue-router",
}
UNAUTHORIZED_DESTINATIONS: tuple[str, ...] = (
    "a personal email account",
    "a personal messaging app",
    "an external cloud storage service",
    "a public code repository",
    "a public social media account",
    "a third-party newsletter platform",
    "personal cloud storage",
    "an external file-sharing link",
)
AGGREGATION_SYSTEM_POOLS: tuple[tuple[str, ...], ...] = (
    (
        "the HR database",
        "the payroll system",
        "the leave management system",
        "the employee self-service portal",
    ),
    (
        "the performance management system",
        "the HR case-management system",
        "the talent management platform",
        "the manager review portal",
    ),
    (
        "the expense management system",
        "the finance ledger",
        "the corporate card platform",
        "the reimbursement portal",
    ),
)

#Measures character randomness as an observable feature for unusually encoded text
def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((n / length * math.log2(n / length) for n in counts.values()))


def has_invisible_unicode(text: str) -> bool:
    return bool(re.search(INVISIBLE_UNICODE_REGEX, text))


def has_base64_looking_text(text: str) -> bool:
    return bool(re.search(BASE64_LOOKING_REGEX, text))
