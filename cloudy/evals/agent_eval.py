"""Tool-selection accuracy: re-run each self-contained golden query through the
real agent (fresh in-memory thread per case, so eval runs never touch the real
session checkpoint DB) and compare which tools actually got called against the
frozen set from the golden dataset.

Deterministic — exact-set comparison, no LLM judge involved. This reuses the
same agent + HITL gating as production, so a query that would trigger an
approval today (e.g. a destructive shell command) just stops at kind="approval"
here too — nothing destructive ever actually executes during an eval run.
"""
import asyncio
import uuid
from dataclasses import dataclass, field

from langgraph.checkpoint.memory import InMemorySaver

from cloudy.agent.orchestrator import handle_query
from cloudy.evals.common import is_self_contained
from cloudy.observability.logger import get_logger


logger = get_logger(__name__)

# Cases are independent (separate thread_ids on a single-threaded asyncio event
# loop — InMemorySaver keys by thread_id with no cross-key contention, so this is
# safe concurrency, not the kind of real multi-threading that caused the embedder
# SIGSEGV elsewhere in this project). Capped, not unbounded, to stay reasonable
# against Anthropic's rate limits rather than firing 35 requests at once.
MAX_CONCURRENT_CASES = 5


@dataclass
class AgentCaseResult:
    query: str
    expected_tools: set
    actual_tools: set
    tools_match: bool
    kind: str  # "answer" | "approval"
    new_answer: str | None


@dataclass
class AgentEvalReport:
    cases: list = field(default_factory=list)
    skipped: list = field(default_factory=list)  # (query, reason)

    @property
    def tool_selection_accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.tools_match) / len(self.cases)


async def _run_case(agent, entry: dict, semaphore: asyncio.Semaphore) -> AgentCaseResult:
    async with semaphore:
        thread_id = f"eval-{uuid.uuid4()}"
        result = await handle_query(agent, entry["query"], thread_id)

    expected_tools = set(entry.get("tools_called") or [])
    actual_tools = set(result.tool_names)

    return AgentCaseResult(
        query=entry["query"],
        expected_tools=expected_tools,
        actual_tools=actual_tools,
        tools_match=expected_tools == actual_tools,
        kind=result.kind,
        new_answer=result.answer if result.kind == "answer" else None,
    )


async def evaluate_agent_reruns(agent, golden_entries: list[dict]) -> AgentEvalReport:
    report = AgentEvalReport()

    runnable = []
    for entry in golden_entries:
        if not is_self_contained(entry["query"]):
            report.skipped.append((entry["query"], "continuation-dependent, not a standalone query"))
            continue
        runnable.append(entry)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CASES)
    report.cases = list(await asyncio.gather(*(_run_case(agent, entry, semaphore) for entry in runnable)))

    return report


def build_eval_checkpointer() -> InMemorySaver:
    return InMemorySaver()
