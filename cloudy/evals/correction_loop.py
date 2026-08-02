"""Bucket eval failures by which layer actually caused them, then ask for a
concrete suggested fix per bucket — not per individual case, since grouping
similar failures avoids redundant or conflicting suggestions.

The bucketing itself is the important part, not the LLM call: retrieval
misses, generation/grounding misses, and tool-selection misses need different
fixes (chunking/embedding config vs. system-prompt grounding instructions vs.
tool descriptions), and treating them as one undifferentiated "answer was
wrong" bucket would point suggestions at the wrong layer.
"""
from dataclasses import dataclass, field

from cloudy.agent.factory import SYSTEM_PROMPT
from cloudy.llm.factory import get_llm
from cloudy.observability.logger import get_logger


logger = get_logger(__name__)

RECALL_FAILURE_THRESHOLD = 0.5
FAITHFULNESS_FAILURE_THRESHOLD = 0.7


@dataclass
class FailureCase:
    query: str
    layer: str  # "retrieval" | "grounding" | "tool_selection"
    detail: str


@dataclass
class CorrectionReport:
    failures: list = field(default_factory=list)
    suggestions: dict = field(default_factory=dict)  # layer -> suggestion text


def bucket_failures(retrieval_report, agent_report, judge_results: list[tuple[dict, object]]) -> list[FailureCase]:
    failures = []

    for case in retrieval_report.cases:
        if case.recall_at_k < RECALL_FAILURE_THRESHOLD:
            failures.append(FailureCase(
                query=case.query,
                layer="retrieval",
                detail=f"recall@k={case.recall_at_k:.2f} — expected chunks {case.relevant} not found in "
                       f"live retrieval {case.retrieved}",
            ))

    for case in agent_report.cases:
        if not case.tools_match:
            failures.append(FailureCase(
                query=case.query,
                layer="tool_selection",
                detail=f"expected tools {case.expected_tools}, actually called {case.actual_tools}",
            ))

    for entry, verdict in judge_results:
        if verdict is None:
            continue
        from cloudy.evals.judge import faithfulness_score
        fscore = faithfulness_score(verdict)
        is_grounding_failure = (
            (fscore is not None and fscore < FAITHFULNESS_FAILURE_THRESHOLD)
            or verdict.consistency_with_baseline == "regressed"
        )
        if is_grounding_failure:
            failures.append(FailureCase(
                query=entry["query"],
                layer="grounding",
                detail=f"faithfulness={fscore}, consistency={verdict.consistency_with_baseline} — {verdict.reasoning}",
            ))

    return failures


_SUGGESTION_PROMPT = """You are helping improve a coding assistant agent. Here is its current system prompt:

--- SYSTEM PROMPT ---
{system_prompt}
--- END SYSTEM PROMPT ---

The following real eval failures were all categorized as "{layer}" issues:

{failure_list}

Layer definitions:
- retrieval: the right code chunk wasn't found — fix belongs in chunking/embedding/retriever config, NOT the prompt.
- tool_selection: the agent called the wrong tool(s) for the query — fix belongs in tool descriptions or the \
system prompt's tool-selection instructions.
- grounding: the answer wasn't well-supported by retrieved context, or regressed from a previous baseline — fix \
belongs in system prompt instructions about staying grounded in retrieved context.

Give a specific, concrete suggested change for this "{layer}" bucket — quote or reference the exact part of the \
system prompt to change if the fix belongs there, or name the specific retrieval/config parameter if it doesn't. \
Do not give generic advice like "improve prompt clarity" — name the actual change."""


def suggest_fixes(failures: list[FailureCase]) -> dict[str, str]:
    if not failures:
        return {}

    llm = get_llm()
    by_layer: dict[str, list[FailureCase]] = {}
    for f in failures:
        by_layer.setdefault(f.layer, []).append(f)

    suggestions = {}
    for layer, cases in by_layer.items():
        failure_list = "\n".join(f"- Query: {c.query}\n  Detail: {c.detail}" for c in cases)
        prompt = _SUGGESTION_PROMPT.format(system_prompt=SYSTEM_PROMPT, layer=layer, failure_list=failure_list)
        response = llm.invoke(prompt)
        suggestions[layer] = response.content
        logger.info(f"Generated suggestion for {len(cases)} '{layer}' failures")

    return suggestions


def run_correction_loop(retrieval_report, agent_report, judge_results: list[tuple[dict, object]]) -> CorrectionReport:
    failures = bucket_failures(retrieval_report, agent_report, judge_results)
    suggestions = suggest_fixes(failures)
    return CorrectionReport(failures=failures, suggestions=suggestions)
