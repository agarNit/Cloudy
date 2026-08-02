"""Run the full eval loop: retrieval metrics -> agent/tool-selection -> LLM
judge -> correction loop with suggested fixes. Prints a report and saves it
as JSON (gitignored — regenerable, and may contain real query content).

Usage: poetry run python -m cloudy.evals.run_evals
"""
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from cloudy.agent.factory import build_agent
from cloudy.evals.agent_eval import build_eval_checkpointer, evaluate_agent_reruns
from cloudy.evals.build_golden_dataset import GOLDEN_DATASET_PATH
from cloudy.evals.correction_loop import run_correction_loop
from cloudy.evals.judge import judge_case
from cloudy.evals.retrieval_metrics import evaluate_retrieval
from cloudy.observability.logger import get_logger
from cloudy.observability.langfuse_handler import flush_langfuse


logger = get_logger(__name__)
REPORT_PATH = Path(__file__).parent / "latest_report.json"


def _load_golden_dataset() -> list[dict]:
    if not GOLDEN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No golden dataset at {GOLDEN_DATASET_PATH} — run "
            "`poetry run python -m cloudy.evals.build_golden_dataset` first."
        )
    return json.loads(GOLDEN_DATASET_PATH.read_text())


async def run_full_eval() -> dict:
    entries = _load_golden_dataset()
    by_query = {e["query"]: e for e in entries}
    print(f"Loaded {len(entries)} golden entries")

    print("\n--- Retrieval evaluation (precision@k / recall@k / MRR@k) ---")
    retrieval_report = evaluate_retrieval(entries)
    print(f"  cases: {len(retrieval_report.cases)}, skipped: {len(retrieval_report.skipped)}")
    print(f"  mean precision@k: {retrieval_report.mean_precision_at_k:.3f}")
    print(f"  mean recall@k:    {retrieval_report.mean_recall_at_k:.3f}")
    print(f"  MRR@k:            {retrieval_report.mrr_at_k:.3f}")
    retrieval_by_query = {c.query: c for c in retrieval_report.cases}

    print("\n--- Agent evaluation (tool selection + fresh answers) ---")
    checkpointer = build_eval_checkpointer()
    agent = await build_agent(checkpointer, str(Path.cwd()))
    agent_report = await evaluate_agent_reruns(agent, entries)
    print(f"  cases: {len(agent_report.cases)}, skipped: {len(agent_report.skipped)}")
    print(f"  tool-selection accuracy: {agent_report.tool_selection_accuracy:.3f}")

    print("\n--- LLM-as-judge (faithfulness + consistency with baseline) ---")
    judge_semaphore = asyncio.Semaphore(5)

    async def _run_judge(entry: dict, case) -> tuple[dict, object]:
        retrieval_case = retrieval_by_query.get(case.query)
        context = retrieval_case.live_chunk_texts if retrieval_case else []
        async with judge_semaphore:
            try:
                verdict = await judge_case(
                    query=case.query,
                    context=context,
                    baseline_answer=entry["answer"],
                    new_answer=case.new_answer,
                )
            except Exception as e:
                # Structured-output parsing occasionally fails on a long claims list —
                # one bad judge call shouldn't lose every other case's results.
                logger.warning(f"Judge failed for '{case.query[:60]}': {e}")
                return entry, None
        return entry, verdict

    judgeable = [c for c in agent_report.cases if c.kind == "answer" and c.new_answer]
    judge_results = list(await asyncio.gather(*(_run_judge(by_query[c.query], c) for c in judgeable)))
    for case, (_, verdict) in zip(judgeable, judge_results):
        print(f"  [{verdict.consistency_with_baseline}] {case.query[:70]}")

    consistency_counts: dict[str, int] = {}
    for _, v in judge_results:
        consistency_counts[v.consistency_with_baseline] = consistency_counts.get(v.consistency_with_baseline, 0) + 1
    print(f"  judged: {len(judge_results)} | breakdown: {consistency_counts}")

    print("\n--- Correction loop ---")
    correction_report = run_correction_loop(retrieval_report, agent_report, judge_results)
    print(f"  failures identified: {len(correction_report.failures)}")
    for layer, suggestion in correction_report.suggestions.items():
        print(f"\n  [{layer}] suggested fix:\n  {suggestion}\n")

    report = {
        "golden_dataset_size": len(entries),
        "retrieval": {
            "cases": len(retrieval_report.cases),
            "skipped": len(retrieval_report.skipped),
            "mean_precision_at_k": retrieval_report.mean_precision_at_k,
            "mean_recall_at_k": retrieval_report.mean_recall_at_k,
            "mrr_at_k": retrieval_report.mrr_at_k,
        },
        "agent": {
            "cases": len(agent_report.cases),
            "skipped": len(agent_report.skipped),
            "tool_selection_accuracy": agent_report.tool_selection_accuracy,
        },
        "judge": {
            "judged": len(judge_results),
            "consistency_breakdown": consistency_counts,
        },
        "failures": [
            {"query": f.query, "layer": f.layer, "detail": f.detail} for f in correction_report.failures
        ],
        "suggestions": correction_report.suggestions,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nFull report saved to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    load_dotenv()
    try:
        asyncio.run(run_full_eval())
    finally:
        # Short-lived script — Langfuse batches spans on an interval, so the most
        # recent traces would be lost on process exit without an explicit flush.
        flush_langfuse()
