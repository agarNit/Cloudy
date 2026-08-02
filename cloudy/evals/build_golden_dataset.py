"""Freeze real agent traces from LangSmith into a golden dataset.

This is a regression baseline, not verified ground truth: each entry captures
what cloudy actually did for a real query at build time (retrieved context,
tools called, final answer), taken as correct as-is — no human review step.
Re-running these same queries later and diffing against this frozen snapshot
surfaces behavior drift after a prompt/retrieval/model change; it does not
independently confirm the frozen answer was correct to begin with. Metrics
that lean on the frozen answer as a reference (context precision/recall,
answer correctness) inherit that caveat — faithfulness and answer relevancy
don't, since they only check today's answer against today's own context.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from cloudy.observability.logger import get_logger


logger = get_logger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def _extract_query(run) -> str | None:
    messages = (run.inputs or {}).get("messages") or []
    for m in reversed(messages):
        if m.get("role") == "user":
            return m["content"]
    return None


def _extract_final_answer(run) -> str | None:
    messages = (run.outputs or {}).get("messages") or []
    for m in reversed(messages):
        if m.get("type") != "ai":
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            if text.strip():
                return text
    return None


def _extract_tool_names(child_runs) -> list[str]:
    return sorted({c.name for c in child_runs if c.run_type == "tool"})


def _extract_retrieved_contexts(child_runs) -> list[str]:
    # search_codebase joins chunks with "\n---\n" (cloudy/agent/tools.py) — split back
    # into individual chunk texts so ragas gets one context per retrieved chunk.
    for c in child_runs:
        if c.name != "search_codebase":
            continue
        output = (c.outputs or {}).get("output")
        text = output.get("content") if isinstance(output, dict) else output
        if not text or text == "No relevant code found.":
            continue
        return [chunk.strip() for chunk in text.split("\n---\n") if chunk.strip()]
    return []


def build_golden_dataset(project: str | None = None, limit: int = 100) -> list[dict]:
    client = Client()
    project = project or os.environ.get("LANGSMITH_PROJECT", "Cloudy")

    roots = list(client.list_runs(project_name=project, run_type="chain", is_root=True, limit=limit))
    logger.info(f"Pulled {len(roots)} root traces from LangSmith project '{project}'")

    by_query: dict[str, dict] = {}
    for root in roots:
        if root.status != "success":
            continue  # errored/pending-approval traces have no clean final answer to freeze

        query = _extract_query(root)
        answer = _extract_final_answer(root)
        if not query or not answer:
            continue

        children = list(client.list_runs(project_name=project, trace_id=root.trace_id))
        entry = {
            "id": str(root.id),
            "query": query,
            "answer": answer,
            "retrieved_contexts": _extract_retrieved_contexts(children),
            "tools_called": _extract_tool_names(children),
            "captured_at": (root.start_time or datetime.now(timezone.utc)).isoformat(),
            "source": "langsmith_trace",
        }

        # Same query captured more than once (re-run during dev/testing) — keep the
        # most recent capture only, not every duplicate.
        existing = by_query.get(query)
        if existing is None or entry["captured_at"] > existing["captured_at"]:
            by_query[query] = entry

    entries = sorted(by_query.values(), key=lambda e: e["captured_at"])
    return entries


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def merge_golden_dataset(existing: list[dict], candidates: list[dict]) -> tuple[list[dict], int]:
    """Add genuinely new queries; never touch an entry already in the set.

    Once a query is frozen as a golden case, it stays exactly as originally captured —
    that's what makes it useful as a regression baseline. A fresher trace for the same
    query is discarded here, not used to update it; only queries not seen before get added.
    """
    by_query = {e["query"]: e for e in existing}
    added = 0
    for candidate in candidates:
        if candidate["query"] not in by_query:
            by_query[candidate["query"]] = candidate
            added += 1
    merged = sorted(by_query.values(), key=lambda e: e["captured_at"])
    return merged, added


def save_golden_dataset(entries: list[dict], path: Path = GOLDEN_DATASET_PATH) -> None:
    path.write_text(json.dumps(entries, indent=2))
    logger.info(f"Wrote {len(entries)} golden examples to {path}")


if __name__ == "__main__":
    load_dotenv()
    existing = load_golden_dataset()
    candidates = build_golden_dataset()
    merged, added = merge_golden_dataset(existing, candidates)
    save_golden_dataset(merged)
    print(
        f"{added} new golden example(s) added, {len(existing)} existing entries kept frozen "
        f"-> {len(merged)} total, written to {GOLDEN_DATASET_PATH}"
    )
