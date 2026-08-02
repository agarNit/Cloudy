"""Deterministic retrieval evaluation: precision@k, recall@k, MRR@k.

No LLM involved — these are closed-form IR metrics computed against the golden
dataset's frozen retrieved_contexts, treated as the relevant set for each query
(per build_golden_dataset's documented caveat: this is a regression baseline,
not independently-verified ground truth). The retriever is re-run live for
every query, so this measures whether retrieval behavior has drifted from the
snapshot, not absolute correctness.
"""
import re
from dataclasses import dataclass, field

from cloudy.context.retrievers.factory import get_retriever
from cloudy.evals.common import is_self_contained


_CHUNK_HEADER = re.compile(r"File: (.+?) \(lines (\d+)-(\d+)\)")

RETRIEVAL_K = 5


def _chunk_identity(chunk_text: str) -> tuple[str, str, str] | None:
    m = _CHUNK_HEADER.search(chunk_text)
    if not m:
        return None
    return (m.group(1), m.group(2), m.group(3))


def _live_chunks(query: str, k: int = RETRIEVAL_K) -> list[dict]:
    retrieve = get_retriever()
    return retrieve(query, k=k)


@dataclass
class RetrievalCaseResult:
    query: str
    relevant: set
    retrieved: list
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    live_chunk_texts: list = field(default_factory=list)


@dataclass
class RetrievalReport:
    cases: list = field(default_factory=list)
    skipped: list = field(default_factory=list)  # (query, reason)

    @property
    def mean_precision_at_k(self) -> float:
        return sum(c.precision_at_k for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def mean_recall_at_k(self) -> float:
        return sum(c.recall_at_k for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def mrr_at_k(self) -> float:
        return sum(c.reciprocal_rank for c in self.cases) / len(self.cases) if self.cases else 0.0


def evaluate_retrieval(golden_entries: list[dict]) -> RetrievalReport:
    report = RetrievalReport()

    for entry in golden_entries:
        if not is_self_contained(entry["query"]):
            report.skipped.append((entry["query"], "continuation-dependent, not a standalone query"))
            continue

        frozen_contexts = entry.get("retrieved_contexts") or []
        if not frozen_contexts:
            report.skipped.append((entry["query"], "no retrieval happened in the original trace"))
            continue

        relevant = {_chunk_identity(c) for c in frozen_contexts}
        relevant.discard(None)
        if not relevant:
            report.skipped.append((entry["query"], "frozen contexts had no parseable chunk header"))
            continue

        live_chunks = _live_chunks(entry["query"])
        retrieved = [(c["source"], str(c["start_line"]), str(c["end_line"])) for c in live_chunks]
        hits = [r for r in retrieved if r in relevant]

        precision = len(hits) / len(retrieved) if retrieved else 0.0
        recall = len(set(hits)) / len(relevant)
        reciprocal_rank = 0.0
        for rank, r in enumerate(retrieved, start=1):
            if r in relevant:
                reciprocal_rank = 1.0 / rank
                break

        report.cases.append(RetrievalCaseResult(
            query=entry["query"],
            relevant=relevant,
            retrieved=retrieved,
            precision_at_k=precision,
            recall_at_k=recall,
            reciprocal_rank=reciprocal_rank,
            live_chunk_texts=[c["content"] for c in live_chunks],
        ))

    return report
