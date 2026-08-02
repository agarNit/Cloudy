"""LLM-as-judge for answer quality.

Two things are checked per case, and they don't have the same trust profile:

- Faithfulness: the new answer's claims are decomposed and each is checked
  against *today's* retrieved context. Self-contained — doesn't depend on the
  golden dataset's frozen answer being correct, so this score is trustworthy
  even though the dataset itself is an unverified snapshot.
- Consistency with baseline: a pairwise comparison against the frozen answer.
  This one inherits the golden dataset's caveat directly — a "regressed"
  verdict means "diverged from what the trace captured," not necessarily
  "got worse," since the frozen answer was never independently verified as
  correct either.

Calibrate before trusting this on the full set: grade a small sample yourself
(judge_case has an inputs param for exactly this — call it directly on a few
hand-picked cases, compare against your own read of the same case) before
relying on it across the whole dataset. This module does not skip that step
for you.
"""
from pydantic import BaseModel, Field

from cloudy.llm.factory import get_llm
from cloudy.observability.logger import get_logger


logger = get_logger(__name__)


class ClaimCheck(BaseModel):
    claim: str = Field(description="One discrete factual/technical claim extracted from the new answer")
    supported_by_context: bool = Field(description="Whether the retrieved context actually supports this claim")


class JudgeVerdict(BaseModel):
    claims: list[ClaimCheck] = Field(
        default_factory=list,
        description="Claims extracted from the new answer, each checked against retrieved context. "
        "Empty if no context was retrieved for this query.",
    )
    consistency_with_baseline: str = Field(
        description="One of: same, improved, regressed, different_but_valid, unclear"
    )
    reasoning: str = Field(
        description="Concise explanation of the consistency verdict and any faithfulness issues found — "
        "specific enough to act on (what's wrong, not just that something's wrong)"
    )


_JUDGE_PROMPT = """You are evaluating a coding assistant's answer for a regression test suite.

QUESTION:
{query}

RETRIEVED CONTEXT (what the assistant had available today):
{context}

BASELINE ANSWER (captured previously, treated as the reference point — not guaranteed correct):
{baseline_answer}

NEW ANSWER (produced just now, what you are evaluating):
{new_answer}

Do two things:

1. Extract the individual factual/technical claims made in the NEW ANSWER (e.g. "X is implemented in file Y", \
"the default timeout is 30s"). For each, mark whether the RETRIEVED CONTEXT actually supports it. If there is no \
retrieved context, return an empty claims list — do not guess.

2. Compare the NEW ANSWER against the BASELINE ANSWER. Classify as:
   - "same": substantively the same information
   - "improved": new answer is more accurate/complete/grounded than baseline
   - "regressed": new answer dropped or contradicted something correct the baseline had, or is now less grounded
   - "different_but_valid": answers differ but both are reasonable/correct (e.g. different phrasing, both accurate)
   - "unclear": can't confidently tell

Give concise, specific reasoning — this feeds into a report suggesting what to fix, so name the actual issue \
(e.g. "new answer claims freshness uses a 5s window, context shows 8s") rather than a vague quality statement."""


async def judge_case(
    query: str,
    context: list[str],
    baseline_answer: str,
    new_answer: str,
) -> JudgeVerdict:
    llm = get_llm().with_structured_output(JudgeVerdict)
    context_text = "\n---\n".join(context) if context else "(no context retrieved for this query)"
    prompt = _JUDGE_PROMPT.format(
        query=query, context=context_text, baseline_answer=baseline_answer, new_answer=new_answer,
    )
    return await llm.ainvoke(prompt)


def faithfulness_score(verdict: JudgeVerdict) -> float | None:
    if not verdict.claims:
        return None
    supported = sum(1 for c in verdict.claims if c.supported_by_context)
    return supported / len(verdict.claims)
