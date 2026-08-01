from langchain.tools import tool

from cloudy.memory.episodic import find_sessions
from cloudy.memory.semantic import save_memory as _save_memory, recall_memory as _recall_memory
from cloudy.observability.logger import get_logger


logger = get_logger(__name__)


@tool
async def save_memory(category_type: str, category: str, summary: str, content: str) -> str:
    """Save or update a durable, general fact/preference/decision so it's available
    in future sessions too. This is NOT for anything specific to the current task.

    Only save something if it passes ALL of these:
    - Durable: still true next week, in a different session — not just for now.
    - General: applies broadly, not just to what's being worked on right now.
    - Not derivable from the code: can't be figured out by just reading the repo.
    - Landed: a firm decision or a correction, not an open exploratory discussion.

    category_type must be exactly one of:
    - "preference": how the user likes to work (verbosity, style, tool choices).
      Save one when the user directly tells you to remember or save it — e.g.
      "remember that I prefer X", "please save this preference", "keep in mind
      going forward that...". If a preference only comes up in passing, without
      being asked to save it — e.g. "I like terse answers, by the way" — just
      note it for this conversation instead of saving it.
    - "feedback": a correction about your approach. Save these automatically,
      without being asked, whenever the user corrects how you're doing something.
    - "project": a durable decision about this codebase (architecture, why
      something was chosen over an alternative). Save these automatically,
      without being asked, whenever such a decision is actually reached.

    category should be a short kebab-case slug, e.g. "shell-guardrails-approach".
    Saving again with the same category_type + category updates that entry in
    place instead of creating a duplicate — use the same slug when correcting
    or refining something already saved.
    """
    logger.info(f"Tool called: save_memory({category_type}, {category})")
    return await _save_memory(category_type, category, summary, content)


@tool
async def recall_memory(category_type: str, category: str) -> str:
    """Load the full detail behind a memory entry you saw summarized in your
    system prompt, under "=== Known Project Memory ===". Use the exact
    category_type and category shown there, e.g.
    recall_memory("feedback", "shell-guardrails-approach").
    """
    logger.info(f"Tool called: recall_memory({category_type}, {category})")
    return await _recall_memory(category_type, category)


@tool
async def find_session(query: str) -> str:
    """Search past conversation sessions (from earlier, different threads) for ones
    relevant to a topic or question. Use this when the user asks whether something
    was discussed before, or wants to find which past session covered a topic.
    Returns matching session ids with their summaries and dates — the user can
    switch to one with /switch <session_id> to see the full conversation again.
    """
    logger.info(f"Tool called: find_session with query: {query}")
    results = await find_sessions(query)
    if not results:
        return "No past sessions found."

    lines = []
    for r in results:
        lines.append(
            f"Session {r['session_id']} (last active {r['ended_at']}): {r['summary']}"
        )
    return "\n".join(lines)
