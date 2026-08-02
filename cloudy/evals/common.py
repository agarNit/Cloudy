"""Shared filtering for eval cases pulled from real traces.

Many entries mined from LangSmith are mid-conversation continuations ("Go.",
"yes", "lets implement this") that only make sense with prior turns as
context — replaying them as a fresh, standalone first message is meaningless
and would unfairly tank retrieval/tool-selection/judge scores for a case that
was never a real standalone query to begin with.
"""

_CONTINUATION_STARTS = (
    "yes", "no", "go", "go.", "lets", "let's", "ok", "okay", "sure",
    "confirmed", "do it", "proceed",
)


def is_self_contained(query: str) -> bool:
    words = query.strip().split()
    if len(words) < 4:
        return False
    first_word = words[0].lower().strip(".,!")
    if first_word in _CONTINUATION_STARTS:
        return False
    return True
