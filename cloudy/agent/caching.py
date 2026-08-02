from langchain.agents.middleware import wrap_model_call

from cloudy.observability.logger import get_logger


logger = get_logger(__name__)


@wrap_model_call
async def cache_system_and_tools(request, handler):
    """Marks the system prompt + tool schemas as one cacheable prefix
    (Anthropic prompt caching). A cache breakpoint on the last tool caches
    everything before it too — system prompt included — so this is the only
    breakpoint needed; no separate one on the system message itself.

    Below Anthropic's minimum cacheable size this is a harmless no-op — the
    marker just does nothing until the combined system+tools block is big
    enough to clear it (confirmed empirically to be well above the
    documented 2048-token figure for tool-based breakpoints specifically —
    somewhere between ~4K and ~11.7K tokens). Safe to leave on regardless;
    it starts paying off automatically as the tool list or system prompt
    grows, with no further changes needed.
    """
    if not request.tools:
        return await handler(request)

    bound = request.model.bind_tools(request.tools)
    anthropic_tools = [dict(t) for t in bound.kwargs["tools"]]
    anthropic_tools[-1] = {**anthropic_tools[-1], "cache_control": {"type": "ephemeral"}}
    request = request.override(tools=anthropic_tools)

    return await handler(request)
