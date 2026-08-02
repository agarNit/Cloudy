from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

from cloudy.observability.logger import get_logger


logger = get_logger(__name__)

# Matches the cap on cloudy's own read_file (cloudy/tools/filesystem_tools.py). Applied
# here too because that cap only covers cloudy's local tool — the MCP filesystem and
# GitHub servers (servers.json) expose their own read tools (read_text_file,
# read_multiple_files, get_file_contents, ...) that cloudy doesn't implement and can't
# cap at the source. This is the single choke point every tool result passes through
# before it can become part of conversation history, regardless of which tool — local,
# MCP, or shell — produced it.
MAX_TOOL_RESULT_CHARS = 200_000

_TRUNCATION_NOTICE = (
    "\n\n[cloudy: output truncated at {limit} characters ({total} total) — "
    "too large to send to the model. Ask a narrower question, or use search_codebase "
    "to pull just the relevant part.]"
)


@wrap_tool_call
async def truncate_oversized_tool_results(request, handler):
    """Cap every tool result to a size that alone can't blow the model's context window.

    A single oversized result is the failure mode this exists to prevent: once it's part
    of conversation history, SummarizationMiddleware's `keep=("messages", N)` policy
    preserves the most recent messages whole regardless of size, so it can't be trimmed
    after the fact — it just breaks every subsequent turn until it ages out. Truncating
    here, before the result ever reaches state, closes that off at the source for any
    tool, not just the ones cloudy implements itself.
    """
    result = await handler(request)

    if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
        return result

    if len(result.content) <= MAX_TOOL_RESULT_CHARS:
        return result

    total = len(result.content)
    logger.warning(
        f"Truncating oversized result from tool '{request.tool_call.get('name')}': "
        f"{total} chars > {MAX_TOOL_RESULT_CHARS} cap"
    )
    truncated = (
        result.content[:MAX_TOOL_RESULT_CHARS]
        + _TRUNCATION_NOTICE.format(limit=MAX_TOOL_RESULT_CHARS, total=total)
    )
    return result.model_copy(update={"content": truncated})
