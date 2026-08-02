import contextlib
import os

from cloudy.observability.logger import get_logger
from cloudy.mcp.config import load_cloudy_mcp_configs
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = get_logger(__name__)


@contextlib.contextmanager
def _suppress_subprocess_stderr():
    """MCP server subprocesses (github/filesystem servers spawned via npx) print
    their own startup banners straight to stderr. Reassigning sys.stderr in
    Python has no effect here: mcp.client.stdio.stdio_client's errlog default is
    `sys.stderr`, bound once when that module is first imported — a classic
    mutable-default-argument-style gotcha — and subprocess creation reads the
    underlying OS file descriptor, not the Python object, regardless. Swapping
    the actual fd is the only thing that reaches it. Cloudy's own per-server
    error handling below already goes through the structured logger, not this
    fd, so no real error information is lost by suppressing it.
    """
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)


async def get_mcp_tools() -> list:
    """Connect to all configured MCP servers and return their tools.

    Each server is loaded independently — a server that fails to start
    (missing binary, bad token, network issue, etc.) is logged and skipped
    rather than taking down the whole app.
    """
    configs = load_cloudy_mcp_configs()
    logger.info("Connecting to MCP servers: %s", list(configs.keys()))
    client = MultiServerMCPClient(configs)

    tools = []
    for name in configs:
        try:
            with _suppress_subprocess_stderr():
                server_tools = await client.get_tools(server_name=name)
            tools.extend(server_tools)
            logger.info(f"Loaded {len(server_tools)} tools from MCP server '{name}'")
        except Exception as e:
            logger.warning(f"Skipping MCP server '{name}': {e}")

    logger.info(f"Loaded {len(tools)} tools from MCP servers")
    return tools
