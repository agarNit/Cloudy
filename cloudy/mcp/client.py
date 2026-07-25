from cloudy.observability.logger import get_logger
from cloudy.mcp.config import load_cloudy_mcp_configs
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = get_logger(__name__)

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
            server_tools = await client.get_tools(server_name=name)
            tools.extend(server_tools)
            logger.info(f"Loaded {len(server_tools)} tools from MCP server '{name}'")
        except Exception as e:
            logger.warning(f"Skipping MCP server '{name}': {e}")

    logger.info(f"Loaded {len(tools)} tools from MCP servers")
    return tools
