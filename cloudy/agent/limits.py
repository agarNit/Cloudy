from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware


# Per-turn safety net against genuine runaway loops (a reasoning cycle that never
# converges, or a tool retried over and over), not a tight budget — limits are set
# well above what a real multi-step turn needs. exit_behavior="end" for model calls
# was verified to produce a clean AIMessage appended to state["messages"], no
# exception and no interrupt, so the orchestrator's existing answer path handles it
# with no special-casing. exit_behavior="continue" for tool calls was verified to
# survive cloudy's parallel tool-calling (search_codebase fan-out) without crashing —
# "end" documents a NotImplementedError risk there, so it's avoided.
def build_execution_limits() -> list:
    return [
        ModelCallLimitMiddleware(run_limit=25, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=40, exit_behavior="continue"),
        # Repeated shell retries are the most concrete runaway scenario — a failing
        # command retried in a loop has real side-effect risk beyond just tokens.
        ToolCallLimitMiddleware(tool_name="shell", run_limit=15, exit_behavior="continue"),
    ]


def build_plan_limits() -> list:
    return [
        ModelCallLimitMiddleware(run_limit=15, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=25, exit_behavior="continue"),
    ]
