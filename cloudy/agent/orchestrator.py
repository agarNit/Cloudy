import time
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import RemoveMessage
from langchain.agents.middleware._redaction import PIIDetectionError
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from langfuse import propagate_attributes

from cloudy.observability.logger import get_logger
from cloudy.observability.langfuse_handler import get_langfuse_handler
from cloudy.memory.session import touch_session


logger = get_logger(__name__)

# Matches the tool-result cap in cloudy.agent.tool_output_guard — used here to spot
# which checkpointed message(s) are actually large enough to be the cause of an
# overflow, so just those can be dropped instead of nuking the whole conversation.
_OVERSIZED_MESSAGE_CHARS = 200_000


async def _recover_from_context_overflow(agent, thread_id: str) -> None:
    """A context-overflow error means the checkpointed history for this thread is now
    too large to ever send to the model again — left alone, every future turn on this
    thread fails identically, permanently, even a trivial "Hi" (this is exactly what
    happened before truncate_oversized_tool_results existed: a single huge tool result
    got pinned in SummarizationMiddleware's "keep last N messages" window forever).
    Strip whatever's oversized so the session can keep going instead of staying wedged.

    Tries to drop just the individual message(s) big enough to be the cause. Falls back
    to clearing the whole history if nothing single is large enough to explain it — an
    accumulation of many smaller messages, which can't be fixed surgically.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await agent.aget_state(config)
    messages = snapshot.values.get("messages", [])

    oversized = [
        m for m in messages
        if isinstance(getattr(m, "content", None), str) and len(m.content) > _OVERSIZED_MESSAGE_CHARS
    ]

    if oversized:
        await agent.aupdate_state(config, {"messages": [RemoveMessage(id=m.id) for m in oversized]})
        logger.warning(f"Context overflow on {thread_id}: dropped {len(oversized)} oversized message(s)")
    else:
        await agent.aupdate_state(config, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]})
        logger.warning(f"Context overflow on {thread_id}: no single oversized message found, cleared history")


class _TokenCounter(BaseCallbackHandler):
  """Sums input/output/cache tokens across every model call made during a single invoke."""

  def __init__(self):
      self.input_tokens = 0
      self.output_tokens = 0
      self.cache_read_tokens = 0
      self.cache_creation_tokens = 0
      self.tool_names: set[str] = set()

  def on_llm_end(self, response, **kwargs):
      for generations in response.generations:
          for generation in generations:
              usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
              if usage:
                  self.input_tokens += usage.get("input_tokens", 0)
                  self.output_tokens += usage.get("output_tokens", 0)
                  details = usage.get("input_token_details") or {}
                  self.cache_read_tokens += details.get("cache_read", 0)
                  self.cache_creation_tokens += details.get("cache_creation", 0)

  def on_tool_start(self, serialized, input_str, **kwargs):
      name = (serialized or {}).get("name")
      if name:
          self.tool_names.add(name)


@dataclass
class ApprovalRequest:
  """One tool call a HumanInTheLoopMiddleware gate is holding, awaiting a decision."""
  name: str
  args: dict
  description: str


@dataclass
class QueryResult:
  """Outcome of a single agent.ainvoke — either a final answer, or a set of pending
  approvals the caller must resolve (via resume_query) before the turn can continue.
  """
  kind: str  # "answer" | "approval"
  answer: str | None = None
  approvals: list[ApprovalRequest] = field(default_factory=list)
  todos: list[dict] = field(default_factory=list)
  stats: dict = field(default_factory=dict)
  tool_names: set[str] = field(default_factory=set)


def _stats(started: float, counter: _TokenCounter) -> dict:
  return {
      "elapsed_seconds": time.monotonic() - started,
      "input_tokens": counter.input_tokens,
      "output_tokens": counter.output_tokens,
      "cache_read_tokens": counter.cache_read_tokens,
      "cache_creation_tokens": counter.cache_creation_tokens,
  }


def _to_result(response: dict, stats: dict) -> QueryResult:
  todos = response.get("todos") or []
  if "__interrupt__" in response:
      payload = response["__interrupt__"][0].value
      approvals = [
          ApprovalRequest(name=a["name"], args=a["args"], description=a.get("description", ""))
          for a in payload["action_requests"]
      ]
      return QueryResult(kind="approval", approvals=approvals, todos=todos, stats=stats)

  answer = response["messages"][-1].content
  return QueryResult(kind="answer", answer=answer, todos=todos, stats=stats)


async def _invoke(agent, payload, thread_id: str, environment: str = "development") -> QueryResult:
  counter = _TokenCounter()
  agent_config = {
      "configurable": {"thread_id": thread_id},
      "callbacks": [counter, get_langfuse_handler()],
  }
  started = time.monotonic()
  try:
      # session_id groups this whole conversation into one Langfuse session; environment
      # keeps eval-script traffic ("eval") separate from real interactive usage
      # ("development") in the dashboard. Must wrap ainvoke itself — propagate_attributes
      # only applies to spans created after entering the context, and the root span is
      # created the moment ainvoke starts.
      with propagate_attributes(session_id=thread_id, environment=environment):
          response = await agent.ainvoke(payload, agent_config)
  except PIIDetectionError as e:
      logger.info(f"Blocked on {e.pii_type}: {len(e.matches)} match(es)")
      answer = (
          f"This looks like it contains {e.pii_type.replace('_', ' ')} data, which I can't "
          f"process — please remove it and try again."
      )
      return QueryResult(kind="answer", answer=answer, stats=_stats(started, counter))
  except ContextOverflowError as e:
      logger.error(f"Context overflow on {thread_id}: {e}")
      await _recover_from_context_overflow(agent, thread_id)
      answer = (
          "This conversation's history grew too large for the model's context window, "
          "so I've cleared the oversized part of it. You can keep chatting in this "
          "session — some earlier context may be gone, but everything else is intact."
      )
      return QueryResult(kind="answer", answer=answer, stats=_stats(started, counter))
  except Exception as e:
      logger.error(f"Agent error: {e}")
      return QueryResult(kind="answer", answer=f"Error: {e}", stats=_stats(started, counter))

  result = _to_result(response, _stats(started, counter))
  result.tool_names = counter.tool_names
  if result.kind == "answer":
      await touch_session(thread_id)
  return result


async def handle_query(agent, question: str, thread_id: str, environment: str = "development") -> QueryResult:
  """Entry point for a new user question — may return a final answer or a pending
  approval if a HumanInTheLoopMiddleware gate fires during this turn.
  """
  logger.info(f"Handling query for session {thread_id}: {question}")
  return await _invoke(
      agent, {"messages": [{"role": "user", "content": question}]}, thread_id, environment
  )


async def resume_query(agent, decisions: list[dict], thread_id: str) -> QueryResult:
  """Continue a turn after the caller has collected decisions for a pending approval.
  May itself return another pending approval if more gates fire further down the line.
  """
  logger.info(f"Resuming session {thread_id} with {len(decisions)} decision(s)")
  return await _invoke(agent, Command(resume={"decisions": decisions}), thread_id)


async def get_pending_approval(agent, thread_id: str) -> list[ApprovalRequest]:
  """Check whether this thread is currently paused at an approval gate — e.g. the
  process died or the user disconnected before answering. Lets a resumed session
  re-present the same pending decision instead of silently dropping it.
  """
  snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
  for task in snapshot.tasks:
      if task.interrupts:
          payload = task.interrupts[0].value
          return [
              ApprovalRequest(name=a["name"], args=a["args"], description=a.get("description", ""))
              for a in payload["action_requests"]
          ]
  return []
