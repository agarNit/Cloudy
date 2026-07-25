import time

from langchain_core.callbacks import BaseCallbackHandler

from cloudy.observability.logger import get_logger


logger = get_logger(__name__)


class _TokenCounter(BaseCallbackHandler):
  """Sums input/output tokens across every model call made during a single invoke."""

  def __init__(self):
      self.input_tokens = 0
      self.output_tokens = 0

  def on_llm_end(self, response, **kwargs):
      for generations in response.generations:
          for generation in generations:
              usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
              if usage:
                  self.input_tokens += usage.get("input_tokens", 0)
                  self.output_tokens += usage.get("output_tokens", 0)


async def handle_query(agent, question: str, thread_id: str) -> tuple[str, dict]:
  """Entry point for all user queries - invokes the agent.

  Returns (answer, stats) where stats has elapsed_seconds/input_tokens/output_tokens
  for just this call — the checkpointer accumulates full thread history, so token
  counts are captured via a per-call callback rather than summed from the messages.
  """
  logger.info(f"Handling query for session {thread_id}: {question}")
  counter = _TokenCounter()
  agent_config = {"configurable": {"thread_id": thread_id}, "callbacks": [counter]}
  started = time.monotonic()
  try:
      response = await agent.ainvoke({"messages": [{"role": "user", "content": question}]}, agent_config)
      answer = response["messages"][-1].content
  except Exception as e:
      logger.error(f"Agent error: {e}")
      answer = f"Error: {e}"

  stats = {
      "elapsed_seconds": time.monotonic() - started,
      "input_tokens": counter.input_tokens,
      "output_tokens": counter.output_tokens,
  }
  return answer, stats
