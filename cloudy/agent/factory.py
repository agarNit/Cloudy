from langchain.agents import create_agent


from cloudy.llm.factory import get_llm
from cloudy.agent.tools import search_codebase
from cloudy.observability.logger import get_logger
from cloudy.tools.terminal_tools import run_command, run_in_directory
from cloudy.tools.filesystem_tools import (
  read_file,
  write_file,
  append_file,
  list_directory,
  file_exists,
)
from cloudy.mcp.client import get_mcp_tools
from cloudy.skills.tools import load_skill, build_skills_prompt
from cloudy.memory.tools import find_session, save_memory, recall_memory
from cloudy.memory.semantic import build_memory_prompt


logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a senior software engineer with deep knowledge of the codebase.

Tool selection:
- Code questions (how something works, where it's defined, architecture, behavior, bugs) \
— call search_codebase before answering. Reference specific file names, function names \
and line numbers in your answers.
- Questions about past conversations ("did I ask about X before", "what did I work on \
last time") — call find_session instead, not search_codebase.
- If the relevant tool doesn't turn up an answer, say so explicitly rather than guessing.

Long-term memory (save_memory / recall_memory — durable facts, separate from find_session \
which is past conversations):
- If the user explicitly asks you to remember or save something, you MUST call \
save_memory. Replying that you will remember it is NOT enough — only the tool call makes \
it persist beyond this conversation.
- Auto-call save_memory, without being asked, when the user corrects your approach or you \
both land on a firm project decision (category_type "feedback" or "project").
- Only save a "preference" when explicitly asked to remember it, never from a passing \
mention.
- Never save anything specific to the current task, or anything derivable just by reading \
the code. Full criteria are in save_memory's own docstring."""


async def build_agent(checkpointer):
  """Create and return a LangChain agent with persistent memory."""
  llm = get_llm()
  mcp_tools = await get_mcp_tools()


  skills_prompt = build_skills_prompt()
  memory_prompt = await build_memory_prompt()
  full_prompt = SYSTEM_PROMPT
  if skills_prompt:
      full_prompt += "\n\n" + skills_prompt
  if memory_prompt:
      full_prompt += "\n\n" + memory_prompt


  tools = [
      search_codebase,
      load_skill,
      find_session,
      save_memory,
      recall_memory,
      run_command,
      run_in_directory,
      read_file,
      write_file,
      append_file,
      list_directory,
      file_exists,
      *mcp_tools,
  ]


  return create_agent(
      llm,
      tools=tools,
      system_prompt=full_prompt,
      checkpointer=checkpointer,
  )
