import re

from langchain.agents import create_agent
from langchain.agents.middleware import (
  TodoListMiddleware,
  HumanInTheLoopMiddleware,
  InterruptOnConfig,
)


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
the code. Full criteria are in save_memory's own docstring.

Task planning: for multi-step work (3+ distinct actions), use write_todos to track \
progress — mark each step in_progress before starting it and completed immediately after, \
never batched. Skip it for simple, few-step requests."""


PLAN_SYSTEM_PROMPT = """You are a senior software engineer in PLANNING mode.

You do not have access to any tool that changes anything — no writing files, no running \
commands. Your only job is to understand what the user wants and produce a clear, ordered \
plan.

- Ask clarifying questions if the request is ambiguous or you're missing information you'd \
need to do it well.
- Use search_codebase, read_file, list_directory, file_exists, find_session, and \
recall_memory freely to understand the current state of the codebase before proposing steps.
- Once you have enough information, call write_todos with the final step-by-step plan. Each \
step should be a concrete, independently-completable action.
- Do not describe the plan in prose instead of calling write_todos — the todo list is the \
plan; prose is only for discussion before it.
- After write_todos, wait — do not repeat or restate the plan in the same turn."""


# Approval is required for every write_file/append_file call (any file write is a "major
# change"), but for shell commands only when the command itself looks destructive — otherwise
# every `ls` or `pytest` would need a manual approval, which trains users to rubber-stamp
# everything and defeats the point.
_DANGEROUS_COMMAND_PATTERN = re.compile(
  r"\b(rm|rmdir|kill|killall|pkill|del|delete|drop|truncate|dd|mkfs|shutdown|reboot|"
  r"chmod|chown|mv|format)\b",
  re.IGNORECASE,
)


def _is_dangerous_command(request) -> bool:
  command = request.tool_call["args"].get("command", "")
  return bool(_DANGEROUS_COMMAND_PATTERN.search(command))


# MCP servers are configurable (servers.json) and their tool names aren't known ahead of
# time, so we can't hardcode which ones to gate the way we do for our own local tools.
# Instead, flag any MCP tool whose name looks like it mutates something — this way a newly
# added MCP server (or a version bump that adds new tools) doesn't silently reopen approval
# gaps like the one found here: the GitHub MCP server's push_files/merge_pull_request/
# create_repository etc., and the filesystem server's edit_file/move_file, were all running
# with zero approval before this existed.
_MUTATING_VERB_STEMS = (
  "write", "edit", "create", "delete", "remove", "push", "merge",
  "update", "move", "fork", "add", "rename", "drop", "truncate",
)


def _is_mutating_tool_name(name: str) -> bool:
  tokens = name.lower().split("_")
  return any(token.startswith(stem) for token in tokens for stem in _MUTATING_VERB_STEMS)


def _build_interrupt_on(mcp_tools) -> dict:
  interrupt_on = {
      "write_file": InterruptOnConfig(allowed_decisions=["approve", "reject"]),
      "append_file": InterruptOnConfig(allowed_decisions=["approve", "reject"]),
      "run_command": InterruptOnConfig(
          allowed_decisions=["approve", "reject"], when=_is_dangerous_command
      ),
      "run_in_directory": InterruptOnConfig(
          allowed_decisions=["approve", "reject"], when=_is_dangerous_command
      ),
  }
  for tool in mcp_tools:
      if tool.name in interrupt_on:
          continue
      if _is_mutating_tool_name(tool.name):
          interrupt_on[tool.name] = InterruptOnConfig(allowed_decisions=["approve", "reject"])
          logger.info(f"Gating MCP tool for approval: {tool.name}")
  return interrupt_on


async def build_plan_agent(checkpointer):
  """Read-only planning agent — structurally incapable of taking action, not just told
  not to. Used by /plan for the discuss-then-propose phase before any execution starts.
  write_todos itself requires approval, so the plan is presented and confirmed before it
  ever becomes part of committed state.
  """
  llm = get_llm()

  tools = [
      search_codebase,
      find_session,
      recall_memory,
      read_file,
      list_directory,
      file_exists,
  ]

  return create_agent(
      llm,
      tools=tools,
      system_prompt=PLAN_SYSTEM_PROMPT,
      middleware=[
          TodoListMiddleware(),
          HumanInTheLoopMiddleware(interrupt_on={
              "write_todos": InterruptOnConfig(allowed_decisions=["approve", "reject"]),
          }),
      ],
      checkpointer=checkpointer,
  )


async def build_agent(checkpointer):
  """Create and return the full execution agent with persistent memory. Write/dangerous
  tools require approval via HumanInTheLoopMiddleware; everything read-only or already
  scoped to cloudy's own memory store (save_memory, recall_memory) auto-approves.
  """
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
      middleware=[
          TodoListMiddleware(),
          HumanInTheLoopMiddleware(interrupt_on=_build_interrupt_on(mcp_tools)),
      ],
      checkpointer=checkpointer,
  )
