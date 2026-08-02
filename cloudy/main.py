import argparse
import asyncio
import random
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt


from cloudy.config import config
from cloudy.context.indexers.factory import get_indexer, get_index_inspector
from cloudy.context.freshness import seed_manifest_if_empty, sync_index
from cloudy.llm.factory import get_llm, get_embedder
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from cloudy.agent.factory import build_agent, build_plan_agent
from cloudy.memory.short_term import get_checkpointer_db_path
from cloudy.agent.orchestrator import (
   handle_query,
   resume_query,
   get_pending_approval,
   QueryResult,
   ApprovalRequest,
)
from cloudy.memory.session import new_session, switch_session, list_sessions, most_recent_session
from cloudy.memory.episodic import close_session, backfill_missing_summaries, get_summaries
from cloudy.memory.semantic import remember
from cloudy.agent import semantic_cache
from cloudy.observability.logger import get_logger


load_dotenv(Path(__file__).parent.parent / ".env")


ACCENT = "#D97757"

# How often the background freshness check runs while a session is open. Cheap
# when nothing's changed (just mtime comparisons), so this can be fairly tight.
FRESHNESS_INTERVAL_SECONDS = 20

# Cycled while the agent is working, purely for flavor — no meaning behind the words.
STATUS_WORDS = [
   "Circumnambulating", "Augmenting", "Percolating", "Marinating",
   "Ruminating", "Cogitating", "Noodling", "Wrangling", "Pontificating",
   "Combobulating", "Synthesizing", "Divining", "Excogitating", "Deliberating",
   "Spelunking", "Unspooling",
]

console = Console()
logger = get_logger(__name__)


class _InputPrompt(Prompt):
   """Plain '❯ ' prompt with no trailing ': ' — Rich adds one by default."""
   prompt_suffix = " "


COMMANDS = {
   "/plan <task>": "discuss and approve a plan before anything gets built",
   "/cancel_plan": "leave plan mode without executing anything",
   "/new_session": "start a fresh conversation",
   "/switch <id>": "resume a past session",
   "/sessions": "list past sessions for this project",
   "/session": "show the current session id",
   "/remember <text>": "save a preference to long-term memory",
   "/reindex": "manually check for and index file changes",
   "/show_index": "inspect indexed code chunks",
   "/help": "show this list",
   "/exit": "quit",
}


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(prog="cloudy", description="Cloudy — RAG-powered code assistant")
   group = parser.add_mutually_exclusive_group()
   group.add_argument(
       "-c", "--continue", dest="continue_session", action="store_true",
       help="resume the most recently active session for this project",
   )
   group.add_argument(
       "-r", "--resume", action="store_true",
       help="pick a past session for this project to resume",
   )
   return parser.parse_args()


def print_welcome(repo_path: str):
   body = (
       f"[bold {ACCENT}]Cloudy[/bold {ACCENT}] — RAG-powered code assistant\n"
       f"[dim]{repo_path}[/dim]\n\n"
       f"Ask anything about this codebase — just type and press enter.\n"
       f"[dim]/plan <task> for guided work · /help for commands · /exit to quit[/dim]"
   )
   console.print(Panel(body, border_style=ACCENT, padding=(1, 2)))


def print_help():
   console.print()
   for cmd, desc in COMMANDS.items():
       console.print(f"  [bold {ACCENT}]{cmd:<16}[/bold {ACCENT}] {desc}")
   console.print()


def print_sessions(sessions: list[dict], summaries: dict[str, str] | None = None, current: str | None = None):
   summaries = summaries or {}
   if not sessions:
       console.print(f"[dim]No past sessions for this project yet.[/dim]\n")
       return
   console.print(f"\n[bold]Past sessions[/bold] (most recent first):\n")
   for i, s in enumerate(sessions, 1):
       marker = f" [{ACCENT}](current)[/{ACCENT}]" if s["session_id"] == current else ""
       console.print(
           f"  [{ACCENT}]{i:>2}[/{ACCENT}]  {s['session_id'][:8]}  ·  last active {s['last_active_at']}{marker}"
       )
       summary = summaries.get(s["session_id"])
       if summary:
           console.print(f"       [dim]{summary}[/dim]")
   console.print()


def _format_tokens(n: int) -> str:
   return f"{n / 1000:.1f}k" if n > 1000 else str(n)


def render_plan(todos: list[dict]):
   icons = {"pending": "☐", "in_progress": "▶", "completed": "✓"}
   if not todos:
       body = "(no steps)"
   else:
       lines = [f"{icons.get(t.get('status'), '☐')} {i}. {t.get('content', '')}" for i, t in enumerate(todos, 1)]
       body = "\n".join(lines)
   console.print(Panel(body, title="Plan", border_style=ACCENT, padding=(1, 2)))
   console.print()


def render_approval(req: ApprovalRequest):
   if req.name == "write_todos":
       render_plan(req.args.get("todos", []))
       return
   lines = [f"[bold]{req.name}[/bold]"] + [f"  {k}: {v}" for k, v in req.args.items()]
   console.print(Panel("\n".join(lines), title="Approval needed", border_style="yellow", padding=(1, 2)))


def print_progress(todos: list[dict]):
   if not todos:
       return
   done = sum(1 for t in todos if t.get("status") == "completed")
   console.print(f"[dim]Progress: {done}/{len(todos)} steps done[/dim]")


def ask_decision(req: ApprovalRequest) -> dict:
   render_approval(req)
   choice = _InputPrompt.ask(
       f"[{ACCENT}]❯[/{ACCENT}] [a]approve / [r]reject / [t]talk about it",
       choices=["a", "r", "t"], default="a",
   )
   if choice == "a":
       return {"type": "approve"}
   if choice == "r":
       reason = _InputPrompt.ask(f"[{ACCENT}]❯[/{ACCENT}] Reason (optional)", default="")
       decision = {"type": "reject"}
       if reason.strip():
           decision["message"] = reason.strip()
       return decision
   message = _InputPrompt.ask(f"[{ACCENT}]❯[/{ACCENT}] What would you like to say?")
   return {"type": "reject", "message": message}


def _accumulate(total: dict, stats: dict):
   total["elapsed_seconds"] += stats.get("elapsed_seconds", 0)
   total["input_tokens"] += stats.get("input_tokens", 0)
   total["output_tokens"] += stats.get("output_tokens", 0)

   total["cache_read_tokens"] += stats.get("cache_read_tokens", 0)
   total["cache_creation_tokens"] += stats.get("cache_creation_tokens", 0)


async def drive_to_completion(
   agent, session_id: str, result: QueryResult, status_label: str = "Working", loop_through: bool = True
):
   """Resolve pending approvals. With loop_through=True (execution turns), keeps
   resolving as long as approving one action leads straight into another gate —
   appropriate mid-task, where several actions in a row are expected. With
   loop_through=False (plan-mode turns), resolves exactly one decision and
   returns control to the caller regardless of what comes back — a rejection
   in plan mode should return to the CLI, not silently chain into approving
   whatever the model proposes next.

   Returns (result, decisions_made, accumulated_stats).
   """
   decisions_made: list[tuple[ApprovalRequest, dict]] = []
   total_stats = {
       "elapsed_seconds": 0.0, "input_tokens": 0, "output_tokens": 0,
       "cache_read_tokens": 0, "cache_creation_tokens": 0,
   }
   _accumulate(total_stats, result.stats)
   all_tool_names: set[str] = set(result.tool_names)

   first_round = True
   while result.kind == "approval" and (first_round or loop_through):
       first_round = False
       decisions = []
       for req in result.approvals:
           decision = ask_decision(req)
           decisions_made.append((req, decision))
           decisions.append(decision)
       with console.status(f"[{ACCENT}]{status_label}…[/{ACCENT}]", spinner="dots"):
           result = await resume_query(agent, decisions, session_id)
       _accumulate(total_stats, result.stats)
       all_tool_names |= result.tool_names

   result.tool_names = all_tool_names
   return result, decisions_made, total_stats


async def run_turn(agent, question: str, session_id: str, status_label: str = "Working", loop_through: bool = True):
   with console.status(f"[{ACCENT}]{status_label}…[/{ACCENT}]", spinner="dots"):
       result = await handle_query(agent, question, session_id)
   return await drive_to_completion(agent, session_id, result, status_label, loop_through)


async def handle_turn(
   agent, question: str, session_id: str, status_label: str = "Working", loop_through: bool = True,
   use_cache: bool = False,
) -> dict | None:
   """Run one logical turn to completion and render the result. Returns None if
   the turn was cancelled via Ctrl+C — the turn itself is abandoned, but any
   already-completed steps stay checkpointed, so the session can be picked back
   up later via /switch or --continue.

   With loop_through=False, the turn can end still holding a fresh pending
   approval (the model reacted to a rejection by proposing again right away) —
   that's surfaced, not silently resolved, and control returns to the CLI.

   use_cache=True checks the semantic cache first and skips the agent
   invocation entirely on a hit, then stores a fresh answer afterward if the
   turn only used cache-eligible tools. Only the plain chat-mode call site
   passes this — plan mode and post-approval execution aren't simple
   repeatable Q&A, so they're never cached.
   """
   if use_cache:
       cached_answer = await semantic_cache.lookup(question)
       if cached_answer is not None:
           console.print(Markdown(cached_answer))
           console.print(f"[dim]instant · cached[/dim]\n")
           return {"plan_approved": False, "todos": [], "pending": False}

   try:
       result, decisions, stats = await run_turn(agent, question, session_id, status_label, loop_through)
   except KeyboardInterrupt:
       console.print(
           f"\n[yellow]Cancelled — progress so far is saved. Resume this session "
           f"later with /switch or --continue.[/yellow]\n"
       )
       return None

   if result.kind == "approval":
       console.print(
           f"[yellow]The agent has another action pending approval — it'll come up "
           f"again on your next input.[/yellow]\n"
       )
       return {"plan_approved": False, "todos": result.todos, "pending": True}

   if result.answer:
       console.print(Markdown(result.answer))
   print_progress(result.todos)
   total_tokens = stats["input_tokens"] + stats["output_tokens"]
   cache_read = stats.get("cache_read_tokens", 0)
   cache_note = f" · {_format_tokens(cache_read)} cached" if cache_read else ""
   console.print(f"[dim]{stats['elapsed_seconds']:.1f}s · {_format_tokens(total_tokens)} tokens{cache_note}[/dim]\n")

   if use_cache and result.answer and semantic_cache.is_cacheable_turn(result.tool_names):
       await semantic_cache.store(question, result.answer)

   plan_approved = any(
       req.name == "write_todos" and dec["type"] == "approve" for req, dec in decisions
   )
   return {"plan_approved": plan_approved, "todos": result.todos, "pending": False}


async def maybe_start_execution(outcome: dict | None, agent, session_id: str) -> str | None:
   """If this turn's outcome included an approved plan, show it and immediately
   kick off execution on the full agent. Returns the new mode ('chat') if so,
   else None — caller keeps whatever mode it already had.
   """
   if not outcome or not outcome["plan_approved"]:
       return None
   render_plan(outcome["todos"])
   console.print(f"[{ACCENT}]Plan approved — starting execution.[/{ACCENT}]\n")
   await handle_turn(agent, "Begin executing the approved plan.", session_id, random.choice(STATUS_WORDS))
   return "chat"


async def check_pending_approval(agent, session_id: str, loop_through: bool = True):
   """A session can be left with an approval gate still open — the process was
   killed, or the terminal closed, before a decision was made. Resurface it
   here instead of silently dropping it and letting the next question confuse
   things.
   """
   pending = await get_pending_approval(agent, session_id)
   if not pending:
       return
   console.print(f"[yellow]This session has a pending approval from before — let's resolve it first.[/yellow]\n")
   starting = QueryResult(kind="approval", approvals=pending)
   try:
       result, _decisions, _stats = await drive_to_completion(agent, session_id, starting, loop_through=loop_through)
   except KeyboardInterrupt:
       console.print(f"\n[yellow]Still pending — it'll be here next time you resume this session.[/yellow]\n")
       return
   if result.kind == "approval":
       console.print(f"[yellow]Another action is pending approval — it'll come up again shortly.[/yellow]\n")
       return
   if result.answer:
       console.print(Markdown(result.answer))
   print_progress(result.todos)


async def cancel_plan(plan_agent, session_id: str):
   """Actually resolve any pending write_todos interrupt (reject it) before
   dropping back to chat mode — leaving it unresolved means the underlying
   LangGraph thread is still paused waiting for a decision, and it'll silently
   re-surface the old plan the next time anything talks to this thread.
   """
   pending = await get_pending_approval(plan_agent, session_id)
   if pending:
       decisions = [{"type": "reject", "message": "Plan cancelled by the user."} for _ in pending]
       await resume_query(plan_agent, decisions, session_id)


def ask_question() -> str:
   width = max(console.size.width, 40)
   console.print(f"[{ACCENT}]╭{'─' * (width - 2)}╮[/{ACCENT}]")
   user_input = _InputPrompt.ask(f"[{ACCENT}]│[/{ACCENT}] [bold {ACCENT}]❯[/bold {ACCENT}]")
   console.print(f"[{ACCENT}]╰{'─' * (width - 2)}╯[/{ACCENT}]")
   return user_input


def get_or_create_index():
   repo_path = str(Path.cwd())
   logger.info(f"Checking index for: {repo_path}")
   with console.status(f"[{ACCENT}]Indexing {repo_path}…[/{ACCENT}]", spinner="dots"):
       index = get_indexer()(repo_path)
   return index


def _freshness_enabled() -> bool:
   # Freshness sync is Qdrant-specific (payload filters, client.delete) — only
   # wire it up when that's actually the active retrieval backend, so switching
   # config to the Chroma path doesn't crash against the wrong vector store shape.
   return config["rag"]["mode"] == "hybrid" and config["vector_store"]["provider"] == "qdrant"


def report_freshness(result: dict, quiet_if_empty: bool = True):
   total = len(result["added"]) + len(result["updated"]) + len(result["deleted"])
   if total == 0:
       if not quiet_if_empty:
           console.print(f"[dim]No changes found.[/dim]\n")
       return
   console.print(
       f"[dim]Index updated — {len(result['added'])} added, "
       f"{len(result['updated'])} updated, {len(result['deleted'])} deleted.[/dim]\n"
   )


async def freshness_loop(repo_path: str, index):
   """Runs for as long as the session is open, checking for file changes on a
   fixed interval. Cancelled cleanly on exit — see the finally block in
   _run_async.
   """
   while True:
       await asyncio.sleep(FRESHNESS_INTERVAL_SECONDS)
       try:
           result = await sync_index(repo_path, index)
           report_freshness(result)
       except asyncio.CancelledError:
           raise
       except Exception as e:
           logger.error(f"Background freshness sync failed: {e}")


async def resolve_session_id(args: argparse.Namespace) -> str:
   """Decide which session to start with, mirroring Claude Code's CLI semantics:
   default is always a brand new session; --continue/--resume are explicit opt-ins
   to reattach to prior state. Sessions are scoped to this project directory via
   the local .memory/memory.db, same as the rest of cloudy's state.
   """
   if args.continue_session:
       session_id = await most_recent_session()
       if session_id:
           console.print(f"[dim]Continuing session {session_id[:8]}[/dim]\n")
           return session_id
       console.print(f"[dim]No previous sessions found — starting fresh.[/dim]\n")
       return await new_session()

   if args.resume:
       sessions = await list_sessions()
       if not sessions:
           console.print(f"[dim]No previous sessions found — starting fresh.[/dim]\n")
           return await new_session()
       print_sessions(sessions, await get_summaries())
       choice = _InputPrompt.ask(
           f"[{ACCENT}]❯[/{ACCENT}] Pick a session number, or press enter to start a new one",
           default="",
       )
       if choice.strip().isdigit() and 1 <= int(choice) <= len(sessions):
           return await switch_session(sessions[int(choice) - 1]["session_id"])
       return await new_session()

   return await new_session()


async def initialize(checkpointer):
   """Bootstrap LLM, embedder, index, and both agents before the REPL starts."""
   with console.status(f"[{ACCENT}]Starting up…[/{ACCENT}]", spinner="dots"):
       llm = get_llm()
       embedder = get_embedder()
       index = get_or_create_index()
       agent = await build_agent(checkpointer)
       plan_agent = await build_plan_agent(checkpointer)

   console.print(
       f"[dim]{config['llm']['provider']}/{config['llm']['model']} · "
       f"{config['embeddings']['provider']}/{config['embeddings']['model']}[/dim]"
   )
   return llm, embedder, index, agent, plan_agent


async def _run_async(args: argparse.Namespace):
   logger.info("Starting Cloudy")
   repo_path = str(Path.cwd())
   print_welcome(repo_path)

   session_id = await resolve_session_id(args)

   async with AsyncSqliteSaver.from_conn_string(get_checkpointer_db_path()) as checkpointer:
       with console.status(f"[{ACCENT}]Checking session history…[/{ACCENT}]", spinner="dots"):
           await backfill_missing_summaries(checkpointer)
       llm, embedder, index, agent, plan_agent = await initialize(checkpointer)
       console.print(f"[dim]session {session_id[:8]}[/dim]\n")

       await check_pending_approval(agent, session_id)

       freshness_task = None
       if _freshness_enabled():
           with console.status(f"[{ACCENT}]Checking for file changes…[/{ACCENT}]", spinner="dots"):
               await seed_manifest_if_empty(repo_path)
               await sync_index(repo_path, index)
           freshness_task = asyncio.create_task(freshness_loop(repo_path, index))

       mode = "chat"  # "chat" | "planning"

       try:
           while True:
               if mode == "planning":
                   # A rejection in plan mode returns to the CLI after exactly one
                   # decision (see loop_through=False below) — if the model reacted
                   # by proposing again right away, that's a fresh pending approval
                   # waiting here, checked before we even show the input prompt.
                   await check_pending_approval(plan_agent, session_id, loop_through=False)

               user_input = ask_question()

               if not user_input.strip():
                   continue

               if not user_input.startswith("/"):
                   logger.info(f"Question received: {user_input}")
                   if mode == "planning":
                       outcome = await handle_turn(plan_agent, user_input, session_id, "Planning", loop_through=False)
                       new_mode = await maybe_start_execution(outcome, agent, session_id)
                       if new_mode:
                           mode = new_mode
                   else:
                       await handle_turn(agent, user_input, session_id, random.choice(STATUS_WORDS), use_cache=True)
                   continue

               command, _, arg = user_input.partition(" ")
               command = command.lower()

               if command in ("/exit", "/quit"):
                   logger.info("Shutting down")
                   console.print(f"[dim]Goodbye![/dim]")
                   break
               elif command == "/help":
                   print_help()
               elif command == "/plan":
                   if not arg.strip():
                       console.print(f"[yellow]Usage: /plan <describe what you want done>[/yellow]\n")
                   else:
                       mode = "planning"
                       console.print(
                           f"[{ACCENT}]Entering plan mode — let's discuss before anything gets built.[/{ACCENT}]\n"
                       )
                       outcome = await handle_turn(plan_agent, arg, session_id, "Planning", loop_through=False)
                       new_mode = await maybe_start_execution(outcome, agent, session_id)
                       if new_mode:
                           mode = new_mode
               elif command == "/cancel_plan":
                   if mode == "planning":
                       await cancel_plan(plan_agent, session_id)
                       mode = "chat"
                       console.print(f"[{ACCENT}]Left plan mode without executing anything.[/{ACCENT}]\n")
                   else:
                       console.print(f"[yellow]Not currently in plan mode.[/yellow]\n")
               elif command == "/new_session":
                   await close_session(checkpointer, session_id)
                   session_id = await new_session()
                   mode = "chat"
                   console.print(f"[{ACCENT}]New session started: {session_id}[/{ACCENT}]\n")
               elif command == "/switch":
                   await close_session(checkpointer, session_id)
                   session_id = await switch_session(arg.strip())
                   mode = "chat"
                   console.print(f"[{ACCENT}]Switched to session: {session_id}[/{ACCENT}]\n")
                   await check_pending_approval(agent, session_id)
               elif command == "/sessions":
                   print_sessions(await list_sessions(), await get_summaries(), current=session_id)
               elif command == "/session":
                   console.print(f"[dim]Current session: {session_id}[/dim]\n")
               elif command == "/remember":
                   if not arg.strip():
                       console.print(f"[yellow]Usage: /remember <text to save as a preference>[/yellow]\n")
                   else:
                       with console.status(f"[{ACCENT}]Saving…[/{ACCENT}]", spinner="dots"):
                           result = await remember(arg)
                       console.print(f"[{ACCENT}]{result}[/{ACCENT}]\n")
               elif command == "/reindex":
                   if not _freshness_enabled():
                       console.print(f"[yellow]Freshness sync isn't available for the current vector store config.[/yellow]\n")
                   else:
                       with console.status(f"[{ACCENT}]Checking for file changes…[/{ACCENT}]", spinner="dots"):
                           result = await sync_index(repo_path, index)
                       report_freshness(result, quiet_if_empty=False)
               elif command == "/show_index":
                   logger.info("Showing index")
                   get_index_inspector()(index)
               else:
                   logger.warning(f"Unknown command received: {user_input}")
                   console.print(f"[yellow]Unknown command: {command}[/yellow]")
                   print_help()
       finally:
           if freshness_task is not None:
               freshness_task.cancel()
               try:
                   await freshness_task
               except asyncio.CancelledError:
                   pass
           # Runs on /exit, on Ctrl+C at the prompt itself, and on any crash inside
           # the loop — the one path we can't cover this way is the process being
           # killed outright (SIGKILL, terminal force-closed); backfill_missing_summaries
           # above is what catches that case, on the next launch.
           await close_session(checkpointer, session_id)


def run():
   args = parse_args()
   asyncio.run(_run_async(args))


if __name__ == "__main__":
   run()
