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
from cloudy.llm.factory import get_llm, get_embedder
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from cloudy.agent.factory import build_agent
from cloudy.memory.short_term import get_checkpointer_db_path
from cloudy.agent.orchestrator import handle_query
from cloudy.memory.session import new_session, switch_session, list_sessions, most_recent_session
from cloudy.memory.episodic import close_session, backfill_missing_summaries, get_summaries
from cloudy.memory.semantic import remember
from cloudy.observability.logger import get_logger


load_dotenv(Path(__file__).parent.parent / ".env")


ACCENT = "#D97757"

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
   "/new_session": "start a fresh conversation",
   "/switch <id>": "resume a past session",
   "/sessions": "list past sessions for this project",
   "/session": "show the current session id",
   "/remember <text>": "save a preference to long-term memory",
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
       f"[dim]/help for commands · /exit to quit[/dim]"
   )
   console.print(Panel(body, border_style=ACCENT, padding=(1, 2)))


def print_help():
   console.print()
   for cmd, desc in COMMANDS.items():
       console.print(f"  [bold {ACCENT}]{cmd:<14}[/bold {ACCENT}] {desc}")
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
   """Bootstrap LLM, embedder, index, and agent before the REPL starts."""
   with console.status(f"[{ACCENT}]Starting up…[/{ACCENT}]", spinner="dots"):
       llm = get_llm()
       embedder = get_embedder()
       index = get_or_create_index()
       agent = await build_agent(checkpointer)

   console.print(
       f"[dim]{config['llm']['provider']}/{config['llm']['model']} · "
       f"{config['embeddings']['provider']}/{config['embeddings']['model']}[/dim]"
   )
   return llm, embedder, index, agent


async def _run_async(args: argparse.Namespace):
   logger.info("Starting Cloudy")
   repo_path = str(Path.cwd())
   print_welcome(repo_path)

   session_id = await resolve_session_id(args)

   async with AsyncSqliteSaver.from_conn_string(get_checkpointer_db_path()) as checkpointer:
       with console.status(f"[{ACCENT}]Checking session history…[/{ACCENT}]", spinner="dots"):
           await backfill_missing_summaries(checkpointer)
       llm, embedder, index, agent = await initialize(checkpointer)
       console.print(f"[dim]session {session_id[:8]}[/dim]\n")

       try:
           while True:
               user_input = ask_question()

               if not user_input.strip():
                   continue

               if not user_input.startswith("/"):
                   logger.info(f"Question received: {user_input}")
                   status_word = random.choice(STATUS_WORDS)
                   with console.status(f"[{ACCENT}]{status_word}…[/{ACCENT}]", spinner="dots"):
                       response, stats = await handle_query(agent, user_input, session_id)
                   console.print(Markdown(response))
                   total_tokens = stats["input_tokens"] + stats["output_tokens"]
                   console.print(
                       f"[dim]{stats['elapsed_seconds']:.1f}s · {total_tokens} tokens[/dim]\n"
                   )
                   continue

               command, _, arg = user_input.partition(" ")
               command = command.lower()

               if command in ("/exit", "/quit"):
                   logger.info("Shutting down")
                   console.print(f"[dim]Goodbye![/dim]")
                   break
               elif command == "/help":
                   print_help()
               elif command == "/new_session":
                   await close_session(checkpointer, session_id)
                   session_id = await new_session()
                   console.print(f"[{ACCENT}]New session started: {session_id}[/{ACCENT}]\n")
               elif command == "/switch":
                   await close_session(checkpointer, session_id)
                   session_id = await switch_session(arg.strip())
                   console.print(f"[{ACCENT}]Switched to session: {session_id}[/{ACCENT}]\n")
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
               elif command == "/show_index":
                   logger.info("Showing index")
                   get_index_inspector()(index)
               else:
                   logger.warning(f"Unknown command received: {user_input}")
                   console.print(f"[yellow]Unknown command: {command}[/yellow]")
                   print_help()
       finally:
           # Runs on /exit, on Ctrl+C, and on any crash inside the loop — the one
           # path we can't cover this way is the process being killed outright
           # (SIGKILL, terminal force-closed); backfill_missing_summaries above
           # is what catches that case, on the next launch.
           await close_session(checkpointer, session_id)


def run():
   args = parse_args()
   asyncio.run(_run_async(args))


if __name__ == "__main__":
   run()
