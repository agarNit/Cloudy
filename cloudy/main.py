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
from cloudy.memory.session import get_current_session, new_session, switch_session
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
   "/session": "show the current session id",
   "/show_index": "inspect indexed code chunks",
   "/help": "show this list",
   "/exit": "quit",
}


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


def ask_question() -> str:
   width = max(console.size.width, 20)
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


async def initialize(checkpointer):
   """Bootstrap LLM, embedder, index, MCP tools, and session before the REPL starts."""
   with console.status(f"[{ACCENT}]Starting up…[/{ACCENT}]", spinner="dots"):
       llm = get_llm()
       embedder = get_embedder()
       index = get_or_create_index()
       agent = await build_agent(checkpointer)
       session_id = get_current_session()

   console.print(
       f"[dim]{config['llm']['provider']}/{config['llm']['model']} · "
       f"{config['embeddings']['provider']}/{config['embeddings']['model']} · "
       f"session {session_id[:8]}[/dim]\n"
   )
   return llm, embedder, index, agent, session_id


async def _run_async():
   logger.info("Starting Cloudy")
   repo_path = str(Path.cwd())
   print_welcome(repo_path)

   async with AsyncSqliteSaver.from_conn_string(get_checkpointer_db_path()) as checkpointer:
       llm, embedder, index, agent, session_id = await initialize(checkpointer)

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
               session_id = new_session()
               console.print(f"[{ACCENT}]New session started: {session_id}[/{ACCENT}]\n")
           elif command == "/switch":
               session_id = switch_session(arg.strip())
               console.print(f"[{ACCENT}]Switched to session: {session_id}[/{ACCENT}]\n")
           elif command == "/session":
               console.print(f"[dim]Current session: {session_id}[/dim]\n")
           elif command == "/show_index":
               logger.info("Showing index")
               get_index_inspector()(index)
           else:
               logger.warning(f"Unknown command received: {user_input}")
               console.print(f"[yellow]Unknown command: {command}[/yellow]")
               print_help()


def run():
   asyncio.run(_run_async())


if __name__ == "__main__":
   run()
