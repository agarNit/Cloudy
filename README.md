# Cloudy

Cloudy is a Claude Code–inspired, RAG-powered coding assistant built from scratch as a hands-on exploration of agentic AI: agent loops, retrieval, tool use, memory, and the Model Context Protocol (MCP).

It runs as a CLI REPL that indexes your repository, retrieves relevant code for a question, and lets an LLM agent reason over it while calling tools — search, filesystem, shell, and external MCP servers — to get things done.

## Features

- **Agentic loop** — built on LangChain/LangGraph's `create_agent`, with a system prompt that forces the agent to search the codebase before answering.
- **Code-aware indexing** — source files are parsed with [tree-sitter](https://tree-sitter.github.io/tree-sitter/) into function/class-level chunks (Python, JS/TS, Java, Go, Rust, C/C++, C#, Ruby, PHP, Swift, Kotlin, Bash); text/config files fall back to an overlapping sliding-window chunker.
- **Pluggable retrieval** — swap between Qdrant (dense/sparse/hybrid) and Chroma (semantic) purely via `config.yaml`, no code changes.
- **Tool-using agent** — `search_codebase` (RAG), filesystem tools (`read_file`, `write_file`, `append_file`, `delete_file`, `list_directory`, `file_exists`), and sandboxed shell tools (`run_command`, `run_in_directory` — blocked dangerous commands, 30s timeout).
- **MCP integration** — additional tools are loaded dynamically from external MCP servers (e.g. GitHub, filesystem) declared in `servers.json`, via `langchain-mcp-adapters`.
- **Persistent memory** — conversations are checkpointed to SQLite (`AsyncSqliteSaver`) with automatic summarization once a conversation grows past a token threshold, plus lightweight session management (start/switch/resume).
- **Observability** — structured logging across every layer (indexing, retrieval, LLM calls, tool calls), written to `.cloudy/cloudy.log` so it never clutters the REPL.
- **Config-driven** — LLM provider/model, embedding model, vector store, and retrieval mode are all controlled from a single `config.yaml`.
- **Claude-style CLI** — boxed input prompt, Markdown-rendered answers, a playful "thinking" status while the agent works, and a per-turn `elapsed time · token count` footer.

## Architecture

```
cloudy/
├── main.py                    # CLI entrypoint / REPL loop
├── config.py, config.yaml     # central runtime configuration
├── agent/
│   ├── factory.py             # builds the LangChain agent + tool list
│   ├── orchestrator.py        # entrypoint that invokes the agent per query
│   └── tools.py                # search_codebase (RAG) tool
├── tools/
│   ├── filesystem_tools.py    # read/write/append/delete/list/exists
│   └── terminal_tools.py      # sandboxed shell execution
├── mcp/
│   ├── client.py               # connects to MCP servers, collects their tools
│   └── config.py               # loads & resolves servers.json
├── context/
│   ├── indexers/               # code_parser (tree-sitter) + Qdrant/Chroma indexers
│   └── retrievers/             # matching retrievers, selected via factory
├── llm/
│   └── factory.py              # LLM + embedder construction (Anthropic, HuggingFace)
├── memory/
│   ├── session.py               # session id lifecycle
│   └── short_term.py            # SQLite checkpointing + summarization middleware
├── skills/
│   └── registry.py
├── observability/
│   └── logger.py
└── servers.json                # declares available MCP servers
```

## Requirements

- Python 3.12+
- [Poetry](https://python-poetry.org/) for dependency management
- Node.js + `npx` (only if you use the MCP servers declared in `servers.json`, which run via `npx`)
- A running [Qdrant](https://qdrant.tech/) instance (Qdrant Cloud or local via Docker) — the default config uses Qdrant in hybrid mode

## Setup

```bash
poetry install
```

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your-anthropic-api-key

QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key

# Optional — only needed if you enable the GitHub MCP server in servers.json
GITHUB_TOKEN=your-github-token

# Optional — LangSmith tracing
LANGSMITH_TRACING=
LANGSMITH_ENDPOINT=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
```

Adjust `cloudy/config.yaml` if you want to switch LLM model, embedding model, vector store provider, or retrieval mode.

## Usage

Run from the root of the repository you want Cloudy to index and chat about:

```bash
poetry run cloudy
```

or

```bash
python -m cloudy.main
```

On first run, Cloudy indexes the current directory into the configured vector store; subsequent runs reuse the existing index.

Just type a question and press enter — no command prefix needed:

```
╭──────────────────────────────────────────────────────────╮
│ ❯ how does the search_codebase tool work?
╰──────────────────────────────────────────────────────────╯
```

The answer is rendered as Markdown, followed by a `<elapsed time> · <tokens used>` footer for that turn.

### REPL commands

Anything not starting with `/` is treated as a question for the agent. Reserved commands:

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/show_index` | Inspect all indexed chunks |
| `/new_session` | Start a fresh conversation (new memory thread) |
| `/switch <session_id>` | Resume a previous session |
| `/session` | Show the current session id |
| `/exit`, `/quit` | Quit |

Logs (indexing, retrieval, tool calls) go to `.cloudy/cloudy.log`, not the terminal.

## Why this project exists

Cloudy is a learning/portfolio project — an end-to-end playground for getting hands-on with the core building blocks of agentic AI systems: retrieval-augmented generation, AST-aware code chunking, tool-calling agents, cross-session memory, and the Model Context Protocol, all wired together into a working Claude Code–style CLI.
