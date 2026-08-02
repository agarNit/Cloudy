# Cloudy

Cloudy is a Claude Code–inspired, RAG-powered coding assistant built from scratch as a hands-on exploration of agentic AI: agent loops, retrieval, tool use, memory, and the Model Context Protocol (MCP).

It runs as a CLI REPL that indexes your repository, retrieves relevant code for a question, and lets an LLM agent reason over it while calling tools — search, filesystem, shell, and external MCP servers — to get things done.

## Features

- **Agentic loop** — built on LangChain/LangGraph's `create_agent`, with a system prompt that forces the agent to search the codebase before answering.
- **Code-aware indexing** — source files are parsed with [tree-sitter](https://tree-sitter.github.io/tree-sitter/) into function/class-level chunks (Python, JS/TS, Java, Go, Rust, C/C++, C#, Ruby, PHP, Swift, Kotlin, Bash); text/config files fall back to an overlapping sliding-window chunker.
- **Pluggable retrieval** — swap between Qdrant (dense/sparse/hybrid) and Chroma (semantic) purely via `config.yaml`, no code changes.
- **Tool-using agent** — `search_codebase` (RAG), `find_session` (semantic search over past session summaries), `save_memory`/`recall_memory` (long-term facts/preferences), filesystem tools (`read_file`, `write_file`, `append_file`, `list_directory`, `file_exists`), and shell tools (`run_command`, `run_in_directory` — blocklist for a few known-dangerous commands, 30s timeout).
- **MCP integration** — additional tools are loaded dynamically from external MCP servers (e.g. GitHub, filesystem) declared in `servers.json`, via `langchain-mcp-adapters`.
- **Short-term memory** — conversations are checkpointed to SQLite (`AsyncSqliteSaver`) with automatic summarization once a conversation grows past a token threshold. Strictly thread-scoped: one session never sees another's messages.
- **Session management** — mirrors Claude Code's CLI: every launch starts a brand new session by default; `--continue`/`-c` reattaches to the most recently active one, `--resume`/`-r` lets you pick from a list — all scoped to the current project directory.
- **Episodic memory** — when a session ends (`/new_session`, `/switch`, `/exit`, or Ctrl+C), its conversation is summarized and embedded, then stored in a `session_log` table. `/sessions` shows past sessions with their real summaries instead of bare timestamps, and the agent can call `find_session` to semantically search across all of them — e.g. "did I ask about X before?" from a brand new session. Sessions that end without a clean exit (crash, killed process) are backfilled on the next launch, so summaries stay eventually-consistent regardless of how a session actually ended.
- **Long-term (semantic) memory** — separate from episodic session summaries: durable, cross-session facts/preferences/decisions, stored in a `long_term_memory` table under one of three categories (`preference`, `feedback`, `project`), upserted by `(category_type, category)` so re-saving the same topic updates it instead of duplicating. A compact summary index is always in the system prompt (progressive disclosure, same shape as skills); full detail loads on demand via `recall_memory`. Save trigger is hybrid: the agent auto-calls `save_memory` when the user corrects its approach or a firm project decision is reached — tested reliable (3/3+ across runs). Getting the model to *reliably volunteer* a save purely from natural-language "remember that..." proved unreliable with `claude-haiku-4-5` once multiple tools/instructions were competing for attention (0/5 in the full prompt despite 4/5 in isolation) — so explicit saves go through a deterministic `/remember <text>` command instead of depending on LLM judgment for that path.
- **Plan mode** — `/plan <task>` hands the conversation to a *separate*, structurally read-only agent (no write/shell tools exist for it — not just told not to use them) that discusses the task, uses `search_codebase`/`read_file`/etc. to understand the repo, and proposes a step-by-step plan via LangChain's `write_todos` tool. The plan itself requires approval before anything is committed; once approved it's rendered in a boxed panel and execution starts automatically on the full agent, in the same session, tracked against the same todo list.
- **Human-in-the-loop approvals** — built on `HumanInTheLoopMiddleware` (real LangGraph interrupts, not prompt-level hoping): `write_file`/`append_file` always require approval, `run_command`/`run_in_directory` only when the command itself looks destructive (`rm`, `kill`, `dd`, `chmod`, etc. — everyday commands like `ls`/`pytest` auto-run). Three-way decision at every gate: **approve**, **reject** (with a reason), or **talk about it** (free-form pushback — the model sees it and can resolve it or re-propose, either within the same turn or after your next message). Rejection was verified with a real, checkable side effect (not just "the command errored") to confirm it actually blocks execution, not just that the model claims it did.
- **Resumable across crashes** — approvals are LangGraph interrupts, which are checkpointed like everything else. Killing the process (or Ctrl+C, which now cancels just the current turn instead of the whole app) mid-approval leaves it durably paused; relaunching and returning to that session (`--continue`, `--resume`, or `/switch`) re-surfaces the exact same pending decision instead of silently dropping it. Verified this works even when the reattaching agent object has a different tool/middleware configuration than the one that raised the interrupt (plan agent → execution agent).
- **Index freshness** — the index is no longer build-once-and-forget. A background check (`FRESHNESS_INTERVAL_SECONDS`, default 20s) diffs the repo against a local manifest (mtime + content hash) and incrementally updates just what changed: new files indexed, modified files have their old chunks deleted and replaced, removed files have their chunks deleted — never a full rebuild. A settle window (default 8s) delays reacting to a change until the file's been untouched for a bit, approximating "this edit is finished" since there's no commit boundary to use as an explicit signal. Runs once at startup (catches anything changed while Cloudy wasn't running) and then on the background interval; `/reindex` triggers it manually. Qdrant-specific (needs a payload index on `metadata.source`, created automatically); degrades to no freshness tracking if the config points at the Chroma backend instead.
- **Prompt caching** — the system prompt and tool schemas (~50 tools including MCP) are marked as one cacheable Anthropic prompt-cache breakpoint via a `wrap_model_call` middleware, so repeated calls within the cache window reuse already-processed tokens instead of re-billing/reprocessing them. Verified against the real agent: ~9.5K tokens cached, read back in full on a later call. Below Anthropic's minimum cacheable size (empirically well above the documented 2048 figure for tool-based breakpoints — somewhere between ~4K and ~11.7K tokens) this is a harmless no-op, which is why it's applied to both agents even though the smaller read-only plan agent doesn't currently clear the threshold on its own. Cache-read tokens are shown in the turn footer when present.
- **Semantic caching** — repeated or near-duplicate questions skip the entire agent pipeline (no LLM calls at all) instead of just getting a token discount. Scoped deliberately narrow for safety: only cached when a turn used exclusively read-only, locally-trackable tools (`search_codebase`, `read_file`, etc. — an explicit allowlist, not a denylist, since the cost of under-caching is a missed speedup but the cost of over-caching is a confidently wrong answer for something that depends on untracked external state like a shell command's output or a live GitHub issue list). Invalidated automatically by reusing freshness's own change-tracking: a generation counter bumps whenever the index or long-term memory actually changes, and a cached answer is only served if both still match what they were when it was cached — no separate invalidation logic to get wrong. Matching is via embedding similarity (threshold calibrated empirically against real paraphrase pairs, not guessed — 0.85, after finding 0.93 missed a real paraphrase that scored 0.91). Verified end-to-end with a spy on the real agent's `ainvoke`: a cache hit made zero calls to the model and returned in 0.02s with the verbatim cached answer, while a genuinely unrelated question still correctly fell through to a real call.
- **Observability** — structured logging across every layer (indexing, retrieval, LLM calls, tool calls), written to `.cloudy/cloudy.log` so it never clutters the REPL.
- **Config-driven** — LLM provider/model, embedding model, vector store, and retrieval mode are all controlled from a single `config.yaml`.
- **Claude-style CLI** — boxed input prompt, Markdown-rendered answers, a playful "thinking" status while the agent works, and a per-turn `elapsed time · token count` footer.

## Architecture

```
cloudy/
├── main.py                    # CLI entrypoint / REPL loop
├── config.py, config.yaml     # central runtime configuration
├── agent/
│   ├── factory.py             # builds the execution agent + read-only plan agent, HITL gating config
│   ├── orchestrator.py        # invokes the agent per turn; detects/resumes interrupts (QueryResult)
│   ├── caching.py              # wrap_model_call middleware: cache_control on system+tools
│   ├── semantic_cache.py        # full-answer cache: purity gate, embedding match, generation-based invalidation
│   └── tools.py                # search_codebase (RAG) tool
├── tools/
│   ├── filesystem_tools.py    # read/write/append/list/exists
│   └── terminal_tools.py      # shell execution (blocklist + timeout, not a real sandbox)
├── mcp/
│   ├── client.py               # connects to MCP servers, collects their tools
│   └── config.py               # loads & resolves servers.json
├── context/
│   ├── indexers/               # code_parser (tree-sitter) + Qdrant/Chroma indexers
│   ├── retrievers/             # matching retrievers, selected via factory
│   └── freshness.py             # incremental re-index: manifest, mtime+hash+settle-window diffing
├── llm/
│   └── factory.py              # LLM + embedder construction (Anthropic, HuggingFace)
├── memory/
│   ├── session.py               # session bookkeeping (id, created_at, last_active_at)
│   ├── short_term.py            # SQLite checkpointing + summarization middleware
│   ├── episodic.py              # per-session summary + embedding on close, semantic search, backfill
│   ├── semantic.py              # long-term facts/preferences: save/recall, dedup, /remember intake
│   ├── db.py                     # shared sqlite path for cloudy's own memory tables
│   └── tools.py                  # find_session, save_memory, recall_memory tools
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
poetry run cloudy              # always starts a brand new session
poetry run cloudy --continue   # resume the most recently active session (-c)
poetry run cloudy --resume     # pick a past session to resume (-r)
```

or the equivalent with `python -m cloudy.main [--continue|--resume]`.

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
| `/plan <task>` | Discuss and approve a plan before anything gets built |
| `/cancel_plan` | Leave plan mode without executing anything |
| `/show_index` | Inspect all indexed chunks |
| `/new_session` | Start a fresh conversation (new memory thread) |
| `/switch <session_id>` | Resume a previous session |
| `/sessions` | List past sessions for this project, with their summaries |
| `/session` | Show the current session id |
| `/remember <text>` | Save a preference to long-term memory, deterministically (no LLM judgment on whether to save) |
| `/reindex` | Manually check for and index file changes (also happens automatically in the background) |
| `/exit`, `/quit` | Quit |

Logs (indexing, retrieval, tool calls) go to `.cloudy/cloudy.log`, not the terminal.

Note: the system prompt (including the long-term memory index) is built once per launch. A `/remember` mid-session is saved immediately, but won't be visible to the agent's own reasoning until the next launch rebuilds the prompt — same tradeoff the skills system already makes.

## Why this project exists

Cloudy is a learning/portfolio project — an end-to-end playground for getting hands-on with the core building blocks of agentic AI systems: retrieval-augmented generation, AST-aware code chunking, tool-calling agents, cross-session memory, and the Model Context Protocol, all wired together into a working Claude Code–style CLI.
