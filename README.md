# vectrize

Local search over folders of Markdown documents, from the CLI and for AI agents. Everything runs on your machine, on the CPU (for now).

- **Hybrid search**: embeddings ([bge-m3](https://huggingface.co/BAAI/bge-m3), multilingual) + BM25, fused with RRF.
  Finds paraphrases, not just matching words.
- **Always up to date**: a small daemon watches your folders and reindexes on save (~0.5 s), re-embedding only what changed.
- **Fast**: ~30 ms per search once the daemon is running.
- **Agent-friendly**: `--json` output with `file:line`, and a Claude Code skill.

## Install

Requires Rust (1.89+). Linux and macOS are supported.

```sh
cargo install --git https://github.com/pablo-freire/vectrize
```

Note: The first run downloads the embedding model (~560 MB) from Hugging Face.

## Usage

```sh
vectrize setup ~/notes                 # index a folder, start the daemon with your session, install the skill
vectrize add ~/work/project/docs       # add more folders
vectrize search "how do we deploy the backend"
vectrize search "ERR_TIMEOUT" --mode bm25       # exact identifiers
vectrize search "auth flow" --in ~/notes        # a single folder
vectrize status                        # folders, index and daemon state
```

```text
notes/Deploy.md:12  Backend > Production
  The backend is deployed with a blue/green switch on the load balancer…
```

`setup` is optional: without it, the first search starts the daemon.

Other commands: `remove <folder>`, `stop`, `watch` (the daemon itself). See `vectrize --help`.

## For agents

```sh
vectrize search "question" --json -k 5
```

Returns the file (absolute path), line, heading path and full text of each chunk. `setup` installs a
[Claude Code](https://claude.com/claude-code) skill that tells the agent when to use it; other agents can call the
same command.

Search always returns `k` results, relevant or not: the agent should judge relevance itself.

## How it works

- Markdown is split with [text-splitter](https://github.com/benbrandt/text-splitter) (≤256 tokens per chunk), keeping
  the heading path of each chunk as context. `.gitignore` and hidden files are respected.
- One SQLite file holds everything: [sqlite-vec](https://github.com/asg017/sqlite-vec) for vectors, FTS5 for BM25.
- The daemon keeps the model in memory (~1.1 GB) and unloads it after 30 min idle (~40 MB); the next search reloads it
  (~1.3 s).

