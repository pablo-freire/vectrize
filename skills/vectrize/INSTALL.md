# Installing vectrize

Follow this only when `vectrize` is not installed (command not found).

1. Ask the user before installing. Tell them it downloads a small binary now and an embedding model (~560 MB) on the first run. If they decline, fall back to Grep.
2. Install:
   - With Homebrew available: `brew install pablofrr/tap/vectrize`
   - Otherwise: `curl -LsSf https://vectrize.com/install | sh` (macOS Apple Silicon, Linux x86_64/arm64 with glibc 2.39+; on Windows, only inside WSL2)
3. Ask the user which folders to index (notes, docs, wikis). Don't guess.
4. Index them: `vectrize setup <first-folder>`, then `vectrize add <folder>` for each extra one.
5. Retry the original search. The first one loads the model and builds the index; if it reports that the index is being built, wait ~30 s and retry.
