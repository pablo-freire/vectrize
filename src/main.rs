//! vectrize: local semantic search over a folder of documents.
//!
//! Split by headings → embeddings (bge-m3 int8) in sqlite-vec + BM25 (FTS5), fused with RRF.
//! Optional: rerank the top hits with a cross-encoder (`--rerank`).
//! `index` is incremental: only files whose hash (blake3) changed are re-embedded, and deleted files are dropped.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::{BufRead, BufReader, IsTerminal as _, Read, Write as _};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Mutex;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use fastembed::{
    InitOptionsUserDefined, Pooling, RerankInitOptionsUserDefined, TextEmbedding, TextRerank, TokenizerFiles,
    UserDefinedEmbeddingModel, UserDefinedRerankingModel,
};
use notify::Watcher as _;
use rusqlite::{params, Connection};

/// int8: same quality as fp32 in our evaluation. Unlike static embeddings, it handles paraphrases.
const EMBEDDER: &str = "onnx-community/bge-m3-ONNX";
const EMBEDDER_ONNX: &str = "onnx/model_int8.onnx";
const EMBEDDER_DIM: usize = 1024;
/// Version of the schema and the chunker: bump it and the index is rebuilt.
const SCHEMA: i64 = 3;
/// Chunks per batch when indexing. Each batch is padded to its longest chunk: with fastembed's default (256),
/// indexing a 33-note wiki peaked at 12.6 GB of RAM and took 60 s; one at a time, 1.7 GB and 26 s (2/4/8/16: slower).
const EMBED_BATCH: usize = 1;
const TEXT_EXTS: &[&str] = &["md", "mmd", "puml"];
const SKIP_DIRS: &[&str] = &[".git", ".obsidian"];
/// Max chunk size (~400 tokens). Longer sections are split by paragraphs.
const MAX_CHUNK_CHARS: usize = 1500;
const RERANKER: &str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1";
const RERANKER_ONNX: &str = "onnx/model_qint8_avx512_vnni.onnx";
/// With 10 we lose hits that BM25 ranks between 11 and 20.
const RERANK_CANDIDATES: usize = 20;
const RERANK_MAX_LENGTH: usize = 256;

#[derive(Parser)]
#[command(about = "Local semantic search over a folder of documents")]
struct Cli {
    /// Index path.
    #[arg(long, global = true, env = "VECTRIZE_DB", default_value_os_t = default_db())]
    db: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Index a folder (only what changed since last time).
    Index {
        dir: PathBuf,
        /// Replace the index even if it belongs to another folder.
        #[arg(long)]
        replace: bool,
    },
    /// First-time setup: index the folder and start the daemon with your session (systemd user service).
    Setup {
        dir: PathBuf,
        /// Replace the index even if it belongs to another folder.
        #[arg(long)]
        replace: bool,
    },
    /// Indexed folder, index size and daemon state.
    Status,
    /// Stop the daemon (the next search starts it again).
    Stop,
    /// Daemon: index, watch the folder (reindex on save) and serve `search` with the models in memory.
    Watch {
        dir: PathBuf,
        /// Minutes without searching or indexing after which the models are unloaded (~1 GB → ~40 MB; the next
        /// search takes ~1 s to reload them). 0 = always warm. The daemon started by `search` reads the env variable.
        #[arg(long, env = "VECTRIZE_UNLOAD_AFTER", default_value_t = 30.0)]
        unload_after: f64,
    },
    /// Search the index.
    Search {
        query: String,
        #[arg(short, default_value_t = 5)]
        k: usize,
        #[arg(long, value_enum, default_value_t = Mode::Hybrid)]
        mode: Mode,
        /// Rerank the top hits with a cross-encoder (+~1 s; did not beat hybrid in our evaluation).
        #[arg(long)]
        rerank: bool,
        /// JSON output for agents: absolute path, line, heading and full text of each chunk.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Mode {
    Hybrid,
    Vec,
    Bm25,
}

const DEBOUNCE: std::time::Duration = std::time::Duration::from_millis(300);
const IDLE_CHECK: std::time::Duration = std::time::Duration::from_secs(10);

/// `{root}` is replaced with the indexed folder.
const SKILL: &str = include_str!("../assets/SKILL.md");

/// Candidates each retriever contributes before fusion.
const CANDIDATES: i64 = 50;

fn default_db() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share"));
    base.join("vectrize/index.db")
}

struct Chunk {
    path: String,    // relative to the indexed folder
    heading: String, // "Title > Section > Subsection"
    text: String,
    line: usize, // where it starts in the file (1-based)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // Rust ignores SIGPIPE and `println!` panics on `vectrize search … | head`; this makes the process exit quietly.
    unsafe { signal(13 /* SIGPIPE */, 0 /* SIG_DFL */) };
    // Registers the statically linked sqlite-vec on every connection.
    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    }
    match cli.cmd {
        Cmd::Index { dir, replace } => {
            if replace {
                let _ = ask_daemon(&cli.db, "STOP"); // it was watching the old folder; the next search restarts it
            }
            index(&cli.db, &dir, replace, &mut Models::default())
        }
        Cmd::Setup { dir, replace } => setup(&cli.db, &dir, replace),
        Cmd::Status => status(&cli.db),
        Cmd::Stop => {
            match ask_daemon(&cli.db, "STOP") {
                Ok(_) => println!("daemon stopped"),
                Err(_) => println!("no daemon was running"),
            }
            Ok(())
        }
        Cmd::Watch { dir, unload_after } => watch(&cli.db, &dir, unload_after),
        Cmd::Search { query, k, mode, rerank, json } => {
            let mode_name = mode.to_possible_value().unwrap();
            let request = format!("{k}\t{}\t{rerank}\t{json}\t{query}", mode_name.get_name());
            // Any daemon failure (not running, or died mid-connection) falls back to the cold path.
            let out = match ask_daemon(&cli.db, &request) {
                Ok(out) => out,
                Err(_) => {
                    match spawn_daemon(&cli.db) {
                        Ok(()) => eprintln!("(starting the daemon; this search runs cold, the next ones warm)"),
                        Err(e) => eprintln!("(no daemon: {e:#})"),
                    }
                    search(&cli.db, &query, k, mode, rerank, json, &mut Models::default())?
                }
            };
            print!("{out}");
            Ok(())
        }
    }
}

fn tokenizer_files(repo: &hf_hub::api::sync::ApiRepo) -> Result<TokenizerFiles> {
    let read = |f: &str| -> Result<Vec<u8>> { Ok(std::fs::read(repo.get(f)?)?) };
    Ok(TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    })
}

/// Models loaded on demand and kept: the daemon loads them once and uses them for everything.
#[derive(Default)]
struct Models {
    embedder: Option<TextEmbedding>,
    reranker: Option<TextRerank>,
    last_use: Option<std::time::Instant>,
}

impl Models {
    fn embedder(&mut self) -> Result<&mut TextEmbedding> {
        self.last_use = Some(std::time::Instant::now());
        if self.embedder.is_none() {
            self.embedder = Some(load_embedder()?);
        }
        Ok(self.embedder.as_mut().unwrap())
    }

    fn reranker(&mut self) -> Result<&mut TextRerank> {
        self.last_use = Some(std::time::Instant::now());
        if self.reranker.is_none() {
            self.reranker = Some(load_reranker()?);
        }
        Ok(self.reranker.as_mut().unwrap())
    }

    fn unload_if_idle(&mut self, after: std::time::Duration) -> bool {
        let loaded = self.embedder.is_some() || self.reranker.is_some();
        if !loaded || self.last_use.is_none_or(|t| t.elapsed() < after) {
            return false;
        }
        self.embedder = None;
        self.reranker = None;
        unsafe { malloc_trim(0) }; // without it glibc keeps the memory and RSS barely drops (~600 MB)
        true
    }
}

unsafe extern "C" {
    fn malloc_trim(pad: usize) -> i32;
    fn signal(signum: i32, handler: usize) -> usize;
}

fn load_embedder() -> Result<TextEmbedding> {
    let repo = hf_hub::api::sync::Api::new()?.model(EMBEDDER.into());
    let model = UserDefinedEmbeddingModel::new(std::fs::read(repo.get(EMBEDDER_ONNX)?)?, tokenizer_files(&repo)?)
        .with_pooling(Pooling::Cls); // the one bge-m3 uses
    TextEmbedding::try_new_from_user_defined(model, InitOptionsUserDefined::new()).context("loading the model")
}

fn load_reranker() -> Result<TextRerank> {
    let repo = hf_hub::api::sync::Api::new()?.model(RERANKER.into());
    let model = UserDefinedRerankingModel::new(repo.get(RERANKER_ONNX)?, tokenizer_files(&repo)?);
    let opts = RerankInitOptionsUserDefined::new().with_max_length(RERANK_MAX_LENGTH);
    TextRerank::try_new_from_user_defined(model, opts).context("loading the reranker")
}

fn index(db: &Path, dir: &Path, replace: bool, models: &mut Models) -> Result<()> {
    let root = dir.canonicalize().with_context(|| format!("{} does not exist", dir.display()))?;
    let t = std::time::Instant::now();
    if let Some(parent) = db.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut conn = Connection::open(db)?;

    // Another folder, model or schema: what is stored is useless, start from scratch.
    let root_s = root.to_string_lossy().into_owned();
    let model = format!("{EMBEDDER}/{EMBEDDER_ONNX}");
    let stored: Option<(String, String, i64)> = conn
        .query_row("SELECT root, model, schema FROM meta", [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .ok();
    // Indexing another folder by mistake would cost the whole current index.
    if let Some((old_root, _, _)) = &stored
        && *old_root != root_s
        && !replace
    {
        anyhow::bail!("the index {} belongs to {old_root}; to replace it with {root_s}, use --replace", db.display());
    }
    // The rebuild goes in the same transaction as the inserts: while the embeddings are computed (~30 s),
    // searches keep seeing the previous index instead of an empty one.
    let rebuild = stored != Some((root_s.clone(), model.clone(), SCHEMA));
    if rebuild && stored.is_some() {
        eprintln!("rebuilding the whole index (the folder, model or format changed)");
    }
    let rebuild_sql = format!(
        "DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS vec; DROP TABLE IF EXISTS meta;
         DROP TABLE IF EXISTS fts; DROP TABLE IF EXISTS files;
         CREATE TABLE meta (root TEXT NOT NULL, model TEXT NOT NULL, schema INTEGER NOT NULL);
         CREATE TABLE files (path TEXT PRIMARY KEY, hash TEXT NOT NULL);
         CREATE TABLE chunks (id INTEGER PRIMARY KEY, path TEXT, heading TEXT, text TEXT, line INTEGER);
         CREATE INDEX chunks_path ON chunks (path);
         CREATE VIRTUAL TABLE vec USING vec0(embedding float[{EMBEDDER_DIM}] distance_metric=cosine);
         CREATE VIRTUAL TABLE fts USING fts5(heading, text, tokenize='unicode61 remove_diacritics 2');"
    );

    // Whatever is left in `known` after walking the folder was deleted from disk.
    let mut known: HashMap<String, String> = if rebuild {
        HashMap::new()
    } else {
        conn.prepare("SELECT path, hash FROM files")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?
    };
    let mut changed = Vec::new(); // (path, hash, content) of new or modified files
    let mut n_files = 0;
    let walker = walkdir::WalkDir::new(&root)
        .into_iter()
        .filter_entry(|e| !SKIP_DIRS.contains(&e.file_name().to_string_lossy().as_ref()));
    for entry in walker {
        let entry = entry?;
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !entry.file_type().is_file() || !TEXT_EXTS.contains(&ext) {
            continue;
        }
        let content = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let rel = path.strip_prefix(&root)?.to_string_lossy().into_owned();
        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        n_files += 1;
        if known.remove(&rel).as_ref() != Some(&hash) {
            changed.push((rel, hash, content));
        }
    }
    let chunks: Vec<Chunk> = changed.iter().flat_map(|(rel, _, content)| chunk_markdown(rel, content)).collect();
    let t_read = t.elapsed();

    // An edit usually touches one chunk: chunks that stay identical (heading and text) keep their vector.
    // A rebuild reuses nothing: the model may have changed.
    let mut old: HashMap<(String, String), Vec<u8>> = HashMap::new();
    if !rebuild {
        let mut stmt = conn.prepare(
            "SELECT c.heading, c.text, v.embedding FROM chunks c JOIN vec v ON v.rowid = c.id WHERE c.path = ?1",
        )?;
        for (rel, _, _) in &changed {
            for row in stmt.query_map([rel], |r| Ok(((r.get(0)?, r.get(1)?), r.get(2)?)))? {
                let (key, vector) = row?;
                old.insert(key, vector);
            }
        }
    }
    let mut vectors: Vec<Option<Vec<u8>>> =
        chunks.iter().map(|c| old.remove(&(c.heading.clone(), c.text.clone()))).collect();
    // We embed "path > headings + text": the chunk's context counts for search.
    let todo: Vec<usize> = (0..chunks.len()).filter(|&i| vectors[i].is_none()).collect();
    if !todo.is_empty() {
        let embedder = models.embedder()?;
        let show = todo.len() > 32 && std::io::stderr().is_terminal();
        for (n, part) in todo.chunks(16).enumerate() {
            let inputs: Vec<String> = part.iter().map(|&i| format!("{}\n{}", chunks[i].heading, chunks[i].text)).collect();
            for (i, e) in part.iter().zip(embedder.embed(inputs, Some(EMBED_BATCH))?) {
                vectors[*i] = Some(as_bytes(&e));
            }
            if show {
                eprint!("\r  embedding chunks: {}/{}", (n * 16 + part.len()), todo.len());
            }
        }
        if show {
            eprintln!();
        }
    }
    let t_embed = t.elapsed();

    let tx = conn.transaction()?;
    if rebuild {
        tx.execute_batch(&rebuild_sql)?;
        tx.execute("INSERT INTO meta VALUES (?1, ?2, ?3)", params![root_s, model, SCHEMA])?;
    }
    for path in known.keys().chain(changed.iter().map(|(rel, _, _)| rel)) {
        let ids: Vec<i64> = tx
            .prepare("SELECT id FROM chunks WHERE path = ?1")?
            .query_map([path], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        for id in ids {
            tx.execute("DELETE FROM vec WHERE rowid = ?1", [id])?;
            tx.execute("DELETE FROM fts WHERE rowid = ?1", [id])?;
        }
        tx.execute("DELETE FROM chunks WHERE path = ?1", [path])?;
        tx.execute("DELETE FROM files WHERE path = ?1", [path])?;
    }
    for (c, e) in chunks.iter().zip(vectors.into_iter().flatten()) {
        tx.execute(
            "INSERT INTO chunks (path, heading, text, line) VALUES (?1, ?2, ?3, ?4)",
            params![c.path, c.heading, c.text, c.line as i64],
        )?;
        let id = tx.last_insert_rowid();
        tx.execute("INSERT INTO vec (rowid, embedding) VALUES (?1, ?2)", params![id, e])?;
        tx.execute("INSERT INTO fts (rowid, heading, text) VALUES (?1, ?2, ?3)", params![id, c.heading, c.text])?;
    }
    for (rel, hash, _) in &changed {
        tx.execute("INSERT INTO files VALUES (?1, ?2)", [rel, hash])?;
    }
    tx.commit()?;

    eprintln!(
        "{n_files} files: {} new or changed ({} chunks, {} embedded), {} deleted → {}\n  read+hash {:?} · model+embeddings {:?} · sqlite {:?}",
        changed.len(),
        chunks.len(),
        todo.len(),
        known.len(),
        db.display(),
        t_read,
        t_embed - t_read,
        t.elapsed() - t_embed,
    );
    Ok(())
}

fn socket_path(db: &Path) -> PathBuf {
    db.with_extension("sock")
}

/// Output goes to `<db>.log`.
fn spawn_daemon(db: &Path) -> Result<()> {
    use std::os::unix::process::CommandExt as _;
    let root: String = Connection::open(db)?.query_row("SELECT root FROM meta", [], |r| r.get(0))?;
    let log = std::fs::OpenOptions::new().create(true).append(true).open(db.with_extension("log"))?;
    std::process::Command::new(std::env::current_exe()?)
        .arg("--db")
        .arg(db)
        .arg("watch")
        .arg(root)
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .process_group(0) // outside the terminal's process group: closing it doesn't kill the daemon
        .spawn()?;
    Ok(())
}

fn watch(db: &Path, dir: &Path, unload_after_min: f64) -> Result<()> {
    let unload_after = (unload_after_min > 0.0).then(|| std::time::Duration::from_secs_f64(unload_after_min * 60.0));
    // One daemon per index: two searches in a row without a daemon would start two. The lock is released on exit.
    let lock = std::fs::File::create(db.with_extension("lock"))?;
    if lock.try_lock().is_err() {
        eprintln!("a daemon is already running for {}", db.display());
        return Ok(());
    }
    let models = Mutex::new(Models::default());
    let root = dir.canonicalize()?;
    index(db, dir, false, &mut models.lock().unwrap())?; // whatever changed while the daemon was off
    models.lock().unwrap().embedder()?; // preloaded: the first search is already warm

    let sock = socket_path(db);
    let _ = std::fs::remove_file(&sock);
    let listener = UnixListener::bind(&sock).with_context(|| format!("creating {}", sock.display()))?;
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(tx)?;
    watcher.watch(dir, notify::RecursiveMode::Recursive)?;
    eprintln!("watching {} · searches on {} (Ctrl+C to quit)", dir.display(), sock.display());

    // One thread serves searches and another reindexes. They share the models: a search that arrives during
    // a reindex waits for it to finish (~0.3 s).
    std::thread::scope(|s| {
        s.spawn(|| {
            for stream in listener.incoming() {
                if let Err(e) = stream.map_err(Into::into).and_then(|st| serve(st, db, &models)) {
                    eprintln!("error serving a search: {e:#}");
                }
            }
        });
        // Reading files also generates events (`index` itself reads them): only writes count.
        let relevant = |ev: &notify::Result<notify::Event>| {
            ev.as_ref().is_ok_and(|ev| {
                !ev.kind.is_access()
                    && ev.paths.iter().any(|p| {
                        !p.components().any(|c| SKIP_DIRS.contains(&c.as_os_str().to_string_lossy().as_ref()))
                    })
            })
        };
        loop {
            let ev = match rx.recv_timeout(IDLE_CHECK) {
                Ok(ev) => Some(ev),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
                Err(e) => return Err(e.into()),
            };
            // An `index --replace` changed the index's folder: this daemon watches the old one and must go.
            let indexed: Option<String> = Connection::open(db)?.query_row("SELECT root FROM meta", [], |r| r.get(0)).ok();
            if indexed.is_some_and(|r| Path::new(&r) != root) {
                eprintln!("the index no longer belongs to {}; exiting", root.display());
                let _ = std::fs::remove_file(&sock);
                std::process::exit(0);
            }
            if let Some(after) = unload_after
                && models.lock().unwrap().unload_if_idle(after)
            {
                eprintln!("models unloaded after {unload_after_min} min idle");
            }
            if !ev.is_some_and(|ev| relevant(&ev)) {
                continue;
            }
            // A save arrives as a burst of events: wait for it to settle before reindexing.
            while rx.recv_timeout(DEBOUNCE).is_ok() {}
            if let Err(e) = index(db, dir, false, &mut models.lock().unwrap()) {
                eprintln!("error: {e:#}"); // e.g. a half-written file; the next save fixes it
            }
        }
    })
}

/// Protocol: one line (`STATUS`, `STOP` or `k \t mode \t rerank \t json \t query`) → the response, or `ERR message`.
fn ask_daemon(db: &Path, request: &str) -> Result<String> {
    let mut stream = UnixStream::connect(socket_path(db))?;
    writeln!(stream, "{}", request.replace('\n', " "))?;
    let mut out = String::new();
    stream.read_to_string(&mut out)?;
    match out.strip_prefix("ERR ") {
        Some(e) => anyhow::bail!("daemon: {e}"),
        None => Ok(out),
    }
}

fn rss_mb() -> Result<u64> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let kb = status.lines().find_map(|l| l.strip_prefix("VmRSS:")).context("no VmRSS")?;
    Ok(kb.trim().trim_end_matches(" kB").parse::<u64>()? / 1024)
}

fn setup(db: &Path, dir: &Path, replace: bool) -> Result<()> {
    let _ = ask_daemon(db, "STOP"); // the systemd one replaces it
    index(db, dir, replace, &mut Models::default())?;
    let root = dir.canonicalize()?;
    let home = PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?);
    let unit_dir = home.join(".config/systemd/user");
    std::fs::create_dir_all(&unit_dir)?;
    let unit = unit_dir.join("vectrize.service");
    std::fs::write(
        &unit,
        format!(
            "[Unit]\nDescription=vectrize: keeps the index of {root} up to date\n\n\
             [Service]\nExecStart=\"{exe}\" --db \"{db}\" watch \"{root}\"\nRestart=on-failure\n\n\
             [Install]\nWantedBy=default.target\n",
            root = root.display(),
            exe = std::env::current_exe()?.display(),
            db = db.display(),
        ),
    )?;
    let systemctl = |args: &[&str]| std::process::Command::new("systemctl").arg("--user").args(args).status();
    let ok = systemctl(&["daemon-reload"]).is_ok_and(|s| s.success())
        && systemctl(&["enable", "--now", "vectrize"]).is_ok_and(|s| s.success());
    println!();
    if ok {
        println!("✓ Daemon installed ({}): starts with your session and keeps the index up to date.", unit.display());
    } else {
        println!("! Could not enable the systemd service; the daemon will start with the first search.");
    }
    if home.join(".claude").is_dir() {
        let skill = home.join(".claude/skills/vectrize/SKILL.md");
        std::fs::create_dir_all(skill.parent().unwrap())?;
        std::fs::write(&skill, SKILL.replace("{root}", &root.to_string_lossy()))?;
        println!("✓ Claude Code skill installed ({}).", skill.display());
    }
    println!("\nTry:     vectrize search \"your question\"");
    println!("Status:  vectrize status");
    println!("Memory:  models are unloaded after 30 min idle (set VECTRIZE_UNLOAD_AFTER=0 in the service to keep them).");
    Ok(())
}

fn status(db: &Path) -> Result<()> {
    anyhow::ensure!(db.exists(), "no index at {}; create one with `vectrize index <folder>`", db.display());
    let conn = Connection::open(db)?;
    let root: String = conn.query_row("SELECT root FROM meta", [], |r| r.get(0))?;
    let count = |table: &str| conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get::<_, i64>(0));
    let age = std::fs::metadata(db)?.modified()?.elapsed().unwrap_or_default().as_secs();
    let age = match age {
        ..60 => format!("{age} s"),
        ..3600 => format!("{} min", age / 60),
        ..86400 => format!("{} h", age / 3600),
        _ => format!("{} d", age / 86400),
    };
    // Lock held but no socket: the daemon exists and is still bringing the index up to date.
    let starting = std::fs::File::open(db.with_extension("lock")).is_ok_and(|f| f.try_lock().is_err());
    let daemon = ask_daemon(db, "STATUS").unwrap_or_else(|_| {
        if starting { "starting (bringing the index up to date)" } else { "off (the next search starts it)" }.into()
    });
    println!("folder  {root}");
    println!("index   {} files · {} chunks · updated {age} ago · {}", count("files")?, count("chunks")?, db.display());
    println!("daemon  {daemon}");
    Ok(())
}

fn serve(mut stream: UnixStream, db: &Path, models: &Mutex<Models>) -> Result<()> {
    let mut line = String::new();
    BufReader::new(&stream).read_line(&mut line)?;
    match line.trim_end() {
        "STATUS" => {
            let loaded = models.lock().unwrap().embedder.is_some();
            let state = if loaded { "warm" } else { "unloaded (the next search reloads it, ~1.3 s)" };
            write!(stream, "{state} · {} MB of RAM · pid {}", rss_mb()?, std::process::id())?;
            return Ok(());
        }
        "STOP" => {
            stream.write_all(b"ok")?;
            let _ = std::fs::remove_file(socket_path(db));
            std::process::exit(0); // a half-done index is discarded whole: it runs in a transaction
        }
        _ => {}
    }
    let result = (|| {
        let [k, mode, rerank, json, query] = line.trim_end().splitn(5, '\t').collect::<Vec<_>>()[..] else {
            anyhow::bail!("malformed request");
        };
        let mode = Mode::from_str(mode, false).map_err(anyhow::Error::msg)?;
        search(db, query, k.parse()?, mode, rerank.parse()?, json.parse()?, &mut models.lock().unwrap())
    })();
    match result {
        Ok(out) => stream.write_all(out.as_bytes())?,
        Err(e) => write!(stream, "ERR {e:#}")?,
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn search(db: &Path, query: &str, k: usize, mode: Mode, rerank: bool, json: bool, models: &mut Models) -> Result<String> {
    let conn = Connection::open(db).with_context(|| format!("no index at {}", db.display()))?;
    // Never a silent empty result: an agent reads it as "nothing there" and gives up on the tool.
    let usable = conn
        .query_row("SELECT schema = ?1 AND EXISTS (SELECT 1 FROM chunks) FROM meta", [SCHEMA], |r| r.get(0))
        .unwrap_or(false);
    anyhow::ensure!(
        usable,
        "the index {} is empty or from an older version. If you just installed or updated vectrize, \
         it is being built (~30 s): retry in a moment (`vectrize status` shows progress). \
         Otherwise, create it with `vectrize index <folder>`.",
        db.display()
    );
    let ids = |sql: &str, arg: &dyn rusqlite::ToSql| -> Result<Vec<i64>> {
        let mut stmt = conn.prepare(sql)?;
        Ok(stmt.query_map(params![arg, CANDIDATES], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?)
    };
    let mut lists = Vec::new();
    if mode != Mode::Bm25 {
        let q = as_bytes(&models.embedder()?.embed([query], None)?[0]);
        lists.push(ids("SELECT rowid FROM vec WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance", &q)?);
    }
    if mode != Mode::Vec {
        // Each word quoted (so `¿` or `-` are not FTS5 syntax), joined with OR.
        let words: Vec<_> = query.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
        let q = words.iter().map(|w| format!("\"{w}\"")).collect::<Vec<_>>().join(" OR ");
        if !q.is_empty() {
            lists.push(ids("SELECT rowid FROM fts WHERE fts MATCH ?1 ORDER BY rank LIMIT ?2", &q)?);
        }
    }
    let mut stmt = conn.prepare("SELECT path, heading, text, line FROM chunks WHERE id = ?1")?;
    let mut chunk = |id: i64| -> Result<(String, String, String, i64)> {
        Ok(stmt.query_row([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?)
    };
    let mut hits = rrf(&lists);
    if rerank {
        hits.truncate(RERANK_CANDIDATES);
        let docs = hits
            .iter()
            .map(|&(id, _)| chunk(id).map(|(_, heading, text, _)| format!("{heading}\n{text}")))
            .collect::<Result<Vec<_>>>()?;
        let ranked = models.reranker()?.rerank(query, docs.iter().map(String::as_str).collect::<Vec<_>>(), false, None)?;
        hits = ranked.into_iter().map(|r| (hits[r.index].0, r.score as f64)).collect();
    }
    let hits = hits.into_iter().take(k).map(|(id, _)| chunk(id)).collect::<Result<Vec<_>>>()?;
    if json {
        let root: String = conn.query_row("SELECT root FROM meta", [], |r| r.get(0))?;
        let items: Vec<_> = hits
            .into_iter()
            .map(|(path, heading, text, line)| {
                let file = Path::new(&root).join(path);
                serde_json::json!({ "file": file, "line": line, "heading": heading, "text": text })
            })
            .collect();
        return Ok(serde_json::to_string_pretty(&items)? + "\n");
    }
    // `path:line` is clickable in terminals and editors.
    let mut out = String::new();
    for (path, heading, text, line) in hits {
        let section = heading.split_once(" > ").map_or("", |(_, rest)| rest);
        writeln!(out, "{path}:{line}  {section}\n  {}\n", snippet(&text, 200))?;
    }
    Ok(out)
}

/// Reciprocal Rank Fusion: each list adds 1/(60 + rank). It only uses positions, so it doesn't matter
/// that cosine and BM25 live on different scales.
fn rrf(lists: &[Vec<i64>]) -> Vec<(i64, f64)> {
    let mut scores: HashMap<i64, f64> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            *scores.entry(*id).or_default() += 1.0 / (61.0 + rank as f64);
        }
    }
    let mut out: Vec<_> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// sqlite-vec expects the f32s as contiguous little-endian bytes.
fn as_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn snippet(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", &flat[..i]), // cut by char, not byte: UTF-8
        None => flat,
    }
}

/// Splits a document by `#` headings (or titles between `=====` lines), with a size cap.
/// Each chunk carries the path of headings that contain it. `#` inside ``` blocks don't count
/// (bash comments, etc.) and ```compressed-json blocks (Excalidraw drawings) are dropped.
fn chunk_markdown(path: &str, content: &str) -> Vec<Chunk> {
    let stem = Path::new(path).file_stem().map_or(path.into(), |s| s.to_string_lossy());
    let mut stack: Vec<(usize, String)> = Vec::new(); // (level, title)
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut in_fence = false;

    let heading_of = |stack: &[(usize, String)]| {
        std::iter::once(stem.to_string())
            .chain(stack.iter().map(|(_, t)| t.clone()))
            .collect::<Vec<_>>()
            .join(" > ")
    };
    let flush = |buf: &mut String, heading: String, out: &mut Vec<Chunk>| {
        for text in split_long(buf.trim(), MAX_CHUNK_CHARS) {
            out.push(Chunk { path: path.into(), heading: heading.clone(), text, line: 0 });
        }
        buf.clear();
    };
    let is_rule = |l: &str| l.len() >= 10 && l.bytes().all(|b| b == b'=');

    let lines: Vec<&str> = content.lines().collect();
    let mut skip = false; // inside a ```compressed-json block
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        i += 1;
        if line.trim_start().starts_with("```") {
            if line.trim_start().starts_with("```compressed-json") {
                skip = true;
                continue;
            }
            if skip {
                skip = false;
                continue;
            }
            in_fence = !in_fence;
        }
        if skip {
            continue;
        }
        // `=====` / TITLE / `=====` counts as a level-1 heading.
        let rule_title = (!in_fence && is_rule(line) && i + 1 < lines.len() && is_rule(lines[i + 1]))
            .then(|| lines[i].trim())
            .filter(|t| !t.is_empty());
        let level = line.bytes().take_while(|&b| b == b'#').count();
        let md_title = (!in_fence && (1..=6).contains(&level) && line[level..].starts_with(' '))
            .then(|| line[level..].trim());
        // A line that is just `**text**` acts as a subtitle (common in Obsidian). Level 7: below any `#`,
        // and the next bold line replaces it.
        let bold_title = line
            .trim()
            .strip_prefix("**")
            .and_then(|t| t.strip_suffix("**"))
            .filter(|t| !in_fence && !t.is_empty() && !t.contains("**"));
        let title = rule_title.map(|t| (1, t)).or(md_title.map(|t| (level, t))).or(bold_title.map(|t| (7, t)));
        // Each task (`- [ ]`, `- [x]`) is its own chunk: a to-do list groups unrelated things, and in a single
        // chunk each one gets diluted (a one-line task in a 24-line note was not found).
        let is_task = !in_fence && ["- [ ] ", "- [x] ", "- [X] "].iter().any(|p| line.starts_with(p));
        if is_task {
            flush(&mut buf, heading_of(&stack), &mut out);
        }
        if let Some((level, title)) = title {
            flush(&mut buf, heading_of(&stack), &mut out);
            stack.retain(|(l, _)| *l < level);
            stack.push((level, title.to_string()));
            if rule_title.is_some() {
                i += 2; // skip the title and the second `=` line
            }
        } else {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    flush(&mut buf, heading_of(&stack), &mut out);
    // Line where each chunk starts: that of its first text line, searching forward from the previous chunk.
    let mut from = 0;
    for c in &mut out {
        let first = c.text.lines().next().unwrap_or("").trim();
        if let Some(i) = lines[from..].iter().position(|l| l.contains(first)) {
            from += i;
        }
        c.line = from + 1;
    }
    out
}

/// Splits a text into chunks of at most `max` bytes: by paragraphs, and if a paragraph doesn't fit, by
/// lines; a line that doesn't fit either is cut by force. Small paragraphs are merged.
fn split_long(text: &str, max: usize) -> Vec<String> {
    let mut units = Vec::new();
    for para in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if para.len() <= max {
            units.push(para);
            continue;
        }
        for mut line in para.lines() {
            while line.len() > max {
                let cut = line.floor_char_boundary(max);
                units.push(&line[..cut]);
                line = &line[cut..];
            }
            units.push(line);
        }
    }
    let mut out: Vec<String> = Vec::new();
    for u in units {
        match out.last_mut() {
            Some(cur) if cur.len() + 2 + u.len() <= max => {
                cur.push_str("\n\n");
                cur.push_str(u);
            }
            _ => out.push(u.to_string()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_by_heading_with_path_and_ignores_fences() {
        let md = "intro\n# A\ntext a\n## B\n```sh\n# not a heading\n```\n# C\ntext c\n";
        let c = chunk_markdown("dir/Doc.md", md);
        let got: Vec<_> = c.iter().map(|c| (c.heading.as_str(), c.text.as_str())).collect();
        assert_eq!(
            got,
            [
                ("Doc", "intro"),
                ("Doc > A", "text a"),
                ("Doc > A > B", "```sh\n# not a heading\n```"),
                ("Doc > C", "text c"),
            ]
        );
        assert_eq!(c.iter().map(|c| c.line).collect::<Vec<_>>(), [1, 3, 5, 9]);
    }

    #[test]
    fn one_chunk_per_task_under_bold_titles() {
        let md = "**Android**\n\n- [ ] DB indexes\n- [x] Fix bug\n  bug details\n**Web**\n- [ ] Tutorials\n";
        let c = chunk_markdown("Improvements.md", md);
        let got: Vec<_> = c.iter().map(|c| (c.heading.as_str(), c.text.as_str(), c.line)).collect();
        assert_eq!(
            got,
            [
                ("Improvements > Android", "- [ ] DB indexes", 3),
                ("Improvements > Android", "- [x] Fix bug\n  bug details", 4),
                ("Improvements > Web", "- [ ] Tutorials", 7),
            ]
        );
    }

    #[test]
    fn rule_headings_drop_excalidraw_and_size_cap() {
        let big = "word ".repeat(MAX_CHUNK_CHARS / 5 * 3); // ~3 times the cap, in a single paragraph
        let md = format!(
            "=====================\nSUMMARY\n=====================\nhello\n\n{big}\n```compressed-json\nGARBAGE\n```\nend\n"
        );
        let c = chunk_markdown("Doc.md", &md);
        assert!(c.iter().all(|c| c.heading == "Doc > SUMMARY" && c.text.len() <= MAX_CHUNK_CHARS));
        assert!(c.len() >= 3);
        let all: String = c.iter().map(|c| c.text.as_str()).collect();
        assert!(all.contains("hello") && all.contains("end") && !all.contains("GARBAGE") && !all.contains("===="));
    }

    #[test]
    fn rrf_rewards_agreement_between_lists() {
        // 2 is second in both lists; 1 and 3 are first in only one.
        let fused: Vec<i64> = rrf(&[vec![1, 2], vec![3, 2]]).into_iter().map(|(id, _)| id).collect();
        assert_eq!(fused, [2, 1, 3]);
    }
}
