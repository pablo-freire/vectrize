//! Local semantic search: Markdown chunks embedded (bge-m3) in sqlite-vec + BM25 (FTS5), fused with RRF.

mod chunk;

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{BufRead, BufReader, IsTerminal as _, Read, Write as _};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use fastembed::{
    InitOptionsUserDefined, Pooling, RerankInitOptionsUserDefined, TextEmbedding, TextRerank, TokenizerFiles,
    UserDefinedEmbeddingModel, UserDefinedRerankingModel,
};
use notify::Watcher as _;
use rusqlite::{Connection, ToSql, params};
use serde::{Deserialize, Serialize};

use chunk::{Chunk, chunk_markdown};
use text_splitter::{ChunkConfig, MarkdownSplitter};

const EMBEDDER: &str = "onnx-community/bge-m3-ONNX";
const EMBEDDER_ONNX: &str = "onnx/model_int8.onnx";
const EMBEDDER_DIM: usize = 1024;
/// Version of the schema and the chunker: if it changes, the index is rebuilt.
const SCHEMA: i64 = 5;
/// Max tokens per chunk (the model truncates at 512).
const CHUNK_TOKENS: usize = 256;
/// A batch is padded to its longest chunk, so larger batches cost a lot of RAM and are not faster on CPU.
const EMBED_BATCH: usize = 1;
const TEXT_EXTS: &[&str] = &["md", "mmd", "puml"];
const RERANKER: &str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1";
const RERANKER_ONNX: &str = "onnx/model_qint8_avx512_vnni.onnx";
const RERANK_CANDIDATES: usize = 20;
const RERANK_MAX_LENGTH: usize = 256;
const CANDIDATES: i64 = 50;
const DEBOUNCE: Duration = Duration::from_millis(300);
const IDLE_CHECK: Duration = Duration::from_secs(10);
/// Version of the daemon protocol: one line (`STATUS`, `STOP` or `<version> <JSON query>`) → the response, or
/// `ERR message`. A daemon from another version (of the protocol or the schema) answers `ERR STALE` and exits, so
/// after an upgrade the next search starts a new one.
const PROTOCOL: u32 = 1;

#[derive(Parser)]
#[command(version, about = "Local semantic search over folders of documents")]
struct Cli {
    /// Index path.
    #[arg(long, global = true, env = "VECTRIZE_DB", default_value_os_t = default_db())]
    db: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Add a folder to the index, or bring it up to date (only what changed is re-embedded).
    Add { dir: PathBuf },
    /// Remove a folder from the index.
    Remove { dir: PathBuf },
    /// Search every indexed folder.
    Search {
        #[command(flatten)]
        query: Query,
        /// JSON output for agents: absolute path, line, heading and full text of each chunk.
        #[arg(long)]
        json: bool,
    },
    /// Indexed folders and daemon state.
    Status,
    /// First-time setup: add the folder and start the daemon with your session (systemd/launchd).
    Setup { dir: PathBuf },
    /// Stop the daemon (the next search starts it again).
    Stop,
    /// Daemon: watch every indexed folder (reindex on save) and serve searches with the models in memory.
    Watch {
        /// Minutes idle after which the models are unloaded (0 = never).
        #[arg(long, env = "VECTRIZE_UNLOAD_AFTER", default_value_t = 30.0)]
        unload_after: f64,
    },
}

#[derive(clap::Args, Serialize, Deserialize)]
struct Query {
    text: String,
    #[arg(short, default_value_t = 5)]
    k: usize,
    #[arg(long, value_enum, default_value_t = Mode::Hybrid)]
    mode: Mode,
    /// Rerank the top hits with a cross-encoder (slower).
    #[arg(long)]
    rerank: bool,
    /// Only search this folder.
    #[arg(long = "in", value_name = "FOLDER")]
    scope: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq, ValueEnum, Serialize, Deserialize)]
enum Mode {
    Hybrid,
    Vec,
    Bm25,
}

#[derive(Serialize, Deserialize)]
struct Hit {
    file: PathBuf,
    line: i64,
    heading: String,
    text: String,
}

struct File {
    root: i64,
    path: String, // relative to its root
    hash: String,
    mtime: i64,
    size: i64,
    content: String,
}

#[derive(Default)]
struct Scan {
    changed: Vec<File>,
    touched: Vec<File>,          // same content, new mtime
    deleted: Vec<(i64, String)>, // (root, path)
}

fn default_db() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share"));
    base.join("vectrize/index.db")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // Otherwise `println!` panics on `vectrize search … | head`.
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };
    // sqlite-vec is statically linked; this registers it on every connection we open.
    unsafe {
        use rusqlite::ffi::{sqlite3, sqlite3_api_routines};
        type Init = unsafe extern "C" fn(*mut sqlite3, *mut *mut std::ffi::c_char, *const sqlite3_api_routines) -> i32;
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<*const (), Init>(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    }
    let db = &cli.db;
    match cli.cmd {
        Cmd::Add { dir } => sync(db, Some(&dir), &mut Models::default()),
        Cmd::Remove { dir } => remove(db, &dir),
        Cmd::Search { mut query, json } => {
            if let Some(dir) = &query.scope {
                query.scope = Some(dir.canonicalize().with_context(|| format!("{} does not exist", dir.display()))?);
            }
            let hits = match ask_daemon(db, &format!("{} {}", version(), serde_json::to_string(&query)?)) {
                Ok(out) => serde_json::from_str(&out)?,
                // No daemon or an outdated one: search cold and start one for the next searches.
                Err(e) if e.is::<std::io::Error>() || e.to_string() == "STALE" => {
                    match spawn_daemon(db) {
                        Ok(()) => eprintln!("(starting the daemon; this search runs cold, the next ones warm)"),
                        Err(e) => eprintln!("(no daemon: {e:#})"),
                    }
                    search(db, &query, &mut Models::default())?
                }
                Err(e) => return Err(e),
            };
            print!("{}", render(&hits, json)?);
            Ok(())
        }
        Cmd::Status => status(db),
        Cmd::Setup { dir } => setup(db, &dir),
        Cmd::Stop => {
            match ask_daemon(db, "STOP") {
                Ok(_) => println!("daemon stopped"),
                Err(_) => println!("no daemon was running"),
            }
            Ok(())
        }
        Cmd::Watch { unload_after } => watch(db, unload_after),
    }
}

/// Hyperthreads share vector units: more threads than physical cores only burns CPU.
fn threads() -> usize {
    num_cpus::get_physical()
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

#[derive(Default)]
struct Models {
    embedder: Option<TextEmbedding>,
    splitter: Option<MarkdownSplitter<tokenizers::Tokenizer>>,
    reranker: Option<TextRerank>,
    last_use: Option<Instant>,
}

impl Models {
    fn embedder(&mut self) -> Result<&mut TextEmbedding> {
        self.last_use = Some(Instant::now());
        if self.embedder.is_none() {
            let repo = hf_hub::api::sync::Api::new()?.model(EMBEDDER.into());
            let onnx = std::fs::read(repo.get(EMBEDDER_ONNX)?)?;
            let model = UserDefinedEmbeddingModel::new(onnx, tokenizer_files(&repo)?).with_pooling(Pooling::Cls);
            let options = InitOptionsUserDefined::new().with_intra_threads(threads());
            self.embedder =
                Some(TextEmbedding::try_new_from_user_defined(model, options).context("loading the model")?);
        }
        Ok(self.embedder.as_mut().unwrap())
    }

    /// Cached: cloning the tokenizer is slow.
    fn splitter(&mut self) -> Result<&MarkdownSplitter<tokenizers::Tokenizer>> {
        if self.splitter.is_none() {
            // Without truncation, or every count would be capped at 512.
            let mut tokenizer = self.embedder()?.tokenizer.clone();
            tokenizer.with_truncation(None).map_err(anyhow::Error::msg)?.with_padding(None);
            self.splitter = Some(MarkdownSplitter::new(ChunkConfig::new(CHUNK_TOKENS).with_sizer(tokenizer)));
        }
        Ok(self.splitter.as_ref().unwrap())
    }

    fn reranker(&mut self) -> Result<&mut TextRerank> {
        self.last_use = Some(Instant::now());
        if self.reranker.is_none() {
            let repo = hf_hub::api::sync::Api::new()?.model(RERANKER.into());
            let model = UserDefinedRerankingModel::new(repo.get(RERANKER_ONNX)?, tokenizer_files(&repo)?);
            let options =
                RerankInitOptionsUserDefined::new().with_max_length(RERANK_MAX_LENGTH).with_intra_threads(threads());
            self.reranker =
                Some(TextRerank::try_new_from_user_defined(model, options).context("loading the reranker")?);
        }
        Ok(self.reranker.as_mut().unwrap())
    }

    fn unload_if_idle(&mut self, after: Duration) -> bool {
        let loaded = self.embedder.is_some() || self.reranker.is_some();
        if !loaded || self.last_use.is_none_or(|t| t.elapsed() < after) {
            return false;
        }
        self.embedder = None;
        self.splitter = None;
        self.reranker = None;
        release_memory();
        true
    }
}

/// Without this, freed model memory stays in the process.
fn release_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
    #[cfg(target_os = "macos")]
    unsafe {
        malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    /// Not in the `libc` crate.
    fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
}

fn schema_sql() -> String {
    format!(
        "DROP TABLE IF EXISTS meta; DROP TABLE IF EXISTS roots; DROP TABLE IF EXISTS files;
         DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS vec; DROP TABLE IF EXISTS fts;
         CREATE TABLE meta (model TEXT NOT NULL, schema INTEGER NOT NULL);
         CREATE TABLE roots (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);
         CREATE TABLE files (root INTEGER NOT NULL, path TEXT NOT NULL, hash TEXT NOT NULL,
                             mtime INTEGER NOT NULL, size INTEGER NOT NULL, PRIMARY KEY (root, path));
         CREATE TABLE chunks (id INTEGER PRIMARY KEY, root INTEGER NOT NULL, path TEXT NOT NULL,
                              heading TEXT NOT NULL, text TEXT NOT NULL, line INTEGER NOT NULL);
         CREATE INDEX chunks_file ON chunks (root, path);
         CREATE VIRTUAL TABLE vec USING vec0(root integer partition key,
                                             embedding float[{EMBEDDER_DIM}] distance_metric=cosine);
         CREATE VIRTUAL TABLE fts USING fts5(heading, text, root UNINDEXED,
                                             tokenize='unicode61 remove_diacritics 2');"
    )
}

fn roots(conn: &Connection) -> Vec<(i64, PathBuf)> {
    conn.prepare("SELECT id, path FROM roots ORDER BY id")
        .and_then(|mut stmt| stmt.query_map([], |r| Ok((r.get(0)?, PathBuf::from(r.get::<_, String>(1)?))))?.collect())
        .unwrap_or_default()
}

fn sync(db: &Path, add: Option<&Path>, models: &mut Models) -> Result<()> {
    let t = Instant::now();
    if let Some(parent) = db.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut conn = Connection::open(db)?;
    let model = format!("{EMBEDDER}/{EMBEDDER_ONNX}");
    let stored: Option<(String, i64)> =
        conn.query_row("SELECT model, schema FROM meta", [], |r| Ok((r.get(0)?, r.get(1)?))).ok();
    // The rebuild shares the inserts' transaction, so searches see the old index until it commits. An older binary
    // (e.g. a daemon left running after an upgrade) must not rebuild a newer index.
    anyhow::ensure!(stored.as_ref().is_none_or(|(_, v)| *v <= SCHEMA), "the index was built by a newer vectrize");
    let rebuild = stored != Some((model.clone(), SCHEMA));
    if rebuild && stored.is_some() {
        eprintln!("rebuilding the whole index (new model or format)");
    }
    let mut roots = roots(&conn);
    if let Some(dir) = add {
        let dir = dir.canonicalize().with_context(|| format!("{} does not exist", dir.display()))?;
        anyhow::ensure!(dir.is_dir(), "{} is not a folder; add the folder that contains it", dir.display());
        if !roots.iter().any(|(_, r)| *r == dir) {
            roots.push((roots.iter().map(|r| r.0).max().unwrap_or(0) + 1, dir));
        }
    }

    let mut scan = Scan::default();
    for (id, root) in &roots {
        scan_root(&conn, *id, root, rebuild, &mut scan)?;
    }
    let Scan { changed, touched, deleted } = scan;
    let chunks: Vec<(&File, Chunk)> = if changed.is_empty() {
        Vec::new()
    } else {
        let splitter = models.splitter()?;
        changed
            .iter()
            .flat_map(|f| chunk_markdown(&f.path, &f.content, splitter).into_iter().map(move |c| (f, c)))
            .collect()
    };
    let t_read = t.elapsed();
    let (vectors, embedded) = embed(&conn, &chunks, rebuild, models)?;
    let t_embed = t.elapsed();

    let tx = conn.transaction()?;
    if rebuild {
        tx.execute_batch(&schema_sql())?;
        tx.execute("INSERT INTO meta VALUES (?1, ?2)", params![model, SCHEMA])?;
    }
    for (id, root) in &roots {
        tx.execute("INSERT OR IGNORE INTO roots VALUES (?1, ?2)", params![id, root.to_string_lossy()])?;
    }
    let gone = deleted.iter().map(|(root, path)| (*root, path.as_str()));
    for (root, path) in gone.chain(changed.iter().map(|f| (f.root, f.path.as_str()))) {
        delete_file(&tx, root, path)?;
    }
    for ((f, c), vector) in chunks.iter().zip(vectors) {
        tx.execute(
            "INSERT INTO chunks (root, path, heading, text, line) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![f.root, f.path, c.heading, c.text, i64::try_from(c.line)?],
        )?;
        let id = tx.last_insert_rowid();
        tx.execute("INSERT INTO vec (rowid, root, embedding) VALUES (?1, ?2, ?3)", params![id, f.root, vector])?;
        tx.execute(
            "INSERT INTO fts (rowid, heading, text, root) VALUES (?1, ?2, ?3, ?4)",
            params![id, c.heading, c.text, f.root],
        )?;
    }
    for f in &changed {
        tx.execute("INSERT INTO files VALUES (?1, ?2, ?3, ?4, ?5)", params![f.root, f.path, f.hash, f.mtime, f.size])?;
    }
    for f in &touched {
        tx.execute(
            "UPDATE files SET mtime = ?3, size = ?4 WHERE root = ?1 AND path = ?2",
            params![f.root, f.path, f.mtime, f.size],
        )?;
    }
    tx.commit()?;

    eprintln!(
        "{} folders: {} files new or changed ({} chunks, {embedded} embedded), {} deleted → {}\n  \
         read+hash {t_read:?} · model+embeddings {:?} · sqlite {:?}",
        roots.len(),
        changed.len(),
        chunks.len(),
        deleted.len(),
        db.display(),
        t_embed.saturating_sub(t_read),
        t.elapsed().saturating_sub(t_embed),
    );
    Ok(())
}

fn scan_root(conn: &Connection, id: i64, root: &Path, rebuild: bool, scan: &mut Scan) -> Result<()> {
    if !root.is_dir() {
        eprintln!("warning: {} is missing; keeping what was indexed (`vectrize remove` drops it)", root.display());
        return Ok(());
    }
    let mut known: HashMap<String, (String, i64, i64)> = if rebuild {
        HashMap::new()
    } else {
        conn.prepare("SELECT path, hash, mtime, size FROM files WHERE root = ?1")?
            .query_map([id], |r| Ok((r.get(0)?, (r.get(1)?, r.get(2)?, r.get(3)?))))?
            .collect::<rusqlite::Result<_>>()?
    };
    // Skips gitignored and hidden files.
    for entry in ignore::WalkBuilder::new(root).require_git(false).build() {
        let entry = entry?;
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !entry.file_type().is_some_and(|t| t.is_file()) || !TEXT_EXTS.contains(&ext) {
            continue;
        }
        let meta = entry.metadata()?;
        let mtime = meta.modified()?.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let (mtime, size) = (i64::try_from(mtime)?, i64::try_from(meta.len())?);
        let rel = path.strip_prefix(root)?.to_string_lossy().into_owned();
        let old = known.remove(&rel);
        if old.as_ref().is_some_and(|(_, m, s)| (*m, *s) == (mtime, size)) {
            continue;
        }
        let content = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        let same = old.is_some_and(|(h, _, _)| h == hash);
        let file = File { root: id, path: rel, hash, mtime, size, content: if same { String::new() } else { content } };
        if same { scan.touched.push(file) } else { scan.changed.push(file) }
    }
    scan.deleted.extend(known.into_keys().map(|p| (id, p)));
    Ok(())
}

/// One vector per chunk, and how many were computed. Unchanged chunks keep their vector, except on a rebuild.
fn embed(
    conn: &Connection,
    chunks: &[(&File, Chunk)],
    rebuild: bool,
    models: &mut Models,
) -> Result<(Vec<Vec<u8>>, usize)> {
    let mut old: HashMap<(i64, String, String), Vec<u8>> = HashMap::new();
    if !rebuild {
        let mut stmt = conn.prepare(
            "SELECT c.heading, c.text, v.embedding FROM chunks c JOIN vec v ON v.rowid = c.id
             WHERE c.root = ?1 AND c.path = ?2",
        )?;
        let files: HashSet<(i64, &str)> = chunks.iter().map(|(f, _)| (f.root, f.path.as_str())).collect();
        for (root, path) in files {
            for row in stmt.query_map(params![root, path], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))? {
                let (heading, text, vector) = row?;
                old.insert((root, heading, text), vector);
            }
        }
    }
    let mut vectors: Vec<Option<Vec<u8>>> =
        chunks.iter().map(|(f, c)| old.remove(&(f.root, c.heading.clone(), c.text.clone()))).collect();
    let todo: Vec<usize> = (0..chunks.len()).filter(|&i| vectors[i].is_none()).collect();
    if !todo.is_empty() {
        let embedder = models.embedder()?;
        let show = todo.len() > 32 && std::io::stderr().is_terminal();
        for (n, part) in todo.chunks(16).enumerate() {
            let inputs: Vec<String> =
                part.iter().map(|&i| format!("{}\n{}", chunks[i].1.heading, chunks[i].1.text)).collect();
            for (i, e) in part.iter().zip(embedder.embed(inputs, Some(EMBED_BATCH))?) {
                vectors[*i] = Some(as_bytes(&e));
            }
            if show {
                eprint!("\r  embedding chunks: {}/{}", n * 16 + part.len(), todo.len());
            }
        }
        if show {
            eprintln!();
        }
    }
    Ok((vectors.into_iter().flatten().collect(), todo.len()))
}

fn delete_file(tx: &Connection, root: i64, path: &str) -> Result<()> {
    let ids: Vec<i64> = tx
        .prepare("SELECT id FROM chunks WHERE root = ?1 AND path = ?2")?
        .query_map(params![root, path], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for id in ids {
        tx.execute("DELETE FROM vec WHERE rowid = ?1", [id])?;
        tx.execute("DELETE FROM fts WHERE rowid = ?1", [id])?;
    }
    tx.execute("DELETE FROM chunks WHERE root = ?1 AND path = ?2", params![root, path])?;
    tx.execute("DELETE FROM files WHERE root = ?1 AND path = ?2", params![root, path])?;
    Ok(())
}

fn remove(db: &Path, dir: &Path) -> Result<()> {
    let dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()); // may be gone from disk
    let mut conn = Connection::open(db)?;
    let (id, _) = roots(&conn).into_iter().find(|(_, r)| *r == dir).with_context(|| not_indexed(&dir))?;
    let tx = conn.transaction()?;
    let paths: Vec<String> = tx
        .prepare("SELECT path FROM files WHERE root = ?1")?
        .query_map([id], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for path in &paths {
        delete_file(&tx, id, path)?;
    }
    tx.execute("DELETE FROM roots WHERE id = ?1", [id])?;
    tx.commit()?;
    println!("removed {} ({} files)", dir.display(), paths.len());
    Ok(())
}

fn not_indexed(dir: &Path) -> String {
    format!("{} is not an indexed folder (see `vectrize status`)", dir.display())
}

fn search(db: &Path, query: &Query, models: &mut Models) -> Result<Vec<Hit>> {
    let conn = Connection::open(db).with_context(|| format!("no index at {}", db.display()))?;
    // An empty result would read as "nothing there": fail loudly instead.
    let usable = conn
        .query_row("SELECT schema = ?1 AND EXISTS (SELECT 1 FROM chunks) FROM meta", [SCHEMA], |r| r.get(0))
        .unwrap_or(false);
    anyhow::ensure!(
        usable,
        "the index {} is empty or from an older version. If you just installed or updated vectrize, \
         it is being built (~30 s): retry in a moment (`vectrize status` shows progress). \
         Otherwise, add a folder with `vectrize add <folder>`.",
        db.display()
    );
    let scope = (query.scope.as_ref())
        .map(|dir| roots(&conn).into_iter().find(|(_, r)| r == dir).map(|(id, _)| id).with_context(|| not_indexed(dir)))
        .transpose()?;
    let filter = if scope.is_some() { " AND root = ?3" } else { "" };
    let ids = |sql: &str, arg: &dyn ToSql| -> Result<Vec<i64>> {
        let mut args: Vec<&dyn ToSql> = vec![arg, &CANDIDATES];
        if let Some(root) = &scope {
            args.push(root);
        }
        let mut stmt = conn.prepare(sql)?;
        Ok(stmt.query_map(&*args, |r| r.get(0))?.collect::<rusqlite::Result<_>>()?)
    };
    let mut lists = Vec::new();
    if query.mode != Mode::Bm25 {
        let q = as_bytes(&models.embedder()?.embed([&query.text], None)?[0]);
        let sql = format!("SELECT rowid FROM vec WHERE embedding MATCH ?1 AND k = ?2{filter} ORDER BY distance");
        lists.push(ids(&sql, &q)?);
    }
    if query.mode != Mode::Vec {
        // Quoted so punctuation is not FTS5 syntax.
        let words: Vec<_> = query.text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
        let q = words.iter().map(|w| format!("\"{w}\"")).collect::<Vec<_>>().join(" OR ");
        if !q.is_empty() {
            let sql = format!("SELECT rowid FROM fts WHERE fts MATCH ?1{filter} ORDER BY rank LIMIT ?2");
            lists.push(ids(&sql, &q)?);
        }
    }
    let mut stmt = conn.prepare(
        "SELECT r.path, c.path, c.heading, c.text, c.line FROM chunks c JOIN roots r ON r.id = c.root WHERE c.id = ?1",
    )?;
    let mut hit = |id: i64| -> Result<Hit> {
        Ok(stmt.query_row([id], |r| {
            Ok(Hit {
                file: Path::new(&r.get::<_, String>(0)?).join(r.get::<_, String>(1)?),
                heading: r.get(2)?,
                text: r.get(3)?,
                line: r.get(4)?,
            })
        })?)
    };
    let mut ranked = rrf(&lists);
    if query.rerank {
        ranked.truncate(RERANK_CANDIDATES);
        let hits = ranked.iter().map(|&(id, _)| hit(id)).collect::<Result<Vec<_>>>()?;
        let docs: Vec<String> = hits.iter().map(|h| format!("{}\n{}", h.heading, h.text)).collect();
        let order = models.reranker()?.rerank(
            query.text.as_str(),
            docs.iter().map(String::as_str).collect::<Vec<_>>(),
            false,
            None,
        )?;
        ranked = order.into_iter().map(|r| (ranked[r.index].0, f64::from(r.score))).collect();
    }
    ranked.into_iter().take(query.k).map(|(id, _)| hit(id)).collect()
}

/// Reciprocal Rank Fusion: ranks only, so cosine and BM25 scales don't matter.
fn rrf(lists: &[Vec<i64>]) -> Vec<(i64, f64)> {
    let mut scores: HashMap<i64, f64> = HashMap::new();
    for list in lists {
        for (rank, id) in (1u32..).zip(list) {
            *scores.entry(*id).or_default() += 1.0 / (60.0 + f64::from(rank));
        }
    }
    let mut out: Vec<_> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// Client side, so paths are relative to the caller's cwd.
fn render(hits: &[Hit], json: bool) -> Result<String> {
    if json {
        return Ok(serde_json::to_string_pretty(hits)? + "\n");
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut out = String::new();
    for h in hits {
        let file = h.file.strip_prefix(&cwd).unwrap_or(&h.file);
        let section = h.heading.split_once(" > ").map_or("", |(_, rest)| rest);
        writeln!(out, "{}:{}  {section}\n  {}\n", file.display(), h.line, snippet(&h.text, 200))?;
    }
    Ok(out)
}

fn as_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn snippet(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", &flat[..i]),
        None => flat,
    }
}

fn socket_path(db: &Path) -> PathBuf {
    db.with_extension("sock")
}

fn spawn_daemon(db: &Path) -> Result<()> {
    use std::os::unix::process::CommandExt as _;
    let log = std::fs::OpenOptions::new().create(true).append(true).open(db.with_extension("log"))?;
    std::process::Command::new(std::env::current_exe()?)
        .arg("--db")
        .arg(db)
        .arg("watch")
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .process_group(0) // survives closing the terminal
        .spawn()?;
    Ok(())
}

fn watch(db: &Path, unload_after_min: f64) -> Result<()> {
    let unload_after = (unload_after_min > 0.0).then(|| Duration::from_secs_f64(unload_after_min * 60.0));
    // One daemon per index.
    let lock = std::fs::File::create(db.with_extension("lock"))?;
    if lock.try_lock().is_err() {
        eprintln!("a daemon is already running for {}", db.display());
        return Ok(());
    }
    anyhow::ensure!(!roots(&Connection::open(db)?).is_empty(), "no folders indexed; run `vectrize add <folder>`");
    let models = Mutex::new(Models::default());
    sync(db, None, &mut models.lock().unwrap())?;
    models.lock().unwrap().embedder()?;

    let sock = socket_path(db);
    let _ = std::fs::remove_file(&sock);
    let listener = UnixListener::bind(&sock).with_context(|| format!("creating {}", sock.display()))?;
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(tx)?;
    let mut watched = HashSet::new();
    eprintln!("serving searches on {} (Ctrl+C to quit)", sock.display());

    std::thread::scope(|s| {
        s.spawn(|| {
            for stream in listener.incoming() {
                if let Err(e) = stream.map_err(Into::into).and_then(|st| serve(st, db, &models)) {
                    eprintln!("error serving a search: {e:#}");
                }
            }
        });
        // `sync` reads files too: ignore access events or it would loop.
        let relevant = |ev: &notify::Result<notify::Event>| ev.as_ref().is_ok_and(|ev| !ev.kind.is_access());
        loop {
            let current: HashSet<PathBuf> = roots(&Connection::open(db)?).into_iter().map(|(_, r)| r).collect();
            for dir in current.difference(&watched) {
                match watcher.watch(dir, notify::RecursiveMode::Recursive) {
                    Ok(()) => eprintln!("watching {}", dir.display()),
                    Err(e) => eprintln!("cannot watch {}: {e}", dir.display()),
                }
            }
            for dir in watched.difference(&current) {
                let _ = watcher.unwatch(dir);
            }
            watched = current;

            let ev = match rx.recv_timeout(IDLE_CHECK) {
                Ok(ev) => Some(ev),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
                Err(e) => return Err(e.into()),
            };
            if let Some(after) = unload_after
                && models.lock().unwrap().unload_if_idle(after)
            {
                eprintln!("models unloaded after {unload_after_min} min idle");
            }
            if !ev.is_some_and(|ev| relevant(&ev)) {
                continue;
            }
            // Debounce: a save is a burst of events.
            while rx.recv_timeout(DEBOUNCE).is_ok() {}
            if let Err(e) = sync(db, None, &mut models.lock().unwrap()) {
                eprintln!("error: {e:#}");
            }
        }
    })
}

fn version() -> String {
    format!("{PROTOCOL}.{SCHEMA}")
}

fn ask_daemon(db: &Path, request: &str) -> Result<String> {
    let mut stream = UnixStream::connect(socket_path(db))?;
    writeln!(stream, "{}", request.replace('\n', " "))?;
    let mut out = String::new();
    stream.read_to_string(&mut out)?;
    match out.strip_prefix("ERR ") {
        Some(e) => anyhow::bail!("{e}"),
        None => Ok(out),
    }
}

fn serve(mut stream: UnixStream, db: &Path, models: &Mutex<Models>) -> Result<()> {
    let mut line = String::new();
    BufReader::new(&stream).read_line(&mut line)?;
    let response = match line.trim_end() {
        "STATUS" => {
            let loaded = models.lock().unwrap().embedder.is_some();
            let state = if loaded { "warm" } else { "unloaded (the next search reloads it, ~1.3 s)" };
            rss_mb().map(|mb| format!("{state} · {mb} MB of RAM · pid {}", std::process::id()))
        }
        "STOP" => {
            stream.write_all(b"ok")?;
            let _ = std::fs::remove_file(socket_path(db));
            std::process::exit(0); // a sync in progress rolls back
        }
        request => match request.split_once(' ') {
            Some((version, query)) if version == self::version() => serde_json::from_str(query)
                .map_err(anyhow::Error::from)
                .and_then(|query| search(db, &query, &mut models.lock().unwrap()))
                .and_then(|hits| Ok(serde_json::to_string(&hits)?)),
            _ => {
                stream.write_all(b"ERR STALE")?;
                let _ = std::fs::remove_file(socket_path(db));
                std::process::exit(0);
            }
        },
    };
    match response {
        Ok(out) => stream.write_all(out.as_bytes())?,
        Err(e) => write!(stream, "ERR {e:#}")?,
    }
    Ok(())
}

fn rss_mb() -> Result<u64> {
    let out = std::process::Command::new("ps").args(["-o", "rss=", "-p", &std::process::id().to_string()]).output()?;
    Ok(String::from_utf8(out.stdout)?.trim().parse::<u64>()? / 1024) // KB on Linux and macOS
}

fn home() -> Result<PathBuf> {
    Ok(PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?))
}

fn setup(db: &Path, dir: &Path) -> Result<()> {
    let _ = ask_daemon(db, "STOP");
    sync(db, Some(dir), &mut Models::default())?;
    let (service, ok) = install_service(db)?;
    println!();
    if ok {
        println!(
            "✓ Daemon installed ({}): starts with your session and keeps the index up to date.",
            service.display()
        );
    } else {
        println!("! Could not enable {}; the daemon will start with the first search.", service.display());
    }
    println!("\nAgents:  npx skills add pablofrr/vectrize");
    println!("Try:     vectrize search \"your question\"");
    println!("More:    vectrize add <another folder> · vectrize status");
    println!(
        "Memory:  models are unloaded after 30 min idle (set VECTRIZE_UNLOAD_AFTER=0 in the service to keep them)."
    );
    Ok(())
}

/// systemd user service or LaunchAgent. Returns the file and whether it was enabled.
fn install_service(db: &Path) -> Result<(PathBuf, bool)> {
    let exe = std::env::current_exe()?;
    let run = |cmd: &str, args: &[&str]| std::process::Command::new(cmd).args(args).status().is_ok_and(|s| s.success());
    let file = if cfg!(target_os = "macos") {
        home()?.join("Library/LaunchAgents/dev.vectrize.watch.plist")
    } else {
        home()?.join(".config/systemd/user/vectrize.service")
    };
    std::fs::create_dir_all(file.parent().unwrap())?;
    if cfg!(target_os = "macos") {
        let xml = |p: &Path| p.display().to_string().replace('&', "&amp;").replace('<', "&lt;");
        std::fs::write(
            &file,
            format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.vectrize.watch</string>
  <key>ProgramArguments</key><array><string>{exe}</string><string>--db</string><string>{db}</string><string>watch</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>StandardErrorPath</key><string>{log}</string>
</dict></plist>
"#,
                exe = xml(&exe),
                db = xml(db),
                log = xml(&db.with_extension("log")),
            ),
        )?;
        let domain = format!("gui/{}", unsafe { libc::getuid() });
        let plist = file.to_string_lossy();
        let _ = run("launchctl", &["bootout", &domain, &plist]);
        Ok((file.clone(), run("launchctl", &["bootstrap", &domain, &plist])))
    } else {
        std::fs::write(
            &file,
            format!(
                "[Unit]\nDescription=vectrize: keeps the search index up to date\n\n\
                 [Service]\nExecStart=\"{exe}\" --db \"{db}\" watch\nRestart=on-failure\n\n\
                 [Install]\nWantedBy=default.target\n",
                exe = exe.display(),
                db = db.display(),
            ),
        )?;
        let ok = run("systemctl", &["--user", "daemon-reload"])
            && run("systemctl", &["--user", "enable", "--now", "vectrize"]);
        Ok((file, ok))
    }
}

fn status(db: &Path) -> Result<()> {
    anyhow::ensure!(db.exists(), "no index at {}; create one with `vectrize add <folder>`", db.display());
    let conn = Connection::open(db)?;
    let schema: Option<i64> = conn.query_row("SELECT schema FROM meta", [], |r| r.get(0)).ok();
    if schema != Some(SCHEMA) {
        println!("index   from an older version: the next search or `vectrize add` rebuilds it (~30 s)");
        return Ok(());
    }
    for (id, root) in roots(&conn) {
        let (files, chunks): (i64, i64) = conn.query_row(
            "SELECT (SELECT count(*) FROM files WHERE root = ?1), (SELECT count(*) FROM chunks WHERE root = ?1)",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        println!("folder  {}  ({files} files · {chunks} chunks)", root.display());
    }
    let age = std::fs::metadata(db)?.modified()?.elapsed().unwrap_or_default().as_secs();
    let age = match age {
        0..60 => format!("{age} s"),
        60..3600 => format!("{} min", age / 60),
        3600..86400 => format!("{} h", age / 3600),
        _ => format!("{} d", age / 86400),
    };
    // Lock held but no socket yet: the daemon is still syncing.
    let starting = std::fs::File::open(db.with_extension("lock")).is_ok_and(|f| f.try_lock().is_err());
    let daemon = ask_daemon(db, "STATUS").unwrap_or_else(|_| {
        if starting { "starting (bringing the index up to date)" } else { "off (the next search starts it)" }.into()
    });
    println!("index   updated {age} ago · {}", db.display());
    println!("daemon  {daemon}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_rewards_agreement_between_lists() {
        let fused: Vec<i64> = rrf(&[vec![1, 2], vec![3, 2]]).into_iter().map(|(id, _)| id).collect();
        assert_eq!(fused, [2, 1, 3]);
    }
}
