//! vectrize: búsqueda semántica local sobre una carpeta de docs.
//!
//! Trocear por encabezados → embeddings estáticos (model2vec) en sqlite-vec + BM25 (FTS5),
//! fusionados con RRF.
//! Cada `index` reconstruye el índice entero (el incremental llega en la fase 3).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use model2vec_rs::model::StaticModel;
use rusqlite::{params, Connection};

/// Multilingüe: la KB está en español. Los `potion-base-*` son solo inglés.
const MODEL: &str = "minishlab/potion-multilingual-128M";
/// Extensiones que se leen como texto plano. PDF/HTML llegan en la fase 5.
const TEXT_EXTS: &[&str] = &["md", "mmd", "puml"];
const SKIP_DIRS: &[&str] = &[".git", ".obsidian"];

#[derive(Parser)]
#[command(about = "Búsqueda semántica local sobre una carpeta de docs")]
struct Cli {
    /// Ruta del índice.
    #[arg(long, global = true, default_value_os_t = default_db())]
    db: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// (Re)construye el índice de una carpeta.
    Index { dir: PathBuf },
    /// Busca en el índice.
    Search {
        query: String,
        #[arg(short, default_value_t = 5)]
        k: usize,
        #[arg(long, value_enum, default_value_t = Mode::Hybrid)]
        mode: Mode,
    },
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Mode {
    Hybrid,
    Vec,
    Bm25,
}

/// Candidatos que aporta cada buscador antes de fusionar.
const CANDIDATES: i64 = 50;

fn default_db() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share"));
    base.join("vectrize/index.db")
}

struct Chunk {
    path: String,    // relativo a la raíz de la KB
    heading: String, // "Título > Sección > Subsección"
    text: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // sqlite-vec va enlazado estáticamente; esto lo registra en cada conexión que se abra.
    // `unsafe` porque es FFI: Rust no puede comprobar la firma de la función de C.
    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    }
    match cli.cmd {
        Cmd::Index { dir } => index(&cli.db, &dir),
        Cmd::Search { query, k, mode } => search(&cli.db, &query, k, mode),
    }
}

fn load_model() -> Result<StaticModel> {
    StaticModel::from_pretrained(MODEL, None, None, None).context("cargando el modelo")
}

fn index(db: &Path, dir: &Path) -> Result<()> {
    let root = dir.canonicalize().with_context(|| format!("no existe {}", dir.display()))?;
    let t = std::time::Instant::now();

    let mut chunks = Vec::new();
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
        let content = std::fs::read_to_string(path).with_context(|| format!("leyendo {}", path.display()))?;
        let rel = path.strip_prefix(&root)?.to_string_lossy().into_owned();
        chunks.extend(chunk_markdown(&rel, &content));
        n_files += 1;
    }
    let t_read = t.elapsed();

    let model = load_model()?;
    let t_model = t.elapsed();
    // Se embebe "ruta > encabezados + texto": el contexto del trozo cuenta para la búsqueda.
    let inputs: Vec<String> = chunks.iter().map(|c| format!("{}\n{}", c.heading, c.text)).collect();
    let embeddings = model.encode_with_args(&inputs, None, 1024); // None: sin truncar
    let dim = embeddings.first().map_or(0, Vec::len);
    let t_embed = t.elapsed();

    if let Some(parent) = db.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut conn = Connection::open(db)?;
    conn.execute_batch(&format!(
        "DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS vec; DROP TABLE IF EXISTS meta; DROP TABLE IF EXISTS fts;
         CREATE TABLE meta (root TEXT NOT NULL);
         CREATE TABLE chunks (id INTEGER PRIMARY KEY, path TEXT, heading TEXT, text TEXT);
         CREATE VIRTUAL TABLE vec USING vec0(embedding float[{dim}] distance_metric=cosine);
         CREATE VIRTUAL TABLE fts USING fts5(heading, text, tokenize='unicode61 remove_diacritics 2');"
    ))?;
    let tx = conn.transaction()?;
    tx.execute("INSERT INTO meta VALUES (?1)", [root.to_string_lossy()])?;
    for (i, (c, e)) in chunks.iter().zip(&embeddings).enumerate() {
        tx.execute(
            "INSERT INTO chunks VALUES (?1, ?2, ?3, ?4)",
            params![i as i64, c.path, c.heading, c.text],
        )?;
        tx.execute("INSERT INTO vec (rowid, embedding) VALUES (?1, ?2)", params![i as i64, as_bytes(e)])?;
        tx.execute("INSERT INTO fts (rowid, heading, text) VALUES (?1, ?2, ?3)", params![i as i64, c.heading, c.text])?;
    }
    tx.commit()?;

    eprintln!(
        "{n_files} archivos, {} trozos, dim {dim} → {}\n  leer+trocear {:?} · modelo {:?} · embeddings {:?} · sqlite {:?}",
        chunks.len(),
        db.display(),
        t_read,
        t_model - t_read,
        t_embed - t_model,
        t.elapsed() - t_embed,
    );
    Ok(())
}

fn search(db: &Path, query: &str, k: usize, mode: Mode) -> Result<()> {
    let conn = Connection::open(db).with_context(|| format!("sin índice en {}", db.display()))?;
    let ids = |sql: &str, arg: &dyn rusqlite::ToSql| -> Result<Vec<i64>> {
        let mut stmt = conn.prepare(sql)?;
        Ok(stmt.query_map(params![arg, CANDIDATES], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?)
    };
    let mut lists = Vec::new();
    if mode != Mode::Bm25 {
        let q = as_bytes(&load_model()?.encode_single(query));
        lists.push(ids("SELECT rowid FROM vec WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance", &q)?);
    }
    if mode != Mode::Vec {
        // Cada palabra entre comillas (así `¿` o `-` no son sintaxis FTS5), unidas con OR.
        let words: Vec<_> = query.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
        let q = words.iter().map(|w| format!("\"{w}\"")).collect::<Vec<_>>().join(" OR ");
        if !q.is_empty() {
            lists.push(ids("SELECT rowid FROM fts WHERE fts MATCH ?1 ORDER BY rank LIMIT ?2", &q)?);
        }
    }
    let mut stmt = conn.prepare("SELECT path, heading, text FROM chunks WHERE id = ?1")?;
    for (i, (id, score)) in rrf(&lists).into_iter().take(k).enumerate() {
        let (path, heading, text): (String, String, String) =
            stmt.query_row([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        println!("{}. {path}  ({score:.4})\n   {heading}\n   {}\n", i + 1, snippet(&text, 160));
    }
    Ok(())
}

/// Reciprocal Rank Fusion: cada lista suma 1/(60 + puesto). Solo usa posiciones, así que da igual
/// que coseno y BM25 estén en escalas distintas.
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

/// sqlite-vec espera los f32 como bytes little-endian contiguos.
fn as_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn snippet(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", &flat[..i]), // cortar por char, no por byte: UTF-8
        None => flat,
    }
}

/// Parte un documento por encabezados `#`. Cada trozo lleva la ruta de títulos que lo contiene.
/// Los `#` dentro de bloques ``` no cuentan (comentarios de bash, etc.).
fn chunk_markdown(path: &str, content: &str) -> Vec<Chunk> {
    let stem = Path::new(path).file_stem().map_or(path.into(), |s| s.to_string_lossy());
    let mut stack: Vec<(usize, String)> = Vec::new(); // (nivel, título)
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
        let text = buf.trim();
        if !text.is_empty() {
            out.push(Chunk { path: path.into(), heading, text: text.into() });
        }
        buf.clear();
    };

    for line in content.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        }
        let level = line.bytes().take_while(|&b| b == b'#').count();
        let is_heading = !in_fence && (1..=6).contains(&level) && line[level..].starts_with(' ');
        if is_heading {
            flush(&mut buf, heading_of(&stack), &mut out);
            stack.retain(|(l, _)| *l < level);
            stack.push((level, line[level..].trim().to_string()));
        } else {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    flush(&mut buf, heading_of(&stack), &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_by_heading_with_path_and_ignores_fences() {
        let md = "intro\n# A\ntexto a\n## B\n```sh\n# no es título\n```\n# C\ntexto c\n";
        let c = chunk_markdown("dir/Doc.md", md);
        let got: Vec<_> = c.iter().map(|c| (c.heading.as_str(), c.text.as_str())).collect();
        assert_eq!(
            got,
            [
                ("Doc", "intro"),
                ("Doc > A", "texto a"),
                ("Doc > A > B", "```sh\n# no es título\n```"),
                ("Doc > C", "texto c"),
            ]
        );
    }

    #[test]
    fn rrf_rewards_agreement_between_lists() {
        // 2 es segundo en ambas listas; 1 y 3 son primeros solo en una.
        let fused: Vec<i64> = rrf(&[vec![1, 2], vec![3, 2]]).into_iter().map(|(id, _)| id).collect();
        assert_eq!(fused, [2, 1, 3]);
    }

}
