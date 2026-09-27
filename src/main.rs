//! vectrize: búsqueda semántica local sobre una carpeta de docs.
//!
//! Trocear por encabezados → embeddings (bge-m3 int8) en sqlite-vec + BM25 (FTS5),
//! fusionados con RRF. Opcional: reordenar los primeros con un cross-encoder (`--rerank`).
//! Cada `index` reconstruye el índice entero (el incremental llega en la fase 3).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use fastembed::{
    InitOptionsUserDefined, Pooling, RerankInitOptionsUserDefined, TextEmbedding, TextRerank, TokenizerFiles,
    UserDefinedEmbeddingModel, UserDefinedRerankingModel,
};
use rusqlite::{params, Connection};

/// bge-m3 (MIT, multilingüe) cuantizado a int8: 559 MB, igual calidad que el fp32 en la evaluación.
/// Frente a potion (estático) resuelve las consultas parafraseadas.
const EMBEDDER: &str = "onnx-community/bge-m3-ONNX";
const EMBEDDER_ONNX: &str = "onnx/model_int8.onnx";
/// Extensiones que se leen como texto plano. PDF/HTML llegan en la fase 5.
const TEXT_EXTS: &[&str] = &["md", "mmd", "puml"];
const SKIP_DIRS: &[&str] = &[".git", ".obsidian"];
/// Tope por trozo (~400 tokens). Las secciones más largas se parten por párrafos.
const MAX_CHUNK_CHARS: usize = 1500;
/// Cross-encoder multilingüe, Apache-2.0. La variante int8 aprovecha AVX-512 VNNI.
const RERANKER: &str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1";
const RERANKER_ONNX: &str = "onnx/model_qint8_avx512_vnni.onnx";
/// Con 10 se pierden aciertos que BM25 deja entre el 11 y el 20.
const RERANK_CANDIDATES: usize = 20;
/// Tokens máximos por par (pregunta, trozo).
const RERANK_MAX_LENGTH: usize = 256;

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
        /// Reordena los primeros con un cross-encoder (+~1 s; en la evaluación no mejoró sobre hybrid).
        #[arg(long)]
        rerank: bool,
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
        Cmd::Search { query, k, mode, rerank } => search(&cli.db, &query, k, mode, rerank),
    }
}

/// Los 4 json del tokenizador de un repo de Hugging Face (caché en ~/.cache/huggingface).
fn tokenizer_files(repo: &hf_hub::api::sync::ApiRepo) -> Result<TokenizerFiles> {
    let read = |f: &str| -> Result<Vec<u8>> { Ok(std::fs::read(repo.get(f)?)?) };
    Ok(TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    })
}

fn load_embedder() -> Result<TextEmbedding> {
    let repo = hf_hub::api::sync::Api::new()?.model(EMBEDDER.into());
    let model = UserDefinedEmbeddingModel::new(std::fs::read(repo.get(EMBEDDER_ONNX)?)?, tokenizer_files(&repo)?)
        .with_pooling(Pooling::Cls); // el que usa bge-m3
    TextEmbedding::try_new_from_user_defined(model, InitOptionsUserDefined::new()).context("cargando el modelo")
}

fn load_reranker() -> Result<TextRerank> {
    let repo = hf_hub::api::sync::Api::new()?.model(RERANKER.into());
    let model = UserDefinedRerankingModel::new(repo.get(RERANKER_ONNX)?, tokenizer_files(&repo)?);
    let opts = RerankInitOptionsUserDefined::new().with_max_length(RERANK_MAX_LENGTH);
    TextRerank::try_new_from_user_defined(model, opts).context("cargando el reranker")
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

    let mut embedder = load_embedder()?;
    let t_model = t.elapsed();
    // Se embebe "ruta > encabezados + texto": el contexto del trozo cuenta para la búsqueda.
    let inputs: Vec<String> = chunks.iter().map(|c| format!("{}\n{}", c.heading, c.text)).collect();
    let embeddings = embedder.embed(inputs, None)?;
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

fn search(db: &Path, query: &str, k: usize, mode: Mode, rerank: bool) -> Result<()> {
    let conn = Connection::open(db).with_context(|| format!("sin índice en {}", db.display()))?;
    let ids = |sql: &str, arg: &dyn rusqlite::ToSql| -> Result<Vec<i64>> {
        let mut stmt = conn.prepare(sql)?;
        Ok(stmt.query_map(params![arg, CANDIDATES], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?)
    };
    let mut lists = Vec::new();
    if mode != Mode::Bm25 {
        let q = as_bytes(&load_embedder()?.embed([query], None)?[0]);
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
    let mut chunk = |id: i64| -> Result<(String, String, String)> {
        Ok(stmt.query_row([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?)
    };
    let mut hits = rrf(&lists);
    if rerank {
        hits.truncate(RERANK_CANDIDATES);
        let docs = hits
            .iter()
            .map(|&(id, _)| chunk(id).map(|(_, heading, text)| format!("{heading}\n{text}")))
            .collect::<Result<Vec<_>>>()?;
        let ranked = load_reranker()?.rerank(query, docs.iter().map(String::as_str).collect::<Vec<_>>(), false, None)?;
        hits = ranked.into_iter().map(|r| (hits[r.index].0, r.score as f64)).collect();
    }
    for (i, (id, score)) in hits.into_iter().take(k).enumerate() {
        let (path, heading, text) = chunk(id)?;
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

/// Parte un documento por encabezados `#` (o títulos entre líneas `=====`), con un tope de tamaño.
/// Cada trozo lleva la ruta de títulos que lo contiene. Los `#` dentro de bloques ``` no cuentan
/// (comentarios de bash, etc.) y los bloques ```compressed-json (dibujos de Excalidraw) se descartan.
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
        for text in split_long(buf.trim(), MAX_CHUNK_CHARS) {
            out.push(Chunk { path: path.into(), heading: heading.clone(), text });
        }
        buf.clear();
    };
    let is_rule = |l: &str| l.len() >= 10 && l.bytes().all(|b| b == b'=');

    let lines: Vec<&str> = content.lines().collect();
    let mut skip = false; // dentro de un bloque ```compressed-json
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
        // `=====` / TÍTULO / `=====` cuenta como encabezado de nivel 1.
        let rule_title = (!in_fence && is_rule(line) && i + 1 < lines.len() && is_rule(lines[i + 1]))
            .then(|| lines[i].trim())
            .filter(|t| !t.is_empty());
        let level = line.bytes().take_while(|&b| b == b'#').count();
        let md_title = (!in_fence && (1..=6).contains(&level) && line[level..].starts_with(' '))
            .then(|| line[level..].trim());
        if let Some((level, title)) = rule_title.map(|t| (1, t)).or(md_title.map(|t| (level, t))) {
            flush(&mut buf, heading_of(&stack), &mut out);
            stack.retain(|(l, _)| *l < level);
            stack.push((level, title.to_string()));
            if rule_title.is_some() {
                i += 2; // saltar el título y la segunda línea de `=`
            }
        } else {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    flush(&mut buf, heading_of(&stack), &mut out);
    out
}

/// Parte un texto en trozos de como mucho `max` bytes: por párrafos, y si un párrafo no cabe, por
/// líneas; una línea que tampoco cabe se corta a la fuerza. Los párrafos pequeños se juntan.
fn split_long(text: &str, max: usize) -> Vec<String> {
    let mut units = Vec::new();
    for para in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if para.len() <= max {
            units.push(para);
            continue;
        }
        for mut line in para.lines() {
            while line.len() > max {
                let cut = line.floor_char_boundary(max); // no partir un carácter UTF-8
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
    fn rule_headings_drop_excalidraw_and_size_cap() {
        let big = "palabra ".repeat(MAX_CHUNK_CHARS / 8 * 3); // ~3 veces el tope, en un solo párrafo
        let md = format!(
            "=====================\nRESUMEN\n=====================\nhola\n\n{big}\n```compressed-json\nBASURA\n```\nfin\n"
        );
        let c = chunk_markdown("Doc.md", &md);
        assert!(c.iter().all(|c| c.heading == "Doc > RESUMEN" && c.text.len() <= MAX_CHUNK_CHARS));
        assert!(c.len() >= 3);
        let all: String = c.iter().map(|c| c.text.as_str()).collect();
        assert!(all.contains("hola") && all.contains("fin") && !all.contains("BASURA") && !all.contains("===="));
    }

    #[test]
    fn rrf_rewards_agreement_between_lists() {
        // 2 es segundo en ambas listas; 1 y 3 son primeros solo en una.
        let fused: Vec<i64> = rrf(&[vec![1, 2], vec![3, 2]]).into_iter().map(|(id, _)| id).collect();
        assert_eq!(fused, [2, 1, 3]);
    }

}
