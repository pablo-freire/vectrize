//! Markdown → chunks. `text-splitter` does the splitting (by the largest Markdown unit that fits, sized in tokens of
//! the embedding model); here we only add what it doesn't give: the path of headings above each chunk, used as
//! context, and the line where the chunk starts.

use std::borrow::Cow;
use std::ops::Range;
use std::path::Path;

use pulldown_cmark::{CodeBlockKind, Event, Parser, Tag, TagEnd};
use text_splitter::{ChunkSizer, MarkdownSplitter};

pub struct Chunk {
    pub heading: String, // "Title > Section > Subsection"
    pub text: String,
    pub line: usize, // where it starts in the file (1-based)
}

pub fn chunk_markdown<S: ChunkSizer>(path: &str, content: &str, splitter: &MarkdownSplitter<S>) -> Vec<Chunk> {
    let stem = Path::new(path).file_stem().map_or(path.into(), |s| s.to_string_lossy());
    let content = blank_drawings(content);
    let headings = headings(&content);
    splitter
        .chunk_indices(&content)
        .map(|(offset, text)| {
            let mut stack: Vec<(usize, &str)> = Vec::new(); // (level, title) of the headings above
            for (_, level, title) in headings.iter().take_while(|(start, _, _)| *start <= offset) {
                stack.retain(|(l, _)| l < level);
                stack.push((*level, title));
            }
            let heading = std::iter::once(stem.as_ref()).chain(stack.iter().map(|(_, t)| *t)).collect::<Vec<_>>();
            Chunk {
                heading: heading.join(" > "),
                text: text.to_string(),
                line: 1 + content[..offset].matches('\n').count(),
            }
        })
        .collect()
}

/// `(offset, level, title)` of every heading (`#` or underlined), ignoring `#` inside code blocks.
fn headings(md: &str) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    let mut current: Option<(usize, usize, String)> = None;
    for (event, range) in Parser::new(md).into_offset_iter() {
        match event {
            Event::Start(Tag::Heading { level, .. }) => current = Some((range.start, level as usize, String::new())),
            Event::Text(t) | Event::Code(t) => {
                if let Some((_, _, title)) = &mut current {
                    title.push_str(&t);
                }
            }
            Event::End(TagEnd::Heading(_)) => out.extend(current.take()),
            _ => {}
        }
    }
    out
}

/// Excalidraw drawings (```compressed-json blocks) are base64 noise: blank them out, keeping the line count.
fn blank_drawings(md: &str) -> Cow<'_, str> {
    let drawings: Vec<Range<usize>> = Parser::new(md)
        .into_offset_iter()
        .filter_map(|(event, range)| match event {
            Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(info))) if &*info == "compressed-json" => Some(range),
            _ => None,
        })
        .collect();
    if drawings.is_empty() {
        return Cow::Borrowed(md);
    }
    let mut out = String::with_capacity(md.len());
    let mut last = 0;
    for r in drawings {
        out.push_str(&md[last..r.start]);
        out.push_str(&"\n".repeat(md[r.clone()].matches('\n').count()));
        last = r.end;
    }
    out.push_str(&md[last..]);
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks(md: &str, chars: usize) -> Vec<(String, String, usize)> {
        let splitter = MarkdownSplitter::new(chars); // sized in characters: no model needed
        chunk_markdown("dir/Doc.md", md, &splitter).into_iter().map(|c| (c.heading, c.text, c.line)).collect()
    }

    #[test]
    fn heading_path_and_lines_ignoring_code_blocks() {
        let md = "intro\n\n# A\n\ntext a\n\n## B\n\n```sh\n# not a heading\n```\n\nSetext C\n========\n\ntext c\n";
        let got = chunks(md, 30);
        let got: Vec<_> = got.iter().map(|(h, t, l)| (h.as_str(), t.as_str(), *l)).collect();
        assert_eq!(
            got,
            [
                ("Doc", "intro", 1),
                ("Doc > A", "# A\n\ntext a", 3),
                ("Doc > A > B", "## B", 7),
                ("Doc > A > B", "```sh\n# not a heading\n```", 9),
                ("Doc > Setext C", "Setext C\n========\n\ntext c", 13),
            ]
        );
    }

    #[test]
    fn drops_excalidraw_drawings_keeping_lines() {
        let md = "# Sketch\n\n```compressed-json\nN4IgLgngDgpiBcIYA8DGBDANmAvgGiTS\nGARBAGE\n```\n\nafter\n";
        let got = chunks(md, 1000);
        assert_eq!(got.len(), 1);
        assert!(!got[0].1.contains("GARBAGE") && got[0].1.ends_with("after"));
        assert_eq!(chunks("x\n\n```compressed-json\nA\nB\n```\n\n# T\n\ny\n", 3).last().unwrap().2, 10);
    }
}
