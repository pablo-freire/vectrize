//! End-to-end smoke test of the real binary: add a folder, a cold search that starts the daemon, a paraphrased
//! (semantic) search, edit → findable, remove. It downloads the embedding model (559 MB) the first time, so it is
//! ignored by default: `cargo test -- --include-ignored`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn vectrize(db: &Path, args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_vectrize")).arg("--db").arg(db).args(args).output().unwrap();
    assert!(out.status.success(), "vectrize {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

fn wait_for(what: &str, mut ok: impl FnMut() -> bool) {
    let start = Instant::now();
    while !ok() {
        assert!(start.elapsed() < Duration::from_secs(30), "timeout: {what}");
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Stops the daemon and deletes the folder, also when the test fails.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = Command::new(env!("CARGO_BIN_EXE_vectrize")).arg("--db").arg(self.0.join("i.db")).arg("stop").output();
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "downloads the embedding model (559 MB)"]
fn add_search_edit_remove() {
    let tmp = Scratch(std::env::temp_dir().join(format!("vectrize-smoke-{}", std::process::id())));
    let (db, kb) = (tmp.0.join("i.db"), tmp.0.join("kb"));
    std::fs::create_dir_all(&kb).unwrap();
    std::fs::write(
        kb.join("tunnel.md"),
        "# Remote access\n\nSupport reaches the store devices through an autossh tunnel on port 2222.\n",
    )
    .unwrap();
    std::fs::write(kb.join("office.md"), "# Office\n\nThe coffee machine is cleaned every Friday.\n").unwrap();
    let kb_arg = kb.to_str().unwrap();

    vectrize(&db, &["add", kb_arg]);
    // A paraphrase with no words in common: only the embeddings can find it.
    assert!(vectrize(&db, &["search", "how do we connect to a device from outside", "-k", "1"]).contains("tunnel.md"));
    wait_for("the first search starts the daemon", || db.with_extension("sock").exists());
    assert!(vectrize(&db, &["status"]).contains("warm"));

    let mut office = std::fs::read_to_string(kb.join("office.md")).unwrap();
    office.push_str("\n# Keys\n\nThe spare key is PINEAPPLE77.\n");
    std::fs::write(kb.join("office.md"), office).unwrap();
    wait_for("the daemon reindexes the edit", || {
        vectrize(&db, &["search", "PINEAPPLE77", "-k", "1", "--json"]).contains("PINEAPPLE77")
    });

    assert!(vectrize(&db, &["remove", kb_arg]).contains("removed"));
}
