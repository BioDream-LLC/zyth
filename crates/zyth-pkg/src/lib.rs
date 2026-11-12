//! Zyth's package tools, open source (Apache-2.0 OR MIT), so anyone can run, check and audit
//! what `zyth` relies on for packages:
//!
//! - [`sumdb`]: the checksum log (`zyth sum-server`), the checks clients make against it, and
//!   the independent auditor and witness (`zyth sum-audit`).
//! - [`registry`]: the package registry (`zyth registry-server`, `zyth search`).
//! - [`tree_hash`]: the content hash of a package's files, as the log records it.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

pub mod registry;
pub mod sumdb;

/// An HTTP(S) request: method, URL, headers, body, timeout -> status and body, or a message.
pub type Http = fn(&str, &str, &[(String, String)], &[u8], Duration) -> Result<(u16, Vec<u8>), String>;

static HTTP: OnceLock<Http> = OnceLock::new();

/// Sets how the package tools make HTTP(S) requests (`zyth` passes its own client at start-up).
pub fn set_http(f: Http) {
    let _ = HTTP.set(f);
}

pub(crate) fn request(method: &str, url: &str, headers: &[(String, String)], body: &[u8], timeout: Duration) -> Result<(u16, Vec<u8>), String> {
    match HTTP.get() {
        Some(f) => f(method, url, headers, body, timeout),
        None => Err("no HTTP client is set for the package tools (zyth_pkg::set_http)".into()),
    }
}

/// A string field of `[project]` in a zyth.toml.
pub fn project_field(toml: &str, key: &str) -> Option<String> {
    let mut in_project = false;
    for line in toml.lines() {
        let l = line.split('#').next().unwrap_or("").trim();
        if l.starts_with('[') {
            in_project = l == "[project]";
            continue;
        }
        if !in_project {
            continue;
        }
        if let Some((k, v)) = l.split_once('=') {
            if k.trim() == key {
                return Some(v.trim().trim_matches('"').to_string());
            }
        }
    }
    None
}

/// Zyth's cache directory: `ZYTH_CACHE_DIR`, else `$XDG_CACHE_HOME/zyth`, `%LOCALAPPDATA%\zyth`
/// on Windows, or `~/.cache/zyth`.
pub fn cache_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("ZYTH_CACHE_DIR") {
        return PathBuf::from(d);
    }
    if let Some(d) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(d).join("zyth");
    }
    if cfg!(windows) {
        if let Some(d) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(d).join("zyth");
        }
    }
    PathBuf::from(std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).unwrap_or_default()).join(".cache").join("zyth")
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The content hash of a package directory (`sha256-<hex>`): every file but `.git`'s, sorted by
/// its path relative to `dir` (`/` separators), each as the path, a zero byte, its length (u64,
/// little-endian) and its bytes, all hashed with SHA-256.
pub fn tree_hash(dir: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let rd = std::fs::read_dir(&d).map_err(|e| format!("cannot read {}: {e}", d.display()))?;
        for e in rd.flatten() {
            let p = e.path();
            if e.file_name() == ".git" {
                continue;
            }
            let ft = e.file_type().map_err(|e| e.to_string())?;
            if ft.is_dir() {
                stack.push(p);
            } else {
                let rel = p.strip_prefix(dir).unwrap_or(&p).to_string_lossy().replace('\\', "/");
                files.push((rel, p));
            }
        }
    }
    files.sort();
    let mut h = Sha256::new();
    for (rel, p) in &files {
        let data = std::fs::read(p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
        h.update(rel.as_bytes());
        h.update([0u8]);
        h.update((data.len() as u64).to_le_bytes());
        h.update(&data);
    }
    Ok(format!("sha256-{}", hex(&h.finalize())))
}

/// `%XX` escapes in a URL query value decoded (invalid escapes kept as they are).
pub fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
