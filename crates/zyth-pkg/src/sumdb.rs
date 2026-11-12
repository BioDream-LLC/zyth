//! The checksum log for Zyth packages (spec 15.5), as Go's checksum database: a public,
//! append-only log of each package version's content hash, so a version's files cannot change
//! unseen, even for a project installing it for the first time.
//!
//! - The log (`zyth sum-server`) computes every hash itself from the package's Git origin (it
//!   never takes a hash or files from a client or from a proxy), appends a record
//!   `<url> <version> <tag> <commit> <hash> <first seen>` and signs the tree head with Ed25519.
//! - The tree is RFC 6962's Merkle tree (leaf `SHA-256(0x00 || record)`, node
//!   `SHA-256(0x01 || left || right)`); lookups come with an inclusion proof, and a client
//!   asks for a consistency proof from the last tree head it saw, so a log that rewrote its
//!   past is caught.
//! - Clients (`zyth add`, `install`, `update`) check each version they take that is not yet in
//!   zyth.lock: the hash must match the log's, the proofs must verify, and the tree head must
//!   be signed by the log's key. The time the log first saw a version also counts for the
//!   cooldown. `ZYTH_SUMDB` names another log (`off`: no checks), `ZYTH_SUMDB_KEY` its key.
//! - Anyone can audit the log (`zyth sum-audit`): it reads every record (`/records`), rebuilds
//!   the tree and checks it against each signed head, checks that each head extends the last
//!   one it saw, refuses two hashes for one version, and works out each new version's hash
//!   from its Git origin itself. A head it has checked it cosigns with its own key and serves
//!   (`/cosigned`), as a witness. Clients given witnesses (`ZYTH_SUMDB_WITNESSES`) require
//!   the log's tree to be consistent with each witness's latest cosigned head, so a log
//!   cannot show one history to them and another to its auditors.

use ring::signature::{Ed25519KeyPair, KeyPair, UnparsedPublicKey, ED25519};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

/// The public key of the log at sum.zyth.io (Ed25519, hex).
pub const ZYTH_LOG_KEY: &str = "c2492b9da22cee872523b20d6acfa3790ce4db5c80490b8b35053f14f756d0b0";
const DEFAULT_LOG: &str = "https://sum.zyth.io";

type H = [u8; 32];

fn sha(parts: &[&[u8]]) -> H {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

fn leaf(record: &[u8]) -> H {
    sha(&[&[0], record])
}

fn node(l: &H, r: &H) -> H {
    sha(&[&[1], l, r])
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok()).collect()
}

fn h32(s: &str) -> Option<H> {
    unhex(s)?.try_into().ok()
}

/// The largest power of two smaller than n (n > 1).
fn split(n: usize) -> usize {
    let mut k = 1;
    while k << 1 < n {
        k <<= 1;
    }
    k
}

/// MTH(D[lo..hi]) (RFC 6962 2.1).
fn root(leaves: &[H]) -> H {
    match leaves.len() {
        0 => sha(&[]),
        1 => leaves[0],
        n => {
            let k = split(n);
            node(&root(&leaves[..k]), &root(&leaves[k..]))
        }
    }
}

/// PATH(m, D[n]) (RFC 6962 2.1.1).
fn inclusion(m: usize, leaves: &[H]) -> Vec<H> {
    let n = leaves.len();
    if n <= 1 {
        return Vec::new();
    }
    let k = split(n);
    if m < k {
        let mut p = inclusion(m, &leaves[..k]);
        p.push(root(&leaves[k..]));
        p
    } else {
        let mut p = inclusion(m - k, &leaves[k..]);
        p.push(root(&leaves[..k]));
        p
    }
}

/// PROOF(m, D[n]) (RFC 6962 2.1.2).
fn consistency(m: usize, leaves: &[H]) -> Vec<H> {
    fn sub(m: usize, leaves: &[H], whole: bool) -> Vec<H> {
        let n = leaves.len();
        if m == n {
            return if whole { Vec::new() } else { vec![root(leaves)] };
        }
        let k = split(n);
        if m <= k {
            let mut p = sub(m, &leaves[..k], whole);
            p.push(root(&leaves[k..]));
            p
        } else {
            let mut p = sub(m - k, &leaves[k..], false);
            p.push(root(&leaves[..k]));
            p
        }
    }
    if m == 0 || m >= leaves.len() {
        return Vec::new();
    }
    sub(m, leaves, true)
}

/// RFC 9162 2.1.3.2: does `proof` show leaf `index` with hash `lh` in a tree of `size` with `root`?
fn verify_inclusion(index: usize, size: usize, lh: H, proof: &[H], want: &H) -> bool {
    if index >= size {
        return false;
    }
    let (mut fnn, mut sn) = (index, size - 1);
    let mut r = lh;
    for p in proof {
        if sn == 0 {
            return false;
        }
        if fnn & 1 == 1 || fnn == sn {
            r = node(p, &r);
            if fnn & 1 == 0 {
                while fnn & 1 == 0 && fnn != 0 {
                    fnn >>= 1;
                    sn >>= 1;
                }
            }
        } else {
            r = node(&r, p);
        }
        fnn >>= 1;
        sn >>= 1;
    }
    sn == 0 && &r == want
}

/// RFC 9162 2.1.4.2: is the tree of size `n2` with `r2` an extension of `n1` with `r1`?
fn verify_consistency(n1: usize, n2: usize, r1: &H, r2: &H, proof: &[H]) -> bool {
    if n1 == n2 {
        return proof.is_empty() && r1 == r2;
    }
    if n1 == 0 {
        return true;
    }
    if n1 > n2 || proof.is_empty() {
        return false;
    }
    let mut path: Vec<H> = proof.to_vec();
    if n1.is_power_of_two() {
        path.insert(0, *r1);
    }
    let (mut fnn, mut sn) = (n1 - 1, n2 - 1);
    while fnn & 1 == 1 {
        fnn >>= 1;
        sn >>= 1;
    }
    let (mut fr, mut sr) = (path[0], path[0]);
    for c in &path[1..] {
        if sn == 0 {
            return false;
        }
        if fnn & 1 == 1 || fnn == sn {
            fr = node(c, &fr);
            sr = node(c, &sr);
            if fnn & 1 == 0 {
                while fnn & 1 == 0 && fnn != 0 {
                    fnn >>= 1;
                    sn >>= 1;
                }
            }
        } else {
            sr = node(&sr, c);
        }
        fnn >>= 1;
        sn >>= 1;
    }
    sn == 0 && &fr == r1 && &sr == r2
}

/// What the tree head's signature covers.
fn head_text(size: usize, root: &H, time: u64) -> String {
    format!("zyth sum log v1\n{size}\n{}\n{time}\n", hex(root))
}

#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub url: String,
    pub version: String,
    pub tag: String,
    pub commit: String,
    pub hash: String,
    pub time: u64,
}

impl Record {
    fn line(&self) -> String {
        format!("{} {} {} {} {} {}", self.url, self.version, self.tag, self.commit, self.hash, self.time)
    }
    fn parse(l: &str) -> Option<Record> {
        let p: Vec<&str> = l.split(' ').collect();
        if p.len() != 6 {
            return None;
        }
        Some(Record { url: p[0].into(), version: p[1].into(), tag: p[2].into(), commit: p[3].into(), hash: p[4].into(), time: p[5].parse().ok()? })
    }
}

// ------------------------------------------------------------------ the server

struct Log {
    dir: PathBuf,
    records: Vec<Record>,
    leaves: Vec<H>,
    key: Ed25519KeyPair,
    allow_local: bool,
    /// appends per client address in the last hour (fetching a repository costs the log)
    recent: std::collections::HashMap<String, Vec<u64>>,
}

impl Log {
    fn head(&self) -> serde_json::Value {
        let r = root(&self.leaves);
        let time = now();
        let sig = self.key.sign(head_text(self.leaves.len(), &r, time).as_bytes());
        serde_json::json!({ "size": self.leaves.len(), "root": hex(&r), "time": time, "sig": hex(sig.as_ref()) })
    }

    fn append(&mut self, rec: Record) -> std::io::Result<usize> {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("records"))?;
        writeln!(f, "{}", rec.line())?;
        f.sync_all()?;
        self.leaves.push(leaf(rec.line().as_bytes()));
        self.records.push(rec);
        Ok(self.records.len() - 1)
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// The commit a tag names at `url`, and the version's hash, worked out by the log from the
/// origin itself.
fn from_origin(url: &str, tag: &str, allow_local: bool) -> Result<(String, String), String> {
    let remote = if url.starts_with('/') || url.contains(":\\") || url.starts_with("file:") {
        if !allow_local {
            return Err("this log takes https:// repositories".into());
        }
        url.to_string()
    } else if url.starts_with("https://") {
        url.to_string()
    } else {
        format!("https://{url}")
    };
    let git = |args: &[&str], dir: Option<&Path>| -> Result<String, String> {
        let mut c = std::process::Command::new("git");
        c.args(args).env("GIT_TERMINAL_PROMPT", "0");
        if let Some(d) = dir {
            c.current_dir(d);
        }
        let o = c.output().map_err(|e| format!("cannot run git: {e}"))?;
        if !o.status.success() {
            return Err(String::from_utf8_lossy(&o.stderr).trim().to_string());
        }
        Ok(String::from_utf8_lossy(&o.stdout).to_string())
    };
    let tmp = std::env::temp_dir().join(format!("zyth-sum-{}-{}", std::process::id(), now()));
    let _ = std::fs::remove_dir_all(&tmp);
    let tmp_s = tmp.to_string_lossy().to_string();
    let r = (|| {
        git(&["clone", "-q", "-c", "core.autocrlf=false", "--depth", "1", "--branch", tag, &remote, &tmp_s], None)?;
        let commit = git(&["rev-parse", "HEAD"], Some(&tmp))?.trim().to_string();
        let _ = std::fs::remove_dir_all(tmp.join(".git"));
        let size: u64 = walk_size(&tmp);
        if size > 100 << 20 {
            return Err("the package is over 100 MB".into());
        }
        Ok((commit, crate::tree_hash(&tmp)?))
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    r
}

fn walk_size(d: &Path) -> u64 {
    std::fs::read_dir(d).map(|rd| rd.flatten().map(|e| if e.path().is_dir() { walk_size(&e.path()) } else { e.metadata().map(|m| m.len()).unwrap_or(0) }).sum()).unwrap_or(0)
}

fn query(q: &str, k: &str) -> Option<String> {
    q.split('&').find_map(|kv| kv.split_once('=').filter(|(a, _)| *a == k).map(|(_, v)| crate::percent_decode(v)))
}

fn send_json(s: &mut TcpStream, status: &str, v: serde_json::Value) {
    let body = v.to_string();
    let _ = write!(s, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{body}", body.len());
}

fn handle(log: &Mutex<Log>, mut s: TcpStream) {
    let ip = s.peer_addr().map(|a| a.ip().to_string()).unwrap_or_default();
    let mut r = BufReader::new(match s.try_clone() {
        Ok(c) => c,
        Err(_) => return,
    });
    let mut line = String::new();
    if r.by_ref().take(8192).read_line(&mut line).is_err() {
        return;
    }
    // headers are not needed; read them so the client is not reset
    loop {
        let mut h = String::new();
        if r.by_ref().take(8192).read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" || h == "\n" {
            break;
        }
    }
    let mut it = line.split_whitespace();
    let (Some(method), Some(target)) = (it.next(), it.next()) else { return };
    let (path, q) = target.split_once('?').unwrap_or((target, ""));
    if method != "GET" {
        return send_json(&mut s, "405 Method Not Allowed", serde_json::json!({ "error": "GET only" }));
    }
    match path {
        "/tree" => {
            let head = log.lock().unwrap().head();
            send_json(&mut s, "200 OK", head)
        }
        "/consistency" => {
            let g = log.lock().unwrap();
            let old: usize = query(q, "old").and_then(|v| v.parse().ok()).unwrap_or(0);
            if old > g.leaves.len() {
                return send_json(&mut s, "400 Bad Request", serde_json::json!({ "error": "the log is smaller than that" }));
            }
            let p: Vec<String> = consistency(old, &g.leaves).iter().map(|h| hex(h)).collect();
            send_json(&mut s, "200 OK", serde_json::json!({ "old": old, "proof": p, "tree": g.head() }))
        }
        "/lookup" => {
            let (Some(url), Some(version), Some(tag)) = (query(q, "url"), query(q, "version"), query(q, "tag")) else {
                return send_json(&mut s, "400 Bad Request", serde_json::json!({ "error": "url, version and tag are needed" }));
            };
            if [&url, &version, &tag].iter().any(|v| v.is_empty() || v.len() > 300 || v.contains(char::is_whitespace)) {
                return send_json(&mut s, "400 Bad Request", serde_json::json!({ "error": "bad url, version or tag" }));
            }
            let found = {
                let g = log.lock().unwrap();
                g.records.iter().position(|r| r.url == url && r.version == version)
            };
            let index = match found {
                Some(i) => i,
                None => {
                    {
                        let mut g = log.lock().unwrap();
                        let t = now();
                        let v = g.recent.entry(ip.clone()).or_default();
                        v.retain(|&x| t - x < 3600);
                        if v.len() >= 200 {
                            return send_json(&mut s, "429 Too Many Requests", serde_json::json!({ "error": "at most 200 new versions an hour" }));
                        }
                        v.push(t);
                    }
                    let allow_local = log.lock().unwrap().allow_local;
                    let (commit, hash) = match from_origin(&url, &tag, allow_local) {
                        Ok(x) => x,
                        Err(e) => return send_json(&mut s, "404 Not Found", serde_json::json!({ "error": format!("cannot read {url} at {tag}: {e}") })),
                    };
                    let mut g = log.lock().unwrap();
                    // (another request may have added it meanwhile)
                    match g.records.iter().position(|r| r.url == url && r.version == version) {
                        Some(i) => i,
                        None => match g.append(Record { url: url.clone(), version: version.clone(), tag: tag.clone(), commit, hash, time: now() }) {
                            Ok(i) => i,
                            Err(e) => return send_json(&mut s, "500 Internal Server Error", serde_json::json!({ "error": format!("cannot write the log: {e}") })),
                        },
                    }
                }
            };
            let g = log.lock().unwrap();
            let rec = &g.records[index];
            let proof: Vec<String> = inclusion(index, &g.leaves).iter().map(|h| hex(h)).collect();
            send_json(&mut s, "200 OK", serde_json::json!({ "record": rec.line(), "index": index, "proof": proof, "tree": g.head() }))
        }
        // the records themselves, so anyone can audit the log (`zyth sum-audit`)
        "/records" => {
            let g = log.lock().unwrap();
            let start: usize = query(q, "start").and_then(|v| v.parse().ok()).unwrap_or(0).min(g.records.len());
            let end: usize = query(q, "end").and_then(|v| v.parse().ok()).unwrap_or(g.records.len()).clamp(start, g.records.len()).min(start + 1000);
            let lines: Vec<String> = g.records[start..end].iter().map(|r| r.line()).collect();
            send_json(&mut s, "200 OK", serde_json::json!({ "start": start, "records": lines, "tree": g.head() }))
        }
        "/" => {
            let (size, key) = {
                let g = log.lock().unwrap();
                (g.leaves.len(), hex(g.key.public_key().as_ref()))
            };
            send_json(&mut s, "200 OK", serde_json::json!({ "log": "zyth checksum log", "size": size, "key": key }))
        }
        _ => send_json(&mut s, "404 Not Found", serde_json::json!({ "error": "no such endpoint" })),
    }
}

/// `zyth sum-server --dir D [--port P] [--bind A] [--allow-local]`, and `zyth sum-server --dir
/// D keygen` (makes the signing key, prints the public one).
pub fn server(args: &[String]) -> ExitCode {
    let opt = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let Some(dir) = opt("--dir").map(PathBuf::from) else {
        eprintln!("usage: zyth sum-server --dir DIR [--port 8091] [--bind 127.0.0.1] [--allow-local]   |   zyth sum-server --dir DIR keygen");
        return ExitCode::from(2);
    };
    let _ = std::fs::create_dir_all(&dir);
    let key_file = dir.join("key.p8");
    if args.iter().any(|a| a == "keygen") {
        if key_file.exists() {
            eprintln!("error: {} exists (a log keeps its key)", key_file.display());
            return ExitCode::from(1);
        }
        let rng = ring::rand::SystemRandom::new();
        let p8 = Ed25519KeyPair::generate_pkcs8(&rng).expect("key generation");
        if let Err(e) = write_private(&key_file, p8.as_ref()) {
            eprintln!("error: cannot write {}: {e}", key_file.display());
            return ExitCode::from(1);
        }
        let k = Ed25519KeyPair::from_pkcs8(p8.as_ref()).unwrap();
        println!("{}", hex(k.public_key().as_ref()));
        return ExitCode::SUCCESS;
    }
    let key = match std::fs::read(&key_file).ok().and_then(|b| Ed25519KeyPair::from_pkcs8(&b).ok()) {
        Some(k) => k,
        None => {
            eprintln!("error: no signing key in {} (make one with `zyth sum-server --dir {} keygen`)", key_file.display(), dir.display());
            return ExitCode::from(1);
        }
    };
    let text = std::fs::read_to_string(dir.join("records")).unwrap_or_default();
    let records: Vec<Record> = text.lines().filter_map(Record::parse).collect();
    let leaves = records.iter().map(|r| leaf(r.line().as_bytes())).collect();
    let port: u16 = opt("--port").and_then(|p| p.parse().ok()).unwrap_or(8091);
    let bind = opt("--bind").unwrap_or_else(|| "127.0.0.1".into());
    let allow_local = args.iter().any(|a| a == "--allow-local");
    let l = match TcpListener::bind((bind.as_str(), port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: cannot listen on {bind}:{port}: {e}");
            return ExitCode::from(1);
        }
    };
    eprintln!("zyth checksum log on http://{} ({} record(s), key {})", l.local_addr().map(|a| a.to_string()).unwrap_or_default(), records.len(), hex(key.public_key().as_ref()));
    let log = Arc::new(Mutex::new(Log { dir, records, leaves, key, allow_local, recent: Default::default() }));
    for c in l.incoming().flatten() {
        let log = log.clone();
        std::thread::spawn(move || handle(&log, c));
    }
    ExitCode::SUCCESS
}

fn write_private(p: &Path, data: &[u8]) -> std::io::Result<()> {
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(p)?.write_all(data)
}

// ------------------------------------------------------------------ the client

/// The log to check against: (base URL, public key), or None when checks are off.
fn log_config() -> Option<(String, Vec<u8>)> {
    let url = std::env::var("ZYTH_SUMDB").unwrap_or_else(|_| DEFAULT_LOG.into());
    if url == "off" || url.is_empty() {
        return None;
    }
    let key = std::env::var("ZYTH_SUMDB_KEY").unwrap_or_else(|_| ZYTH_LOG_KEY.into());
    Some((url.trim_end_matches('/').to_string(), unhex(&key)?))
}

fn get(url: &str) -> Result<serde_json::Value, String> {
    let (code, body) = crate::request("GET", url, &[], &[], std::time::Duration::from_secs(90))?;
    let v: serde_json::Value = serde_json::from_slice(&body).map_err(|_| format!("the checksum log answered {code} with something that is not JSON"))?;
    if code != 200 {
        return Err(v["error"].as_str().unwrap_or("error").to_string());
    }
    Ok(v)
}

fn percent(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

/// A signed tree head, checked.
fn checked_head(v: &serde_json::Value, key: &[u8]) -> Result<(usize, H, u64), String> {
    let size = v["size"].as_u64().ok_or("no tree size")? as usize;
    let r = v["root"].as_str().and_then(h32).ok_or("no tree root")?;
    let time = v["time"].as_u64().unwrap_or(0);
    let sig = v["sig"].as_str().and_then(unhex).ok_or("no signature")?;
    UnparsedPublicKey::new(&ED25519, key).verify(head_text(size, &r, time).as_bytes(), &sig).map_err(|_| "the tree head is not signed by the log's key".to_string())?;
    Ok((size, r, time))
}

/// The last tree head this computer saw from the log (`<cache>/sumdb/<log>/tree`).
fn seen_path(log: &str) -> PathBuf {
    let slug: String = log.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    crate::cache_dir().join("sumdb").join(slug).join("tree")
}

/// Check version `version` of `url` (tag, commit, and the hash computed here) against the
/// log. Ok(Some(time)): the log has it, with the time it first saw it; Ok(None): checks are
/// off or the log could not be reached (said once). Err: a mismatch, or a log that does not
/// prove what it says.
pub fn check(url: &str, version: &str, tag: &str, commit: &str, hash: &str) -> Result<Option<u64>, String> {
    let Some((log, key)) = log_config() else { return Ok(None) };
    let v = match get(&format!("{log}/lookup?url={}&version={}&tag={}", percent(url), percent(version), percent(tag))) {
        Ok(v) => v,
        Err(e) => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| eprintln!("warning: the checksum log {log} was not reached ({e}); versions are checked against zyth.lock only (ZYTH_SUMDB=off hides this)"));
            return Ok(None);
        }
    };
    let (size, r, _) = checked_head(&v["tree"], &key)?;
    let line = v["record"].as_str().ok_or("the log sent no record")?;
    let rec = Record::parse(line).ok_or("the log sent a bad record")?;
    let index = v["index"].as_u64().ok_or("the log sent no index")? as usize;
    let proof: Vec<H> = v["proof"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().and_then(h32)).collect()).unwrap_or_default();
    if !verify_inclusion(index, size, leaf(line.as_bytes()), &proof, &r) {
        return Err(format!("the checksum log {log} sent a record it cannot prove is in its tree"));
    }
    // the log has not rewritten what it showed this computer before
    let seen = seen_path(&log);
    if let Some((n0, r0)) = std::fs::read_to_string(&seen).ok().and_then(|s| {
        let mut it = s.split_whitespace();
        Some((it.next()?.parse::<usize>().ok()?, h32(it.next()?)?))
    }) {
        if n0 > size {
            return Err(format!("the checksum log {log} is smaller ({size}) than it was ({n0}): it lost or hid records"));
        }
        if n0 < size || r0 != r {
            let c = get(&format!("{log}/consistency?old={n0}"))?;
            let (size2, r2, _) = checked_head(&c["tree"], &key)?;
            let cp: Vec<H> = c["proof"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().and_then(h32)).collect()).unwrap_or_default();
            // (the log may have grown again between the two answers)
            if !verify_consistency(n0, size2, &r0, &r2, &cp) {
                return Err(format!("the checksum log {log} does not extend the tree this computer saw before (size {n0}): it changed its past"));
            }
        }
    }
    check_witnesses(&log, &key, size, &r)?;
    let _ = std::fs::create_dir_all(seen.parent().unwrap());
    let _ = std::fs::write(&seen, format!("{size} {}\n", hex(&r)));
    if rec.url != url || rec.version != version {
        return Err(format!("the checksum log {log} answered for {} {}, not {url} {version}", rec.url, rec.version));
    }
    if rec.commit != commit || rec.hash != hash {
        return Err(format!(
            "{url} {version} differs from the checksum log {log}: the log has commit {} with files {}, this download is commit {commit} with files {hash}. The tag was moved or the files changed after the version was first published; do not use it until you know why",
            rec.commit, rec.hash
        ));
    }
    Ok(Some(rec.time))
}

// ------------------------------------------------------------------ auditors and witnesses

/// What a witness signs: the log (by its key) and one of its tree heads.
fn cosign_text(log_key: &[u8], size: usize, root: &H, time: u64) -> String {
    format!("zyth sum witness v1\n{}\n{size}\n{}\n{time}\n", hex(log_key), hex(root))
}

struct Audit {
    log: String,
    log_key: Vec<u8>,
    dir: PathBuf,
    recheck: bool,
    allow_local: bool,
    records: Vec<Record>,
    leaves: Vec<H>,
    key: Ed25519KeyPair,
    /// the latest head checked and cosigned: { size, root, time, sig, cosig }
    cosigned: Option<serde_json::Value>,
    /// what the audit found wrong, newest last (also in `<dir>/findings`)
    findings: Vec<String>,
}

impl Audit {
    fn found(&mut self, what: String) {
        eprintln!("FINDING: {what}");
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("findings")) {
            let _ = writeln!(f, "{} {what}", now());
        }
        self.findings.push(what);
    }

    /// Read the log up to its current signed head and check it; cosign the head when it holds.
    fn round(&mut self) -> Result<String, String> {
        let head = get(&format!("{}/tree", self.log))?;
        let (size, r, time) = checked_head(&head, &self.log_key)?;
        if size < self.records.len() {
            let what = format!("the log is smaller ({size}) than when this auditor last read it ({}): it lost or hid records", self.records.len());
            self.found(what.clone());
            return Err(what);
        }
        // the records this auditor has not read yet, up to the signed head
        let mut new: Vec<Record> = Vec::new();
        let mut at = self.records.len();
        while at < size {
            let v = get(&format!("{}/records?start={at}&end={size}", self.log))?;
            let lines: Vec<&str> = v["records"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
            if lines.is_empty() {
                return Err(format!("the log sent no records from {at}"));
            }
            for l in lines.into_iter().take(size - at) {
                let rec = Record::parse(l).ok_or_else(|| format!("the log sent a bad record: {l}"))?;
                new.push(rec);
                at += 1;
            }
        }
        let mut leaves = self.leaves.clone();
        leaves.extend(new.iter().map(|r| leaf(r.line().as_bytes())));
        if root(&leaves) != r {
            let what = format!("the log's signed tree head (size {size}) does not match its records: it changed its past or signs a tree it does not show");
            self.found(what.clone());
            return Err(what);
        }
        // one hash per version: the log never takes a version twice
        let mut seen: std::collections::HashMap<(String, String), String> = self.records.iter().map(|r| ((r.url.clone(), r.version.clone()), r.hash.clone())).collect();
        for rec in &new {
            if let Some(h) = seen.insert((rec.url.clone(), rec.version.clone()), rec.hash.clone()) {
                self.found(format!("the log has {} {} twice (files {h} and {})", rec.url, rec.version, rec.hash));
            }
        }
        // each new version's hash, worked out here from its origin
        let mut rechecked = 0;
        if self.recheck {
            for rec in &new {
                match from_origin(&rec.url, &rec.tag, self.allow_local) {
                    Ok((commit, hash)) if commit == rec.commit && hash == rec.hash => rechecked += 1,
                    Ok((commit, hash)) => self.found(format!(
                        "{} {} (tag {}): the log has commit {} with files {}, its origin has commit {commit} with files {hash} (the log lied, or the tag moved after it was logged)",
                        rec.url, rec.version, rec.tag, rec.commit, rec.hash
                    )),
                    Err(e) => eprintln!("note: cannot read {} at {} to recheck it: {e}", rec.url, rec.tag),
                }
            }
        }
        // keep what was checked, then cosign
        if !new.is_empty() {
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("records")).map_err(|e| e.to_string())?;
            for rec in &new {
                writeln!(f, "{}", rec.line()).map_err(|e| e.to_string())?;
            }
            f.sync_all().map_err(|e| e.to_string())?;
        }
        self.records.extend(new.iter().cloned());
        self.leaves = leaves;
        let cosig = self.key.sign(cosign_text(&self.log_key, size, &r, time).as_bytes());
        let v = serde_json::json!({ "size": size, "root": hex(&r), "time": time, "sig": head["sig"], "cosig": hex(cosig.as_ref()), "witness": hex(self.key.public_key().as_ref()) });
        std::fs::write(self.dir.join("cosigned.json"), v.to_string()).map_err(|e| e.to_string())?;
        self.cosigned = Some(v);
        Ok(format!("checked the log at size {size}: {} new record(s), {rechecked} rechecked from their origins, {} finding(s) in all", new.len(), self.findings.len()))
    }
}

fn audit_handle(a: &Mutex<Audit>, mut s: TcpStream) {
    let mut r = BufReader::new(match s.try_clone() {
        Ok(c) => c,
        Err(_) => return,
    });
    let mut line = String::new();
    if r.by_ref().take(8192).read_line(&mut line).is_err() {
        return;
    }
    loop {
        let mut h = String::new();
        if r.by_ref().take(8192).read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" || h == "\n" {
            break;
        }
    }
    let path = line.split_whitespace().nth(1).unwrap_or("/").split('?').next().unwrap_or("/").to_string();
    let g = a.lock().unwrap();
    match path.as_str() {
        "/cosigned" => match &g.cosigned {
            Some(v) => send_json(&mut s, "200 OK", v.clone()),
            None => send_json(&mut s, "503 Service Unavailable", serde_json::json!({ "error": "no tree head checked yet" })),
        },
        "/findings" => send_json(&mut s, "200 OK", serde_json::json!({ "findings": g.findings })),
        "/" => send_json(&mut s, "200 OK", serde_json::json!({ "witness": hex(g.key.public_key().as_ref()), "log": g.log, "size": g.records.len(), "findings": g.findings.len() })),
        _ => send_json(&mut s, "404 Not Found", serde_json::json!({ "error": "no such endpoint" })),
    }
}

/// `zyth sum-audit --log URL --log-key HEX --dir DIR [--every SECONDS] [--port P] [--bind A]
/// [--once] [--no-recheck] [--allow-local]`, and `zyth sum-audit --dir DIR keygen`.
pub fn audit(args: &[String]) -> ExitCode {
    let opt = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let usage = "usage: zyth sum-audit --log URL [--log-key HEX] --dir DIR [--every 300] [--port 8092] [--bind 127.0.0.1] [--once] [--no-recheck] [--allow-local]   |   zyth sum-audit --dir DIR keygen";
    let Some(dir) = opt("--dir").map(PathBuf::from) else {
        eprintln!("{usage}");
        return ExitCode::from(2);
    };
    let _ = std::fs::create_dir_all(&dir);
    let key_file = dir.join("witness.p8");
    if args.iter().any(|a| a == "keygen") {
        if key_file.exists() {
            eprintln!("error: {} exists (a witness keeps its key)", key_file.display());
            return ExitCode::from(1);
        }
        let p8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).expect("key generation");
        if let Err(e) = write_private(&key_file, p8.as_ref()) {
            eprintln!("error: cannot write {}: {e}", key_file.display());
            return ExitCode::from(1);
        }
        println!("{}", hex(Ed25519KeyPair::from_pkcs8(p8.as_ref()).unwrap().public_key().as_ref()));
        return ExitCode::SUCCESS;
    }
    let Some(log) = opt("--log").map(|l| l.trim_end_matches('/').to_string()) else {
        eprintln!("{usage}");
        return ExitCode::from(2);
    };
    let Some(log_key) = unhex(&opt("--log-key").unwrap_or_else(|| ZYTH_LOG_KEY.into())).filter(|k| k.len() == 32) else {
        eprintln!("error: --log-key is the log's Ed25519 public key in hex (64 characters)");
        return ExitCode::from(2);
    };
    let Some(key) = std::fs::read(&key_file).ok().and_then(|b| Ed25519KeyPair::from_pkcs8(&b).ok()) else {
        eprintln!("error: no witness key in {} (make one with `zyth sum-audit --dir {} keygen`)", key_file.display(), dir.display());
        return ExitCode::from(1);
    };
    let records: Vec<Record> = std::fs::read_to_string(dir.join("records")).unwrap_or_default().lines().filter_map(Record::parse).collect();
    let leaves = records.iter().map(|r| leaf(r.line().as_bytes())).collect();
    let cosigned = std::fs::read_to_string(dir.join("cosigned.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
    let findings = std::fs::read_to_string(dir.join("findings")).unwrap_or_default().lines().map(|l| l.split_once(' ').map(|(_, w)| w.to_string()).unwrap_or_default()).collect();
    let a = Arc::new(Mutex::new(Audit { log, log_key, dir, recheck: !args.iter().any(|x| x == "--no-recheck"), allow_local: args.iter().any(|x| x == "--allow-local"), records, leaves, key, cosigned, findings }));
    if args.iter().any(|x| x == "--once") {
        let mut g = a.lock().unwrap();
        let before = g.findings.len();
        return match g.round() {
            Ok(m) => {
                println!("{m}");
                if g.findings.len() > before { ExitCode::from(1) } else { ExitCode::SUCCESS }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(1)
            }
        };
    }
    let every: u64 = opt("--every").and_then(|v| v.parse().ok()).unwrap_or(300).max(1);
    let port: u16 = opt("--port").and_then(|p| p.parse().ok()).unwrap_or(8092);
    let bind = opt("--bind").unwrap_or_else(|| "127.0.0.1".into());
    let l = match TcpListener::bind((bind.as_str(), port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: cannot listen on {bind}:{port}: {e}");
            return ExitCode::from(1);
        }
    };
    {
        let g = a.lock().unwrap();
        eprintln!("zyth checksum-log witness on http://{} for {} (witness key {})", l.local_addr().map(|a| a.to_string()).unwrap_or_default(), g.log, hex(g.key.public_key().as_ref()));
    }
    let ar = a.clone();
    std::thread::spawn(move || loop {
        // (the round holds the lock: requests wait for it rather than see half a round)
        let r = ar.lock().unwrap().round();
        match r {
            Ok(m) => eprintln!("{m}"),
            Err(e) => eprintln!("error: {e}"),
        }
        std::thread::sleep(std::time::Duration::from_secs(every));
    });
    for c in l.incoming().flatten() {
        let a = a.clone();
        std::thread::spawn(move || audit_handle(&a, c));
    }
    ExitCode::SUCCESS
}

/// Witnesses clients ask (`ZYTH_SUMDB_WITNESSES=URL=KEY,URL=KEY`).
fn witnesses() -> Vec<(String, Vec<u8>)> {
    std::env::var("ZYTH_SUMDB_WITNESSES")
        .unwrap_or_default()
        .split(',')
        .filter_map(|w| {
            let (u, k) = w.trim().rsplit_once('=')?;
            Some((u.trim_end_matches('/').to_string(), unhex(k).filter(|k| k.len() == 32)?))
        })
        .collect()
}

/// The log's current head, proved to extend the tree (`n`, `r`).
fn extends(log: &str, key: &[u8], n: usize, r: &H) -> Result<(usize, H), String> {
    let c = get(&format!("{log}/consistency?old={n}"))?;
    let (size, cr, _) = checked_head(&c["tree"], key)?;
    let cp: Vec<H> = c["proof"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().and_then(h32)).collect()).unwrap_or_default();
    if n > 0 && !verify_consistency(n, size, r, &cr, &cp) {
        return Err(format!("the checksum log {log} does not extend the tree of size {n} it showed before"));
    }
    Ok((size, cr))
}

/// The tree (`size`, `r`) this computer got is one history with each witness's latest
/// cosigned head: both extend to the log's current head.
fn check_witnesses(log: &str, key: &[u8], size: usize, r: &H) -> Result<(), String> {
    for (w, wkey) in witnesses() {
        let v = match get(&format!("{w}/cosigned")) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("warning: the checksum-log witness {w} was not reached ({e})");
                continue;
            }
        };
        let (ws, wr, wt) = checked_head(&v, key).map_err(|e| format!("witness {w}: {e}"))?;
        let cosig = v["cosig"].as_str().and_then(unhex).ok_or_else(|| format!("witness {w} sent no cosignature"))?;
        UnparsedPublicKey::new(&ED25519, &wkey).verify(cosign_text(key, ws, &wr, wt).as_bytes(), &cosig).map_err(|_| format!("the head witness {w} sent is not cosigned by its key"))?;
        if ws == size {
            if wr != *r {
                return Err(format!("the checksum log {log} showed this computer another tree (size {size}) than it showed witness {w}: it keeps two histories"));
            }
            continue;
        }
        let mut a = extends(log, key, size, r)?;
        let b = extends(log, key, ws, &wr).map_err(|_| format!("the checksum log {log} showed witness {w} a tree (size {ws}) that its current tree does not extend: it keeps two histories"))?;
        if a != b {
            // (the log grew between the two answers)
            a = extends(log, key, size, r)?;
        }
        if a != b && a.0 == b.0 {
            return Err(format!("the checksum log {log} gives this computer and witness {w} different trees: it keeps two histories"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proofs_verify_for_every_size() {
        let leaves: Vec<H> = (0..37u8).map(|i| leaf(&[i])).collect();
        for n in 1..=leaves.len() {
            let t = &leaves[..n];
            let r = root(t);
            for m in 0..n {
                assert!(verify_inclusion(m, n, t[m], &inclusion(m, t), &r), "inclusion {m} of {n}");
                assert!(!verify_inclusion(m, n, leaf(b"other"), &inclusion(m, t), &r));
            }
            for m in 1..=n {
                let p = consistency(m, t);
                assert!(verify_consistency(m, n, &root(&leaves[..m]), &r, &p), "consistency {m} -> {n}");
                if m < n {
                    assert!(!verify_consistency(m, n, &leaf(b"x"), &r, &p));
                }
            }
        }
    }
}
