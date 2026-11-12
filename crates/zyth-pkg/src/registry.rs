//! The Zyth package registry (registry.zyth.io): an index of package names to the Git
//! repositories they live in, so `zyth add geom` works without a URL. Packages stay in Git
//! (spec 15.5); the registry holds only the name, the repository and a description, and every
//! version is still checked against the checksum log (sumdb.rs), which the registry does not
//! run.
//!
//! - `zyth publish --registry` (after tagging) registers the project's name and repository
//!   with the publisher's token (`zyth login --registry URL --token T`). The registry reads the
//!   repository's newest version tag itself and accepts the name only if that version's
//!   `zyth.toml` says it; a name belongs to the token that first registered it.
//! - `zyth add <name>` asks the registry for the repository and continues as `zyth add
//!   git:<repository>`; `zyth search <words>` lists matching packages.
//! - `zyth registry-server --dir D [--port P]` runs one; `... adduser <name>` makes a
//!   publisher's token. `ZYTH_REGISTRY` points the client at another.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

const DEFAULT_REGISTRY: &str = "https://registry.zyth.io";

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn sha256_hex(b: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

fn valid_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= 64 && n.chars().next().is_some_and(|c| c.is_ascii_lowercase()) && n.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

// ------------------------------------------------------------------ the server

struct Reg {
    dir: PathBuf,
    /// name -> { url, description, owner (token hash), added, updated }
    names: serde_json::Map<String, serde_json::Value>,
    /// token hash -> publisher name
    users: serde_json::Map<String, serde_json::Value>,
    allow_local: bool,
}

impl Reg {
    fn save(&self) -> std::io::Result<()> {
        let tmp = self.dir.join("names.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&self.names).unwrap_or_default())?;
        std::fs::rename(tmp, self.dir.join("names.json"))
    }
}

fn json(s: &mut TcpStream, status: &str, v: serde_json::Value) {
    let body = v.to_string();
    let _ = write!(s, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{body}", body.len());
}

/// The newest version tag of `url` and the `[project]` name and description in its zyth.toml.
fn read_package(url: &str, allow_local: bool) -> Result<(String, String, String), String> {
    let local = url.starts_with('/') || url.contains(":\\");
    if local && !allow_local {
        return Err("the registry takes https:// repositories".into());
    }
    let remote = if local || url.starts_with("https://") { url.to_string() } else { format!("https://{url}") };
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
    let tags = git(&["ls-remote", "--tags", &remote], None)?;
    let newest = tags
        .lines()
        .filter_map(|l| l.split('\t').nth(1)?.strip_prefix("refs/tags/").map(|t| t.trim_end_matches("^{}").to_string()))
        .filter_map(|t| {
            let v: Vec<u64> = t.trim_start_matches('v').split('.').map_while(|p| p.parse().ok()).collect();
            (v.len() == 3).then_some((v, t))
        })
        .max()
        .map(|(_, t)| t)
        .ok_or("the repository has no version tags (v1.2.3); publish one with `zyth publish` first")?;
    let tmp = std::env::temp_dir().join(format!("zyth-reg-{}-{}", std::process::id(), now()));
    let _ = std::fs::remove_dir_all(&tmp);
    let r = (|| {
        git(&["clone", "-q", "--depth", "1", "--branch", &newest, &remote, &tmp.to_string_lossy()], None)?;
        let toml = std::fs::read_to_string(tmp.join("zyth.toml")).map_err(|_| format!("{newest} has no zyth.toml"))?;
        let field = |k: &str| crate::project_field(&toml, k).unwrap_or_default();
        Ok((newest.clone(), field("name"), field("description")))
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    r
}

fn handle(reg: &Mutex<Reg>, mut s: TcpStream) {
    let mut r = BufReader::new(match s.try_clone() {
        Ok(c) => c,
        Err(_) => return,
    });
    let mut line = String::new();
    if r.by_ref().take(8192).read_line(&mut line).is_err() {
        return;
    }
    let (mut auth, mut len) = (String::new(), 0usize);
    for _ in 0..100 {
        let mut h = String::new();
        if r.by_ref().take(8192).read_line(&mut h).unwrap_or(0) == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "authorization" => auth = v.trim().trim_start_matches("Bearer ").to_string(),
                "content-length" => len = v.trim().parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    let mut it = line.split_whitespace();
    let (Some(method), Some(target)) = (it.next(), it.next()) else { return };
    let (path, q) = target.split_once('?').unwrap_or((target, ""));
    match (method, path) {
        ("GET", "/") => {
            let n = reg.lock().unwrap().names.len();
            json(&mut s, "200 OK", serde_json::json!({ "registry": "zyth packages", "packages": n }))
        }
        ("GET", "/search") => {
            let words: Vec<String> = q.split('&').find_map(|kv| kv.strip_prefix("q=")).map(crate::percent_decode).unwrap_or_default().to_lowercase().split_whitespace().map(String::from).collect();
            let g = reg.lock().unwrap();
            let mut hits: Vec<serde_json::Value> = g
                .names
                .iter()
                .filter(|(n, v)| {
                    let text = format!("{n} {}", v["description"].as_str().unwrap_or("")).to_lowercase();
                    words.iter().all(|w| text.contains(w.as_str()))
                })
                .map(|(n, v)| serde_json::json!({ "name": n, "url": v["url"], "description": v["description"] }))
                .collect();
            hits.truncate(50);
            json(&mut s, "200 OK", serde_json::json!({ "results": hits }))
        }
        ("GET", p) if p.starts_with("/names/") => {
            let name = &p["/names/".len()..];
            let g = reg.lock().unwrap();
            match g.names.get(name) {
                Some(v) => json(&mut s, "200 OK", serde_json::json!({ "name": name, "url": v["url"], "description": v["description"] })),
                None => json(&mut s, "404 Not Found", serde_json::json!({ "error": format!("no package `{name}` in the registry") })),
            }
        }
        ("POST", p) if p.starts_with("/names/") => {
            let name = p["/names/".len()..].to_string();
            let mut body = vec![0u8; len.min(4096)];
            let _ = r.read_exact(&mut body);
            let url = serde_json::from_slice::<serde_json::Value>(&body).ok().and_then(|v| v["url"].as_str().map(String::from)).unwrap_or_default();
            let th = sha256_hex(auth.as_bytes());
            let (known, allow_local) = {
                let g = reg.lock().unwrap();
                (g.users.contains_key(&th), g.allow_local)
            };
            if auth.is_empty() || !known {
                return json(&mut s, "401 Unauthorized", serde_json::json!({ "error": "a publisher's token is needed (`zyth login --registry URL --token T`)" }));
            }
            if !valid_name(&name) {
                return json(&mut s, "400 Bad Request", serde_json::json!({ "error": "a package name is lower-case letters, digits and _ (starting with a letter), at most 64" }));
            }
            if let Some(v) = reg.lock().unwrap().names.get(&name) {
                if v["owner"].as_str() != Some(th.as_str()) {
                    return json(&mut s, "409 Conflict", serde_json::json!({ "error": format!("`{name}` belongs to another publisher") }));
                }
            }
            let (tag, declared, desc) = match read_package(&url, allow_local) {
                Ok(x) => x,
                Err(e) => return json(&mut s, "400 Bad Request", serde_json::json!({ "error": format!("cannot read {url}: {e}") })),
            };
            if declared != name {
                return json(&mut s, "400 Bad Request", serde_json::json!({ "error": format!("{url} at {tag} is the package `{declared}`, not `{name}`") }));
            }
            let mut g = reg.lock().unwrap();
            let added = g.names.get(&name).and_then(|v| v["added"].as_u64()).unwrap_or_else(now);
            g.names.insert(name.clone(), serde_json::json!({ "url": url, "description": desc, "owner": th, "added": added, "updated": now(), "tag": tag }));
            match g.save() {
                Ok(()) => json(&mut s, "200 OK", serde_json::json!({ "name": name, "url": url, "tag": tag })),
                Err(e) => json(&mut s, "500 Internal Server Error", serde_json::json!({ "error": format!("cannot save: {e}") })),
            }
        }
        _ => json(&mut s, "404 Not Found", serde_json::json!({ "error": "no such endpoint" })),
    }
}

/// `zyth registry-server --dir D [--port P] [--bind A] [--allow-local]`;
/// `zyth registry-server --dir D adduser <name>` prints a new publisher's token.
pub fn server(args: &[String]) -> ExitCode {
    let opt = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let Some(dir) = opt("--dir").map(PathBuf::from) else {
        eprintln!("usage: zyth registry-server --dir DIR [--port 8092] [--bind 127.0.0.1] [--allow-local]   |   zyth registry-server --dir DIR adduser NAME");
        return ExitCode::from(2);
    };
    let _ = std::fs::create_dir_all(&dir);
    let users_file = dir.join("users.json");
    let mut users: serde_json::Map<String, serde_json::Value> = std::fs::read(&users_file).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    if let Some(i) = args.iter().position(|a| a == "adduser") {
        let Some(name) = args.get(i + 1) else {
            eprintln!("usage: zyth registry-server --dir DIR adduser NAME");
            return ExitCode::from(2);
        };
        let mut b = [0u8; 24];
        if getrandom(&mut b).is_err() {
            eprintln!("error: no random numbers");
            return ExitCode::from(1);
        }
        let token: String = b.iter().map(|x| format!("{x:02x}")).collect();
        users.insert(sha256_hex(token.as_bytes()), serde_json::json!({ "name": name, "created": now() }));
        if let Err(e) = std::fs::write(&users_file, serde_json::to_vec_pretty(&users).unwrap_or_default()) {
            eprintln!("error: cannot write {}: {e}", users_file.display());
            return ExitCode::from(1);
        }
        eprintln!("added publisher `{name}`; the token (shown once; only its hash is kept):");
        println!("{token}");
        return ExitCode::SUCCESS;
    }
    let names = std::fs::read(dir.join("names.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    let port: u16 = opt("--port").and_then(|p| p.parse().ok()).unwrap_or(8092);
    let bind = opt("--bind").unwrap_or_else(|| "127.0.0.1".into());
    let l = match TcpListener::bind((bind.as_str(), port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: cannot listen on {bind}:{port}: {e}");
            return ExitCode::from(1);
        }
    };
    let allow_local = args.iter().any(|a| a == "--allow-local");
    let reg = Arc::new(Mutex::new(Reg { dir, names, users, allow_local }));
    eprintln!("zyth package registry on http://{}", l.local_addr().map(|a| a.to_string()).unwrap_or_default());
    for c in l.incoming().flatten() {
        let reg = reg.clone();
        std::thread::spawn(move || handle(&reg, c));
    }
    ExitCode::SUCCESS
}

fn getrandom(b: &mut [u8]) -> Result<(), ()> {
    use ring::rand::SecureRandom;
    ring::rand::SystemRandom::new().fill(b).map_err(|_| ())
}

// ------------------------------------------------------------------ the client

pub fn registry_url() -> String {
    std::env::var("ZYTH_REGISTRY").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| load_login().0.unwrap_or_else(|| DEFAULT_REGISTRY.into())).trim_end_matches('/').to_string()
}

fn login_path() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(|h| PathBuf::from(h).join(".config"))).unwrap_or_default().join("zyth").join("registry.json")
}

fn load_login() -> (Option<String>, Option<String>) {
    let v: serde_json::Value = std::fs::read(login_path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    (v["server"].as_str().map(String::from), v["token"].as_str().map(String::from))
}

/// `zyth login --registry URL --token T`.
pub fn login(server: &str, token: &str) -> ExitCode {
    let p = login_path();
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    let data = serde_json::json!({ "server": server.trim_end_matches('/'), "token": token }).to_string();
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    match o.open(&p).and_then(|mut f| f.write_all(data.as_bytes())) {
        Ok(()) => {
            eprintln!("logged in to the registry {server} (token saved to {})", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: cannot write {}: {e}", p.display());
            ExitCode::from(1)
        }
    }
}

/// The repository a registered name points to.
pub fn resolve(name: &str) -> Result<String, String> {
    let base = registry_url();
    let (code, body) = crate::request("GET", &format!("{base}/names/{name}"), &[], &[], std::time::Duration::from_secs(20)).map_err(|e| format!("cannot reach the registry {base}: {e}"))?;
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    if code == 404 {
        return Err(format!("no package `{name}` in the registry {base} (`zyth search {name}` looks for close ones; `zyth add git:<repository>` takes any repository)"));
    }
    v["url"].as_str().map(String::from).ok_or_else(|| format!("the registry {base} answered {code}: {}", v["error"].as_str().unwrap_or("")))
}

/// `zyth publish --registry`: register the project's name and repository.
pub fn register(name: &str, url: &str) -> Result<String, String> {
    let base = registry_url();
    let token = load_login().1.ok_or("no registry login: `zyth login --registry https://registry.zyth.io --token <token>`")?;
    let body = serde_json::json!({ "url": url }).to_string();
    let (code, resp) = crate::request("POST", &format!("{base}/names/{name}"), &[("Authorization".into(), format!("Bearer {token}")), ("Content-Type".into(), "application/json".into())], body.as_bytes(), std::time::Duration::from_secs(90)).map_err(|e| format!("cannot reach the registry {base}: {e}"))?;
    let v: serde_json::Value = serde_json::from_slice(&resp).unwrap_or_default();
    if code != 200 {
        return Err(format!("the registry {base} refused: {}", v["error"].as_str().unwrap_or("error")));
    }
    Ok(format!("registered `{name}` -> {url} ({}) at {base}", v["tag"].as_str().unwrap_or("")))
}

/// `zyth search <words>`.
pub fn search(args: &[String]) -> ExitCode {
    let base = registry_url();
    let q: String = args.join(" ");
    let enc: String = q.bytes().map(|b| if b.is_ascii_alphanumeric() { (b as char).to_string() } else { format!("%{b:02X}") }).collect();
    match crate::request("GET", &format!("{base}/search?q={enc}"), &[], &[], std::time::Duration::from_secs(20)) {
        Ok((200, body)) => {
            let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
            let rs = v["results"].as_array().cloned().unwrap_or_default();
            if rs.is_empty() {
                println!("no packages match `{q}` in {base}");
            }
            for r in rs {
                println!("{:<20} {}  {}", r["name"].as_str().unwrap_or(""), r["description"].as_str().unwrap_or(""), r["url"].as_str().unwrap_or(""));
            }
            ExitCode::SUCCESS
        }
        Ok((c, _)) => {
            eprintln!("error: the registry {base} answered {c}");
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("error: cannot reach the registry {base}: {e}");
            ExitCode::from(1)
        }
    }
}
