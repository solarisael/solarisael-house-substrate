use chrono::{Local, NaiveDate};
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sqlx::{postgres::{PgConnectOptions, PgPoolOptions}, PgPool};
use std::{collections::BTreeSet, env, fs, io, path::Path, process::Command, str::FromStr};
use thiserror::Error;
use uuid::Uuid;

const DEFAULT_EMBED_URL: &str = "http://127.0.0.1:11435/api/embed";
const DEFAULT_EMBED_MODEL: &str = "hf.co/zenmagnets/Nemotron-3-Embed-1B-Q4_K_M-GGUF:latest";
const EMBED_DIMENSION: usize = 2048;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("invalid request: {0}")] Invalid(String),
    #[error("configuration error: {0}")] Config(String),
    #[error("database error: {0}")] Database(#[from] sqlx::Error),
    #[error("embedding error: {0}")] Embedding(String),
    #[error("protocol error: {0}")] Protocol(String),
    #[error("io error: {0}")] Io(#[from] io::Error),
}

#[derive(Clone)]
pub struct Config {
    pub database_url: String,
    pub embed_url: Option<String>,
    pub embed_model: String,
    pub embed_dimension: usize,
    pub embed_required: bool,
    pub test_embedding_disabled: bool,
}

fn dotenv_value(key: &str) -> Option<String> {
    if let Ok(v) = env::var(key) { return Some(v); }
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(".env");
    let text = fs::read_to_string(path).ok()?;
    text.lines().find_map(|line| {
        let line = line.trim();
        let (k, v) = line.split_once('=')?;
        if k.trim() == key { Some(v.trim().trim_matches('"').trim_matches('\'').to_string()) } else { None }
    })
}

impl Config {
    pub fn from_env() -> Result<Self, AppError> {
        let database_url = env::var("SOLARISAEL_SUBSTRATE_TEST_DATABASE_URL")
            .ok()
            .or_else(|| dotenv_value("DATABASE_URL"))
            .or_else(|| {
                let host = dotenv_value("PGHOST")?;
                let port = dotenv_value("PGPORT").unwrap_or_else(|| "5432".into()).parse().ok()?;
                let user = dotenv_value("PGUSER")?;
                let password = dotenv_value("PGPASSWORD")?;
                let database = dotenv_value("PGDATABASE")?;
                let mut url = reqwest::Url::parse("postgres://localhost").ok()?;
                url.set_host(Some(&host)).ok()?;
                url.set_port(Some(port)).ok()?;
                url.set_username(&user).ok()?;
                url.set_password(Some(&password)).ok()?;
                url.set_path(&database);
                Some(url.to_string())
            })
            .ok_or_else(|| AppError::Config("DATABASE_URL or complete PG* variables required".into()))?;
        let embed_url = Some(dotenv_value("SOLARISAEL_EMBED_URL").unwrap_or_else(|| DEFAULT_EMBED_URL.into()));
        let embed_dimension = dotenv_value("SOLARISAEL_EMBED_DIMENSION").unwrap_or_else(|| EMBED_DIMENSION.to_string()).parse().map_err(|_| AppError::Config("SOLARISAEL_EMBED_DIMENSION must be an integer".into()))?;
        if embed_dimension != EMBED_DIMENSION { return Err(AppError::Config("embedding dimension must be 2048 for migration 0002".into())); }
        let test_embedding_disabled = dotenv_value("SOLARISAEL_TEST_DISABLE_EMBEDDING").as_deref() == Some("1");
        Ok(Self { database_url, embed_model: dotenv_value("SOLARISAEL_EMBED_MODEL").unwrap_or_else(|| DEFAULT_EMBED_MODEL.into()), embed_dimension, embed_required: !test_embedding_disabled, test_embedding_disabled, embed_url })
    }
    pub async fn pool(&self) -> Result<PgPool, AppError> {
        let options = PgConnectOptions::from_str(&self.database_url)
            .map_err(|e| AppError::Config(format!("invalid database configuration: {e}")))?;
        let pool = PgPoolOptions::new().max_connections(4).connect_with(options).await?;
        let shape: String = sqlx::query_scalar("SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid WHERE c.relname='memory_chunks' AND a.attname='body_embedding' AND NOT a.attisdropped")
        .fetch_optional(&pool).await?.ok_or_else(|| AppError::Config("memory_chunks.body_embedding is missing; apply migration 0002".into()))?;
        if shape != "vector(2048)" { return Err(AppError::Config(format!("incompatible embedding schema: {shape}"))); }
        Ok(pool)
    }
}

#[derive(Debug, Deserialize)]
pub struct RememberRequest {
    pub room: String, pub kind: String, pub title: String, pub body: String,
    pub source_path: Option<String>, #[serde(default)] pub threads: Vec<String>,
    #[serde(default)] pub supersedes: Vec<i64>, #[serde(default = "default_backup")] pub backup: bool,
}
fn default_backup() -> bool { true }

#[derive(Debug, Serialize)]
pub struct RememberReceipt { pub memory_id: i64, pub room: String, pub source_path: String, pub durable: bool, pub authority: &'static str, pub warnings: Vec<String> }

impl RememberRequest {
    pub fn validate(&self) -> Result<(), AppError> {
        let room_re = Regex::new(r"^[a-z0-9]+(?:-[a-z0-9]+)*$").unwrap();
        if !room_re.is_match(&self.room) || self.room == "house" { return Err(AppError::Invalid("room must be a lowercase slug and cannot be house".into())); }
        if self.kind != "memory" { return Err(AppError::Invalid("only kind=memory is supported".into())); }
        if self.title.trim().is_empty() { return Err(AppError::Invalid("title must not be empty".into())); }
        if self.body.trim().is_empty() { return Err(AppError::Invalid("body must not be empty".into())); }
        if self.source_path.as_ref().is_some_and(|p| p.trim().is_empty()) { return Err(AppError::Invalid("source_path must not be empty".into())); }
        if self.supersedes.iter().any(|id| *id <= 0) { return Err(AppError::Invalid("supersedes must contain positive IDs".into())); }
        Ok(())
    }
    fn source_path(&self) -> String { self.source_path.clone().unwrap_or_else(|| format!("db-only/{}/{}", self.room, Uuid::new_v4())) }
}

pub async fn remember(pool: &PgPool, cfg: &Config, req: RememberRequest) -> Result<RememberReceipt, AppError> {
    req.validate()?;
    let source_path = req.source_path();
    let threads = normalize_threads(&req.threads);
    let primary_date = Local::now().date_naive();
    let dates = derive_dates(&source_path, primary_date);
    let mut tx = pool.begin().await?;
    let meta = serde_json::json!({"origin":"direct-db-write", "recorded_at": chrono::Utc::now().to_rfc3339()});
    let memory_id: i64 = sqlx::query_scalar(r#"INSERT INTO memories (room,type,date,dates,title,source_path,body,threads,meta) VALUES ($1,'memory',$2,$3,$4,$5,$6,$7,$8) ON CONFLICT (room,source_path) DO UPDATE SET type='memory',date=EXCLUDED.date,dates=EXCLUDED.dates,title=EXCLUDED.title,body=EXCLUDED.body,threads=EXCLUDED.threads,meta=EXCLUDED.meta RETURNING id"#)
        .bind(&req.room).bind(primary_date).bind(&dates).bind(&req.title).bind(&source_path).bind(&req.body).bind(&threads).bind(meta).fetch_one(&mut *tx).await?;
    sqlx::query("DELETE FROM memory_threads WHERE memory_id=$1").bind(memory_id).execute(&mut *tx).await?;
    for thread in &threads { sqlx::query("INSERT INTO memory_threads (memory_id,thread_key) VALUES ($1,$2)").bind(memory_id).bind(thread).execute(&mut *tx).await?; }
    for old_id in req.supersedes.iter().copied().collect::<BTreeSet<_>>() { sqlx::query("UPDATE memories SET superseded_by=$1 WHERE id=$2 AND id<>$1").bind(memory_id).bind(old_id).execute(&mut *tx).await?; }
    sqlx::query("DELETE FROM memory_chunks WHERE memory_id=$1").bind(memory_id).execute(&mut *tx).await?;
    let chunks = chunk_body(&req.body);
    let mut warnings = Vec::new();
    if cfg.test_embedding_disabled {
        warnings.push("embedding disabled for isolated test; chunks cleared".into());
    } else {
        let url = cfg.embed_url.as_deref().ok_or_else(|| AppError::Embedding("embedding endpoint is required".into()))?;
        let vectors = embed(&Client::new(), url, &cfg.embed_model, &chunks, cfg.embed_dimension).await?;
        if vectors.len() != chunks.len() { return Err(AppError::Embedding("embedding count mismatch".into())); }
        for (idx, (text, start, end, heading)) in chunks.iter().enumerate() {
            let vector_text = format!("[{}]", vectors[idx].iter().map(ToString::to_string).collect::<Vec<_>>().join(","));
            sqlx::query("INSERT INTO memory_chunks (memory_id,chunk_index,heading_path,body,char_start,char_end,token_estimate,body_embedding,embedded_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8::vector,NOW())")
                .bind(memory_id).bind(idx as i32).bind(heading).bind(text).bind(*start as i32).bind(*end as i32).bind(token_estimate(text)).bind(vector_text).execute(&mut *tx).await?;
        }
    }
    tx.commit().await?;
    if req.backup { if let Err(e) = run_backup(&cfg.database_url) { warnings.push(format!("backup failed: {e}")); } }
    Ok(RememberReceipt { memory_id: memory_id, room: req.room, source_path, durable: true, authority: "postgres", warnings })
}

fn backup_target(database_url: &str) -> Result<(String, Option<String>), String> {
    let mut url = reqwest::Url::parse(database_url).map_err(|e| format!("invalid database URL: {e}"))?;
    let password = url.password().map(str::to_owned);
    if password.is_some() { url.set_password(None).map_err(|_| "invalid database URL password".to_string())?; }
    Ok((url.to_string(), password))
}

fn build_backup_command(database_url: &str) -> Result<Command, String> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("backup.sh");
    if !script.exists() { return Err(format!("backup.sh not found at {}", script.display())); }
    let (target, password) = backup_target(database_url)?;
    let mut command = Command::new("bash");
    command.arg(&script).env("SOLARISAEL_BACKUP_DATABASE_URL", target);
    if let Some(password) = password { command.env("SOLARISAEL_BACKUP_PASSWORD", password); }
    Ok(command)
}

fn run_backup(database_url: &str) -> Result<(), String> {
    let output = build_backup_command(database_url)?.output().map_err(|e| e.to_string())?;
    if output.status.success() { Ok(()) } else { Err(String::from_utf8_lossy(&output.stderr).trim().to_string()) }
}

fn normalize_threads(values: &[String]) -> Vec<String> { values.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).map(str::to_string).collect::<BTreeSet<_>>().into_iter().collect() }

fn derive_dates(path: &str, primary_date: NaiveDate) -> Vec<NaiveDate> {
    let re = Regex::new(r"(20\d{2})[-_](\d{2})[-_](\d{2})").unwrap();
    let mut out = vec![primary_date];
    for c in re.captures_iter(path) { if let (Ok(y),Ok(m),Ok(d))=(c[1].parse(),c[2].parse(),c[3].parse()) { if let Some(x)=NaiveDate::from_ymd_opt(y,m,d) { out.push(x); } } }
    let stitched = Regex::new(r"(20\d{2})[-_](\d{2})[-_](\d{2})[_-](\d{2})").unwrap();
    for c in stitched.captures_iter(path) { if let (Ok(y),Ok(m),Ok(d),Ok(hour))=(c[1].parse::<i32>(),c[2].parse::<u32>(),c[3].parse::<u32>(),c[4].parse::<u32>()) { if hour < 24 { if let Some(x)=NaiveDate::from_ymd_opt(y,m,d) { out.push(x + chrono::Duration::days(1)); } } } }
    out.sort(); out.dedup(); out
}


fn chunk_body(body: &str) -> Vec<(String, usize, usize, Option<String>)> {
    if body.is_empty() { return vec![]; }
    let chars: Vec<char> = body.chars().collect();
    let mut headings: Vec<(usize, String)> = Vec::new();
    let mut offset = 0usize;
    for line in body.split_inclusive('\n') {
        let content = line.trim_end_matches('\n');
        if content.strip_prefix("## ").is_some_and(|h| !h.trim().is_empty()) { headings.push((offset, content.trim().to_string())); }
        offset += line.chars().count();
    }
    let mut sections = Vec::new();
    if headings.is_empty() { sections.push((0, chars.len(), "__preamble__".to_string())); }
    else {
        if headings[0].0 > 0 { sections.push((0, headings[0].0, "__preamble__".to_string())); }
        for (i, (start, heading)) in headings.iter().enumerate() {
            sections.push((*start, headings.get(i + 1).map(|(s, _)| *s).unwrap_or(chars.len()), heading.clone()));
        }
    }
    let mut out = Vec::new();
    for (start, end, heading) in sections {
        let text: String = chars[start..end].iter().collect();
        if text.chars().count() <= 4000 {
            if !text.trim().is_empty() { out.push((text, start, end, Some(heading))); }
            continue;
        }
        let mut paragraphs = Vec::new();
        let mut paragraph_start = start;
        for i in start..end.saturating_sub(1) {
            if chars[i] == '\n' && chars[i + 1] == '\n' {
                paragraphs.push((paragraph_start, i));
                paragraph_start = i + 2;
            }
        }
        paragraphs.push((paragraph_start, end));
        let mut pieces = Vec::new();
        let mut buf_start = paragraphs[0].0;
        let mut buf_end = paragraphs[0].1;
        for &(paragraph_start, paragraph_end) in paragraphs.iter().skip(1) {
            if buf_end - buf_start + (paragraph_end - paragraph_start) + 2 > 2200 {
                pieces.push((buf_start, buf_end));
                let tail_start = buf_end.saturating_sub(200);
                buf_start = tail_start;
                buf_end = paragraph_end;
                // The source separator is between the overlap tail and this paragraph.
                if buf_start > 0 && buf_start < paragraph_start { buf_end = paragraph_end; }
            } else {
                buf_end = paragraph_end;
            }
        }
        pieces.push((buf_start, buf_end));
        for (piece_start, piece_end) in pieces {
            let piece: String = chars[piece_start..piece_end].iter().collect();
            if !piece.trim().is_empty() { out.push((piece, piece_start, piece_end, Some(heading.clone()))); }
        }
    }
    out
}
fn token_estimate(text: &str) -> i32 { (text.chars().count() / 4).max(1) as i32 }

async fn embed(client: &Client, url: &str, model: &str, chunks: &[(String,usize,usize,Option<String>)], dim: usize) -> Result<Vec<Vec<f32>>, AppError> {
    #[derive(Serialize)] struct Input<'a> { model: &'a str, input: Vec<String> }
    let input = chunks.iter().map(|x| format!("passage: {}", x.0)).collect();
    let value: serde_json::Value = client.post(url).json(&Input { model, input }).send().await.map_err(|e| AppError::Embedding(e.to_string()))?.error_for_status().map_err(|e| AppError::Embedding(e.to_string()))?.json().await.map_err(|e| AppError::Embedding(e.to_string()))?;
    let arr=value.get("embeddings").or_else(||value.get("data")).and_then(|v|v.as_array()).ok_or_else(||AppError::Embedding("response lacks embeddings".into()))?; let mut out=Vec::new();
    for item in arr { let v=item.as_array().or_else(||item.get("embedding").and_then(|x|x.as_array())).ok_or_else(||AppError::Embedding("invalid embedding vector".into()))?; let row:Vec<f32>=v.iter().map(|x|x.as_f64().map(|n|n as f32).ok_or_else(||AppError::Embedding("non-numeric embedding".into()))).collect::<Result<_,_>>()?; if row.len()!=dim{return Err(AppError::Embedding(format!("embedding dimension {} != {}",row.len(),dim)));} out.push(row); } Ok(out)
}

pub async fn process_request(pool: &PgPool, cfg: &Config, req: RememberRequest) -> Result<RememberReceipt, AppError> { remember(pool,cfg,req).await }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn rejects_bad_room(){let r=RememberRequest{room:"House".into(),kind:"memory".into(),title:"x".into(),body:"y".into(),source_path:None,threads:vec![],supersedes:vec![],backup:false};assert!(r.validate().is_err());}
    #[test] fn source_is_db_only(){let r=RememberRequest{room:"room".into(),kind:"memory".into(),title:"x".into(),body:"y".into(),source_path:None,threads:vec![],supersedes:vec![],backup:false};assert!(r.source_path().starts_with("db-only/"));}
    #[test] fn unicode_chunks_use_chars(){let c=chunk_body("éé");assert_eq!((c[0].1,c[0].2), (0,2));assert!(token_estimate("é")>0);}
    #[test] fn oversized_chunks_preserve_separator_and_span(){
        let first="a".repeat(2200); let body=format!("{first}\n\né{}", "b".repeat(2500));
        let chunks=chunk_body(&body); let (text,start,end,_)=&chunks[1];
        assert_eq!(text, &body.chars().collect::<Vec<_>>()[*start..*end].iter().collect::<String>());
        assert!(text.contains("\n\né"));
    }
    #[test] fn derive_dates_uses_injected_primary_date(){let d=NaiveDate::from_ymd_opt(2026,7,22).unwrap();assert!(derive_dates("db-only/room/no-date",d).contains(&d));}
    #[test] fn backup_target_strips_password(){let (target,password)=backup_target("postgres://alice:secret@db.example/memory").unwrap();assert_eq!(password.as_deref(),Some("secret"));assert!(!target.contains("secret"));assert!(target.contains("db.example/memory"));}
    #[test] fn threads_normalize(){assert_eq!(normalize_threads(&[" a ".into()," ".into(),"a".into()]),vec!["a"]);}
    #[test] fn backup_command_targets_resolved_db_without_password_arg(){let command=build_backup_command("postgres://alice:secret@db.example/memory").unwrap();let envs=command.get_envs().filter_map(|(k,v)|Some((k.to_string_lossy().into_owned(),v?.to_string_lossy().into_owned()))).collect::<std::collections::BTreeMap<_,_>>();assert_eq!(envs.get("SOLARISAEL_BACKUP_DATABASE_URL").map(String::as_str),Some("postgres://alice@db.example/memory"));assert_eq!(envs.get("SOLARISAEL_BACKUP_PASSWORD").map(String::as_str),Some("secret"));}
}
