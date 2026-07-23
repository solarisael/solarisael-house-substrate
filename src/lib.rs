use chrono::{Local, NaiveDate};
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize, Serializer};
use serde::ser::SerializeStruct;
use sqlx::{PgPool, PgConnection, Row, postgres::{PgConnectOptions, PgPoolOptions}};
use std::{collections::{BTreeMap, BTreeSet}, env, fs, io};
use std::path::Path;
use std::process::Command;
use std::str::FromStr;
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
    pub room: String,
    pub kind: String,
    pub title: String,
    #[serde(default)]
    pub body: String,
    #[serde(default)]
    pub lesson: Option<String>,
    #[serde(default)]
    pub source_path: Option<String>,
    #[serde(default)]
    pub source_memory_path: Option<String>,
    #[serde(default)]
    pub threads: Vec<String>,
    #[serde(default)]
    pub supersedes: Vec<i64>,
    #[serde(default)]
    pub shape: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default, alias = "proofPattern")]
    pub proof_pattern: Option<String>,
    #[serde(default, alias = "triggerContext")]
    pub trigger_context: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_backup")]
    pub backup: bool,
}
fn default_backup() -> bool { true }

fn is_zero(value: &i64) -> bool { *value == 0 }

#[derive(Debug)]
pub struct RememberReceipt {
    pub memory_id: i64,
    pub lesson_id: i64,
    pub kind: String,
    pub room: String,
    pub source_path: String,
    pub durable: bool,
    pub authority: &'static str,
    pub warnings: Vec<String>,
}

impl Serialize for RememberReceipt {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.memory_id != 0 {
            let mut out = serializer.serialize_struct("RememberReceipt", 6)?;
            out.serialize_field("memory_id", &self.memory_id)?;
            out.serialize_field("room", &self.room)?;
            out.serialize_field("source_path", &self.source_path)?;
            out.serialize_field("durable", &self.durable)?;
            out.serialize_field("authority", &self.authority)?;
            out.serialize_field("warnings", &self.warnings)?;
            out.end()
        } else {
            let mut out = serializer.serialize_struct("RememberReceipt", 5)?;
            out.serialize_field("lesson_id", &self.lesson_id)?;
            out.serialize_field("kind", &self.kind)?;
            out.serialize_field("durable", &self.durable)?;
            out.serialize_field("authority", &self.authority)?;
            out.serialize_field("warnings", &self.warnings)?;
            out.end()
        }
    }
}

impl RememberRequest {
    pub fn validate(&self) -> Result<(), AppError> {
        let room_re = Regex::new(r"^[a-z0-9]+(?:-[a-z0-9]+)*$").unwrap();
        if !room_re.is_match(&self.room) || self.room == "house" { return Err(AppError::Invalid("room must be a lowercase slug and cannot be house".into())); }
        let lessons = ["coding-lesson", "project-lesson", "writing-lesson", "audio-lesson"];
        let text = self.lesson.as_deref().unwrap_or(&self.body);
        if self.title.trim().is_empty() { return Err(AppError::Invalid("title must not be empty".into())); }
        if text.trim().is_empty() { return Err(AppError::Invalid("body/lesson must not be empty".into())); }
        if self.kind == "memory" {
            if self.lesson.is_some() || self.source_memory_path.is_some() || self.shape.is_some() || self.voice.is_some() || self.scope.is_some() || self.project.is_some() || self.proof_pattern.is_some() || self.trigger_context.is_some() || !self.tags.is_empty() {
                return Err(AppError::Invalid("lesson-only fields are not valid for memory".into()));
            }
            if self.source_path.as_ref().is_some_and(|p| p.trim().is_empty()) { return Err(AppError::Invalid("source_path must not be empty".into())); }
            if self.supersedes.iter().any(|id| *id <= 0) { return Err(AppError::Invalid("supersedes must contain positive IDs".into())); }
        } else if lessons.contains(&self.kind.as_str()) {
            if !self.threads.is_empty() || !self.supersedes.is_empty() || self.source_path.is_some() { return Err(AppError::Invalid("threads/supersedes/source_path are memory-only".into())); }
            if self.source_memory_path.as_ref().is_some_and(|p| p.trim().is_empty()) { return Err(AppError::Invalid("source_memory_path must not be empty".into())); }
            let unsupported = match self.kind.as_str() {
                "coding-lesson" => false,
                "project-lesson" => self.voice.is_some() || self.shape.is_some() || self.scope.is_some(),
                "writing-lesson" => self.scope.is_some() || self.project.is_some() || self.proof_pattern.is_some(),
                "audio-lesson" => self.voice.is_some() || self.scope.is_some() || self.project.is_some() || self.proof_pattern.is_some(),
                _ => true,
            };
            if unsupported { return Err(AppError::Invalid("lesson fields are unsupported by this lesson table".into())); }
            if self.kind == "project-lesson" && self.project.as_deref().unwrap_or("").trim().is_empty() { return Err(AppError::Invalid("project is required for project lessons".into())); }
        } else {
            return Err(AppError::Invalid("unsupported remember kind".into()));
        }
        Ok(())
    }
    fn source_path(&self) -> String { self.source_path.clone().unwrap_or_else(|| format!("db-only/{}/{}", self.room, Uuid::new_v4())) }
    fn lesson_body(&self) -> &str { self.lesson.as_deref().unwrap_or(&self.body) }
}

pub async fn remember(pool: &PgPool, cfg: &Config, req: RememberRequest) -> Result<RememberReceipt, AppError> {
    req.validate()?;
    if req.kind != "memory" {
        return remember_lesson(pool, cfg, &req).await;
    }
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
    if cfg.test_embedding_disabled { warnings.push("embedding disabled for isolated test; chunks cleared".into()); }
    else {
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
    Ok(RememberReceipt { memory_id, lesson_id: 0, kind: "memory".into(), room: req.room, source_path, durable: true, authority: "postgres", warnings })
}

async fn remember_lesson(pool: &PgPool, cfg: &Config, req: &RememberRequest) -> Result<RememberReceipt, AppError> {
    let text = req.lesson_body();
    let tags = req.tags.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).map(str::to_owned).collect::<Vec<_>>();
    let meta = serde_json::json!({"origin":"direct-db-write", "kind": req.kind, "recorded_at": chrono::Utc::now().to_rfc3339()});
    let mut tx = pool.begin().await?;
    let id = match req.kind.as_str() {
        "coding-lesson" => sqlx::query_scalar::<_, i64>("INSERT INTO coding_lessons (scope,project,voice,shape,title,lesson,trigger_context,proof_pattern,tags,source_memory_path,meta) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) ON CONFLICT (scope,project,title) DO UPDATE SET project=EXCLUDED.project,voice=EXCLUDED.voice,shape=EXCLUDED.shape,lesson=EXCLUDED.lesson,trigger_context=EXCLUDED.trigger_context,proof_pattern=EXCLUDED.proof_pattern,tags=EXCLUDED.tags,source_memory_path=EXCLUDED.source_memory_path,meta=EXCLUDED.meta RETURNING id")
            .bind(req.scope.as_deref().unwrap_or("shared")).bind(&req.project).bind(&req.voice).bind(&req.shape).bind(&req.title).bind(text).bind(&req.trigger_context).bind(&req.proof_pattern).bind(&tags).bind(&req.source_memory_path).bind(meta).fetch_one(&mut *tx).await?,
        "project-lesson" => sqlx::query_scalar::<_, i64>("INSERT INTO project_lessons (project,title,lesson,trigger_context,proof_pattern,tags,source_memory_path,meta) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT (project,title) DO UPDATE SET lesson=EXCLUDED.lesson,trigger_context=EXCLUDED.trigger_context,proof_pattern=EXCLUDED.proof_pattern,tags=EXCLUDED.tags,source_memory_path=EXCLUDED.source_memory_path,meta=EXCLUDED.meta RETURNING id")
            .bind(req.project.as_deref().unwrap()).bind(&req.title).bind(text).bind(&req.trigger_context).bind(&req.proof_pattern).bind(&tags).bind(&req.source_memory_path).bind(meta).fetch_one(&mut *tx).await?,
        "writing-lesson" => sqlx::query_scalar::<_, i64>("INSERT INTO writing_lessons (voice,shape,title,lesson,trigger_context,tags,source_memory_path,meta) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT (voice,title) DO UPDATE SET shape=EXCLUDED.shape,lesson=EXCLUDED.lesson,trigger_context=EXCLUDED.trigger_context,tags=EXCLUDED.tags,source_memory_path=EXCLUDED.source_memory_path,meta=EXCLUDED.meta RETURNING id")
            .bind(req.voice.as_deref().unwrap_or("general")).bind(&req.shape).bind(&req.title).bind(text).bind(&req.trigger_context).bind(&tags).bind(&req.source_memory_path).bind(meta).fetch_one(&mut *tx).await?,
        "audio-lesson" => sqlx::query_scalar::<_, i64>("INSERT INTO audio_lessons (shape,title,lesson,trigger_context,tags,source_memory_path) VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT (title) DO UPDATE SET shape=EXCLUDED.shape,lesson=EXCLUDED.lesson,trigger_context=EXCLUDED.trigger_context,tags=EXCLUDED.tags,source_memory_path=EXCLUDED.source_memory_path RETURNING id")
            .bind(&req.shape).bind(&req.title).bind(text).bind(&req.trigger_context).bind(&tags).bind(&req.source_memory_path).fetch_one(&mut *tx).await?,
        _ => return Err(AppError::Invalid("unsupported remember kind".into())),
    };
    tx.commit().await?;
    let mut warnings = Vec::new();
    if req.backup && matches!(req.kind.as_str(), "project-lesson" | "audio-lesson") {
        if let Err(e) = run_backup(&cfg.database_url) { warnings.push(format!("backup failed: {e}")); }
    }
    Ok(RememberReceipt { memory_id: 0, lesson_id: id, kind: req.kind.clone(), room: req.room.clone(), source_path: String::new(), durable: true, authority: "postgres", warnings })
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
    let byte_at = |char_index: usize| -> usize {
        body.char_indices().nth(char_index).map(|(i, _)| i).unwrap_or(body.len())
    };
    let mut out = Vec::new();
    for (start, end, heading) in sections {
        let text: String = chars[start..end].iter().collect();
        if text.chars().count() <= 4000 {
            if !text.trim().is_empty() { out.push((text, byte_at(start), byte_at(end), Some(heading))); }
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
                buf_start = buf_end.saturating_sub(200);
                buf_end = paragraph_end;
            } else {
                buf_end = paragraph_end;
            }
        }
        pieces.push((buf_start, buf_end));
        for (piece_start, piece_end) in pieces {
            let piece: String = chars[piece_start..piece_end].iter().collect();
            if !piece.trim().is_empty() { out.push((piece, byte_at(piece_start), byte_at(piece_end), Some(heading.clone()))); }
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
fn default_semantic_top_k() -> u32 { 8 }
fn default_semantic_min_similarity() -> f64 { 0.50 }
fn default_content_top_k() -> u32 { 8 }
fn default_content_min_similarity() -> f64 { 0.30 }

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecallParams {
    pub room: String,
    pub query: String,
    #[serde(default = "default_semantic_top_k")] pub semantic_top_k: u32,
    #[serde(default = "default_semantic_min_similarity")] pub semantic_min_similarity: f64,
    #[serde(default = "default_content_top_k")] pub content_top_k: u32,
    #[serde(default = "default_content_min_similarity")] pub content_min_similarity: f64,
}

impl RecallParams {
    pub fn validate(&self) -> Result<(), AppError> {
        let room_re = Regex::new(r"^[a-z0-9]+(?:-[a-z0-9]+)*$").unwrap();
        if !room_re.is_match(&self.room) || self.room == "house" { return Err(AppError::Invalid("room must be a lowercase slug and cannot be house".into())); }
        if self.query.trim().is_empty() { return Err(AppError::Invalid("query must not be empty".into())); }
        for (name, value) in [("semantic_top_k", self.semantic_top_k), ("content_top_k", self.content_top_k)] {
            if value == 0 || value > 1000 { return Err(AppError::Invalid(format!("{name} must be positive and at most 1000"))); }
        }
        for (name, value) in [("semantic_min_similarity", self.semantic_min_similarity), ("content_min_similarity", self.content_min_similarity)] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) { return Err(AppError::Invalid(format!("{name} must be finite and in [0, 1]"))); }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct RecallResult {
    pub ok: bool,
    pub query: String,
    pub found: bool,
    pub source: &'static str,
    pub warnings: Vec<String>,
    #[serde(rename = "retrievalCandidates")] pub retrieval_candidates: Vec<serde_json::Value>,
    #[serde(rename = "canonMatches")] pub canon_matches: Vec<serde_json::Value>,
    #[serde(rename = "semanticChunks")] pub semantic_chunks: Vec<serde_json::Value>,
    #[serde(rename = "contentChunks")] pub content_chunks: Vec<serde_json::Value>,
    #[serde(rename = "dateMatches")] pub date_matches: Vec<serde_json::Value>,
    #[serde(rename = "queryDates")] pub query_dates: Vec<serde_json::Value>,
    pub taxonomy: serde_json::Value,
    #[serde(rename = "clusterStaleness", skip_serializing_if = "Option::is_none")]
    pub cluster_staleness: Option<serde_json::Value>,
    #[serde(rename = "clusterResonance", skip_serializing_if = "Option::is_none")]
    pub cluster_resonance: Option<serde_json::Value>,
}

fn query_dates(query: &str) -> Vec<NaiveDate> {
    let re = Regex::new(r"\b(20\d{2})-(\d{2})-(\d{2})\b").unwrap();
    re.captures_iter(query).filter_map(|c| NaiveDate::from_ymd_opt(c[1].parse().ok()?, c[2].parse().ok()?, c[3].parse().ok()?)).collect::<BTreeSet<_>>().into_iter().collect()
}
fn query_terms(query: &str) -> Vec<String> {
    query.split(|c: char| !c.is_alphanumeric())
        .map(|term| term.trim().to_ascii_lowercase())
        .filter(|term| term.len() >= 2)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn term_evidence(terms: &[String], fields: &[&str]) -> (Vec<String>, Vec<String>) {
    let haystack = fields.join(" ").to_ascii_lowercase();
    let matched = terms.iter().filter(|term| haystack.contains(term.as_str())).cloned().collect::<Vec<_>>();
    let missing = terms.iter().filter(|term| !matched.contains(term)).cloned().collect::<Vec<_>>();
    (matched, missing)
}


async fn embed_query(client: &Client, url: &str, model: &str, query: &str, dim: usize) -> Result<Vec<f32>, AppError> {
    #[derive(Serialize)] struct Input<'a> { model: &'a str, input: Vec<String> }
    let value: serde_json::Value = client.post(url).json(&Input { model, input: vec![format!("query: {query}")] }).send().await.map_err(|e| AppError::Embedding(e.to_string()))?.error_for_status().map_err(|e| AppError::Embedding(e.to_string()))?.json().await.map_err(|e| AppError::Embedding(e.to_string()))?;
    let item = value.get("embeddings").or_else(|| value.get("data")).and_then(|v| v.as_array()).and_then(|a| a.first()).ok_or_else(|| AppError::Embedding("response lacks query embedding".into()))?;
    let v = item.as_array().or_else(|| item.get("embedding").and_then(|x| x.as_array())).ok_or_else(|| AppError::Embedding("invalid query embedding".into()))?;
    let row = v.iter().map(|x| x.as_f64().map(|n| n as f32).ok_or_else(|| AppError::Embedding("non-numeric query embedding".into()))).collect::<Result<Vec<_>,_>>()?;
    if row.len() != dim { return Err(AppError::Embedding(format!("embedding dimension {} != {}",row.len(),dim))); }
    Ok(row)
}

fn bounded_excerpt(body: &str) -> String {
    const MAX: usize = 1200;
    let excerpt: String = body.chars().take(MAX).collect();
    if body.chars().count() > MAX { format!("{excerpt}…") } else { excerpt }
}

fn candidate_terms(terms: &[String], fields: &[&str]) -> (Vec<String>, Vec<String>, f64) {
    let (matched, missing) = term_evidence(terms, fields);
    let coverage = if terms.is_empty() { 0.0 } else { matched.len() as f64 / terms.len() as f64 };
    (matched, missing, coverage)
}
pub async fn recall(pool: &PgPool, cfg: &Config, params: RecallParams) -> Result<RecallResult, AppError> {
    params.validate()?;
    let query_dates = query_dates(&params.query);
    let query_terms = query_terms(&params.query);
    let rooms = vec![params.room.clone(), "house".to_string()];
    let mut warnings = Vec::new();
    let vector_text = match (cfg.test_embedding_disabled, cfg.embed_url.as_deref()) {
        (true, _) => { warnings.push("semantic lane absent: embedding disabled".to_string()); None }
        (false, Some(url)) => match embed_query(&Client::new(), url, &cfg.embed_model, &params.query, EMBED_DIMENSION).await {
            Ok(vector) => Some(format!("[{}]", vector.iter().map(ToString::to_string).collect::<Vec<_>>().join(","))),
            Err(e) => { warnings.push(format!("semantic lane absent: {e}")); None }
        },
        (false, None) => { warnings.push("semantic lane absent: embedding endpoint is required".to_string()); None }
    };
    let mut semantic_chunks = Vec::new();
    if let Some(vector_text) = vector_text.clone() {
        let semantic_rows = sqlx::query("SELECT m.source_path,coalesce(m.title,'') AS title,coalesce(c.heading_path,'') AS heading_path,c.body,c.char_start,c.char_end,c.chunk_index,(1-(c.body_embedding <=> $1::vector))::double precision AS sim FROM memory_chunks c JOIN memories m ON m.id=c.memory_id WHERE m.room = ANY($2::text[]) AND m.archived_at IS NULL AND m.superseded_by IS NULL AND c.body_embedding IS NOT NULL ORDER BY sim DESC,m.source_path,c.chunk_index LIMIT $3").bind(&vector_text).bind(&rooms).bind(params.semantic_top_k as i64).fetch_all(pool).await?;
        for row in semantic_rows {
            let sim: f64 = row.try_get("sim")?;
            if sim < params.semantic_min_similarity { continue; }
            let source_path: String = row.try_get("source_path")?;
            let title: Option<String> = row.try_get("title")?;
            let heading_path: Option<String> = row.try_get("heading_path")?;
            let body: String = row.try_get("body")?;
            let (matched_terms, missing_terms, coverage) = candidate_terms(&query_terms, &[&source_path, title.as_deref().unwrap_or(""), heading_path.as_deref().unwrap_or(""), &body]);
            semantic_chunks.push(serde_json::json!({"source_path":source_path,"title":title,"heading_path":heading_path,"body":bounded_excerpt(&body),"char_start":row.try_get::<i32,_>("char_start")?,"char_end":row.try_get::<i32,_>("char_end")?,"chunk_index":row.try_get::<i32,_>("chunk_index")?,"sim":sim,"matched_terms":matched_terms,"missing_terms":missing_terms,"term_coverage":coverage,"evidence":"semantic cosine similarity"}));
        }
    }
    let content_rows = sqlx::query("SELECT m.source_path,coalesce(m.title,'') AS title,coalesce(c.heading_path,'') AS heading_path,c.body,c.char_start,c.char_end,c.chunk_index,word_similarity(c.body,$1)::double precision AS sim FROM memory_chunks c JOIN memories m ON m.id=c.memory_id WHERE m.room = ANY($2::text[]) AND m.archived_at IS NULL AND m.superseded_by IS NULL AND ($5::text[] = '{}'::text[] OR EXISTS (SELECT 1 FROM unnest($5::text[]) term WHERE lower(c.body) LIKE '%' || term || '%')) AND word_similarity(c.body,$1) >= $3 ORDER BY sim DESC,m.source_path,c.chunk_index LIMIT $4")
        .bind(&params.query).bind(&rooms).bind(params.content_min_similarity).bind(params.content_top_k as i64).bind(&query_terms).fetch_all(pool).await?;
    let mut content_chunks = Vec::new();
    for row in content_rows {
        let sim: f64 = row.try_get("sim")?;
        let source_path: String = row.try_get("source_path")?;
        let title: Option<String> = row.try_get("title")?;
        let heading_path: Option<String> = row.try_get("heading_path")?;
        let body: String = row.try_get("body")?;
        let (matched_terms, missing_terms, coverage) = candidate_terms(&query_terms, &[&source_path, title.as_deref().unwrap_or(""), heading_path.as_deref().unwrap_or(""), &body]);
        content_chunks.push(serde_json::json!({"source_path":source_path,"title":title,"heading_path":heading_path,"body":bounded_excerpt(&body),"char_start":row.try_get::<i32,_>("char_start")?,"char_end":row.try_get::<i32,_>("char_end")?,"chunk_index":row.try_get::<i32,_>("chunk_index")?,"ws":sim,"matched_terms":matched_terms,"missing_terms":missing_terms,"term_coverage":coverage,"evidence":"lexical word_similarity"}));
    }
    let mut date_matches = Vec::new();
    if !query_dates.is_empty() {
        let rows = sqlx::query("SELECT source_path,title,body,date,dates FROM memories WHERE room = ANY($1::text[]) AND archived_at IS NULL AND superseded_by IS NULL AND dates && $2::date[] ORDER BY source_path LIMIT 5")
            .bind(&rooms).bind(&query_dates).fetch_all(pool).await?;
        for row in rows {
            let source_path: String = row.try_get("source_path")?;
            let title: Option<String> = row.try_get("title")?;
            let body: String = row.try_get("body")?;
            let dates: Vec<NaiveDate> = row.try_get("dates")?;
            let (matched_terms, missing_terms, coverage) = candidate_terms(&query_terms, &[&source_path, title.as_deref().unwrap_or(""), &body]);
            date_matches.push(serde_json::json!({"source_path":source_path,"title":title,"body_excerpt":bounded_excerpt(&body),"excerpt":bounded_excerpt(&body),"date":row.try_get::<Option<NaiveDate>,_>("date")?.map(|d|d.to_string()),"dates":dates.into_iter().map(|d|d.to_string()).collect::<Vec<_>>(),"score":1.0,"reason":"date match","matched_terms":matched_terms,"missing_terms":missing_terms,"term_coverage":coverage}));
        }
    }
    let mut fused: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for (rank, c) in semantic_chunks.iter().enumerate() {
        let key = format!("{}#{}", c["source_path"].as_str().unwrap_or(""), c["chunk_index"].as_i64().unwrap_or(0));
        let score = c["sim"].as_f64().unwrap_or(0.0) * 0.6 + 1.0 / (rank as f64 + 1.0) * 0.4;
        fused.insert(key, serde_json::json!({"source_path":c["source_path"],"title":c["title"],"heading_path":c["heading_path"],"excerpt":c["body"],"sources":[c["source_path"]],"term_coverage":c["term_coverage"],"matched_terms":c["matched_terms"],"missing_terms":c["missing_terms"],"score":score,"semantic_score":c["sim"],"reasons":["semantic cosine similarity"],"source":"semantic","chunk_index":c["chunk_index"]}));
    }
    for (rank, c) in content_chunks.iter().enumerate() {
        let key = format!("{}#{}", c["source_path"].as_str().unwrap_or(""), c["chunk_index"].as_i64().unwrap_or(0));
        let score = c["ws"].as_f64().unwrap_or(0.0) * 0.6 + 1.0 / (rank as f64 + 1.0) * 0.4;
        if let Some(existing) = fused.get_mut(&key) {
            existing["score"] = serde_json::json!(existing["score"].as_f64().unwrap_or(0.0) + score);
            existing["content_score"] = c["ws"].clone();
            existing["source"] = serde_json::json!("semantic+content");
            existing["reasons"] = serde_json::json!(["semantic cosine similarity","lexical word_similarity"]);
        } else {
            fused.insert(key, serde_json::json!({"source_path":c["source_path"],"title":c["title"],"heading_path":c["heading_path"],"excerpt":c["body"],"sources":[c["source_path"]],"term_coverage":c["term_coverage"],"matched_terms":c["matched_terms"],"missing_terms":c["missing_terms"],"score":score,"content_score":c["ws"],"reasons":["lexical word_similarity"],"source":"content","chunk_index":c["chunk_index"]}));
        }
    }
    let mut retrieval_candidates: Vec<_> = fused.into_values().collect();
    retrieval_candidates.sort_by(|a,b| b["score"].as_f64().unwrap_or(0.0).partial_cmp(&a["score"].as_f64().unwrap_or(0.0)).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a["source_path"].as_str().cmp(&b["source_path"].as_str())).then_with(|| a["chunk_index"].as_i64().cmp(&b["chunk_index"].as_i64())));
    retrieval_candidates.truncate(params.semantic_top_k.max(params.content_top_k) as usize);
    let canon_rows = sqlx::query("SELECT name,kind,summary,aliases,weighty,pointer_files FROM named_entities WHERE room = ANY($1::text[]) AND (lower(name) = ANY($2::text[]) OR EXISTS (SELECT 1 FROM unnest(aliases) alias WHERE lower(alias) = ANY($2::text[]))) ORDER BY name LIMIT 12")
        .bind(&rooms).bind(&query_terms).fetch_all(pool).await?;
    let mut canon_matches = Vec::new();
    let mut named_entities = Vec::new();
    for row in canon_rows {
        let name: String = row.try_get("name")?;
        let kind: String = row.try_get("kind")?;
        let summary: String = row.try_get("summary")?;
        let aliases: Vec<String> = row.try_get("aliases")?;
        let weighty: bool = row.try_get("weighty")?;
        let files: serde_json::Value = row.try_get("pointer_files")?;
        canon_matches.push(serde_json::json!({"termKey":name,"entry":{"type":kind,"summary":bounded_excerpt(&summary),"aliases":aliases,"weighty":weighty,"files":files}}));
        named_entities.push(name);
    }
    let memory_types: Vec<String> = sqlx::query_scalar("SELECT DISTINCT type FROM memories WHERE room = ANY($1::text[]) AND archived_at IS NULL AND superseded_by IS NULL ORDER BY type LIMIT 20").bind(&rooms).fetch_all(pool).await?;
    let thread_keys: Vec<String> = sqlx::query_scalar("SELECT DISTINCT thread_key FROM memory_threads t JOIN memories m ON m.id=t.memory_id WHERE m.room = ANY($1::text[]) AND m.archived_at IS NULL AND m.superseded_by IS NULL ORDER BY thread_key LIMIT 20").bind(&rooms).fetch_all(pool).await?;
    let taxonomy = serde_json::json!({"rooms":rooms,"memoryTypes":memory_types,"threadKeys":thread_keys,"namedEntities":named_entities});
    let cluster_staleness = cluster_staleness(pool, None).await.ok().and_then(|s| serde_json::to_value(s).ok());
    let cluster_resonance = if let Some(v) = vector_text.as_deref() {
        cluster_resonance(pool, v, &rooms).await.ok()
    } else { None };
    Ok(RecallResult { ok:true, query:params.query, found:!retrieval_candidates.is_empty() || !canon_matches.is_empty() || !date_matches.is_empty(), source:"rust-postgres", warnings, retrieval_candidates, canon_matches, semantic_chunks, content_chunks, date_matches, query_dates:query_dates.into_iter().map(|d|serde_json::json!(d.to_string())).collect(), taxonomy, cluster_staleness, cluster_resonance })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnamnesisParams {
    pub room: String,
    #[serde(default = "default_anamnesis_mode")]
    pub mode: String,
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub limit: Option<u32>,
}
fn default_anamnesis_mode() -> String { "wake".into() }

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnamnesisSeed {
    pub number: i32,
    #[serde(default, alias = "occurredOn")]
    pub occurred_on: Option<NaiveDate>,
    #[serde(alias = "howItWent")]
    pub how_it_went: String,
    #[serde(alias = "portalPull")]
    pub portal_pull: String,
    pub lighter: String,
    #[serde(default, alias = "sourcePath")]
    pub source_path: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnamnesisWrite {
    pub room: String,
    pub operation: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub fidelity: Option<String>,
    #[serde(default)]
    pub activation: Option<String>,
    #[serde(default)]
    pub dormant: bool,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub shape: Option<String>,
    pub ramp: Option<String>,
    pub counsel: Option<String>,
    pub peak: Option<String>,
    pub beginning: Option<String>,
    #[serde(default, alias = "verifyNote")]
    pub verify_note: Option<String>,
    #[serde(default, alias = "sourcePaths")]
    pub source_paths: Vec<String>,
    #[serde(default, alias = "canon")]
    pub canon_links: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, alias = "allowEmptyCycle")]
    pub allow_empty_cycle: bool,
    #[serde(default, alias = "seedRep")]
    pub seed_rep: Option<AnamnesisSeed>,
    #[serde(default = "default_backup")]
    pub backup: bool,
    #[serde(default, alias = "repNumber")]
    pub rep_number: Option<i32>,
    #[serde(default, alias = "occurredOn")]
    pub occurred_on: Option<NaiveDate>,
    #[serde(default, alias = "howItWent")]
    pub how_it_went: Option<String>,
    #[serde(default, alias = "portalPull")]
    pub portal_pull: Option<String>,
    pub lighter: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AnamnesisReceipt {
    pub operation: String,
    pub room: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(rename = "repNumber", skip_serializing_if = "Option::is_none")]
    pub rep_number: Option<i32>,
    pub durable: bool,
    pub authority: String,
    pub warnings: Vec<String>,
}

fn validate_anamnesis_room(room: &str) -> Result<(), AppError> {
    let room_re = Regex::new(r"^[a-z0-9]+(?:-[a-z0-9]+)*$").unwrap();
    if !room_re.is_match(room) {
        return Err(AppError::Invalid("room must be a lowercase slug".into()));
    }
    Ok(())
}

impl AnamnesisParams {
    pub fn validate(&self) -> Result<(String, u32), AppError> {
        validate_anamnesis_room(&self.room)?;
        if self.mode != "wake" && self.mode != "consult" {
            return Err(AppError::Invalid(format!("invalid anamnesis mode: {}", self.mode)));
        }
        if self.mode == "consult" && self.query.trim().is_empty() {
            return Err(AppError::Invalid("consult requires a non-empty query".into()));
        }
        let limit = self.limit.unwrap_or(10).clamp(1, 50);
        Ok((self.mode.clone(), limit))
    }
}
async fn anamnesis_embedding(cfg: &Config, text: &str) -> Result<Option<String>, AppError> {
    if cfg.test_embedding_disabled { return Ok(None); }
    let url = cfg.embed_url.as_deref().ok_or_else(|| AppError::Embedding("embedding endpoint is required".into()))?;
    let rows = embed(&Client::new(), url, &cfg.embed_model, &[(text.to_owned(), 0, text.len(), None)], cfg.embed_dimension).await?;
    Ok(rows.first().map(|v| format!("[{}]", v.iter().map(ToString::to_string).collect::<Vec<_>>().join(","))))
}
pub async fn anamnesis_write(pool: &PgPool, cfg: &Config, req: AnamnesisWrite) -> Result<AnamnesisReceipt, AppError> {
    validate_anamnesis_room(&req.room)?;
    let mut warnings = Vec::new();
    let mut tx = pool.begin().await?;
    let (id, kind, rep_number);
    match req.operation.as_str() {
        "add" => {
            let cabinet_kind = req.kind.as_deref().ok_or_else(|| AppError::Invalid("kind is required".into()))?;
            if !["pillar", "cycle"].contains(&cabinet_kind) { return Err(AppError::Invalid("kind must be pillar or cycle".into())); }
            if cabinet_kind == "pillar" && req.seed_rep.is_some() { return Err(AppError::Invalid("pillar cannot include seedRep".into())); }
            let fidelity = req.fidelity.as_deref().unwrap_or("record");
            if !["record", "raw-material"].contains(&fidelity) { return Err(AppError::Invalid("fidelity must be record or raw-material".into())); }
            let activation = req.activation.as_deref().unwrap_or("fork");
            if !["wake", "fork"].contains(&activation) { return Err(AppError::Invalid("activation must be wake or fork".into())); }
            if req.title.trim().is_empty() { return Err(AppError::Invalid("title is required".into())); }
            let ramp = req.ramp.as_deref().unwrap_or("");
            if ramp.trim().is_empty() { return Err(AppError::Invalid("ramp is required".into())); }
            if cabinet_kind == "cycle" && !req.allow_empty_cycle && req.seed_rep.is_none() { return Err(AppError::Invalid("cycle requires seedRep unless allowEmptyCycle".into())); }
            if cabinet_kind == "cycle" && activation == "wake" && req.verify_note.as_deref().unwrap_or("").trim().is_empty() { return Err(AppError::Invalid("wake cycle requires verifyNote".into())); }
            let embedding = anamnesis_embedding(cfg, &[&req.title, req.shape.as_deref().unwrap_or(""), ramp, req.counsel.as_deref().unwrap_or(""), req.peak.as_deref().unwrap_or("")].join("\n")).await?;
            if embedding.is_none() && cfg.test_embedding_disabled { warnings.push("embedding disabled for isolated test; cabinet embedding omitted".into()); }
            id = sqlx::query_scalar::<_, i64>("INSERT INTO anamnesis (room,kind,fidelity,activation,active,title,shape,peak,beginning,ramp,counsel,verify_note,source_paths,canon_links,tags,body_embedding,embedded_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16::vector,CASE WHEN $16 IS NULL THEN NULL ELSE NOW() END) ON CONFLICT (room,title) DO UPDATE SET kind=EXCLUDED.kind,fidelity=EXCLUDED.fidelity,activation=EXCLUDED.activation,active=EXCLUDED.active,shape=EXCLUDED.shape,peak=EXCLUDED.peak,beginning=EXCLUDED.beginning,ramp=EXCLUDED.ramp,counsel=EXCLUDED.counsel,verify_note=EXCLUDED.verify_note,source_paths=EXCLUDED.source_paths,canon_links=EXCLUDED.canon_links,tags=EXCLUDED.tags,body_embedding=EXCLUDED.body_embedding,embedded_at=EXCLUDED.embedded_at RETURNING id")
                .bind(&req.room).bind(cabinet_kind).bind(fidelity).bind(activation).bind(!req.dormant).bind(&req.title).bind(&req.shape).bind(&req.peak).bind(&req.beginning).bind(ramp).bind(&req.counsel).bind(&req.verify_note).bind(&req.source_paths).bind(&req.canon_links).bind(&req.tags).bind(embedding).fetch_one(&mut *tx).await?;
            if let Some(seed) = req.seed_rep {
                sqlx::query("INSERT INTO anamnesis_reps (cabinet_id,rep_number,occurred_on,how_it_went,portal_pull,lighter,source_path) VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT (cabinet_id,rep_number) DO UPDATE SET occurred_on=EXCLUDED.occurred_on,how_it_went=EXCLUDED.how_it_went,portal_pull=EXCLUDED.portal_pull,lighter=EXCLUDED.lighter,source_path=EXCLUDED.source_path")
                    .bind(id).bind(seed.number).bind(seed.occurred_on).bind(seed.how_it_went).bind(seed.portal_pull).bind(seed.lighter).bind(seed.source_path).execute(&mut *tx).await?;
            }
            kind = Some(cabinet_kind.to_string());
            rep_number = None;
        },
        "append-rep" => {
            let number = req.rep_number.ok_or_else(|| AppError::Invalid("append-rep requires repNumber".into()))?;
            let how = req.how_it_went.as_deref().filter(|s| !s.trim().is_empty()).ok_or_else(|| AppError::Invalid("append-rep requires howItWent".into()))?;
            let portal = req.portal_pull.as_deref().filter(|s| !s.trim().is_empty()).ok_or_else(|| AppError::Invalid("append-rep requires portalPull".into()))?;
            let lighter = req.lighter.as_deref().filter(|s| !s.trim().is_empty()).ok_or_else(|| AppError::Invalid("append-rep requires lighter".into()))?;
            let title = req.title.trim();
            if title.is_empty() { return Err(AppError::Invalid("title is required".into())); }
            id = sqlx::query_scalar::<_, i64>("SELECT id FROM anamnesis WHERE room=$1 AND title=$2 AND kind='cycle'").bind(&req.room).bind(title).fetch_optional(&mut *tx).await?.ok_or_else(|| AppError::Invalid("append-rep target cycle not found".into()))?;
            sqlx::query("INSERT INTO anamnesis_reps (cabinet_id,rep_number,occurred_on,how_it_went,portal_pull,lighter,source_path) VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT (cabinet_id,rep_number) DO UPDATE SET occurred_on=EXCLUDED.occurred_on,how_it_went=EXCLUDED.how_it_went,portal_pull=EXCLUDED.portal_pull,lighter=EXCLUDED.lighter,source_path=EXCLUDED.source_path")
                .bind(id).bind(number).bind(req.occurred_on).bind(how).bind(portal).bind(lighter).bind(req.source_paths.first()).execute(&mut *tx).await?;
            kind = Some("cycle".into());
            rep_number = Some(number);
        },
        _ => return Err(AppError::Invalid("operation must be add or append-rep".into())),
    }
    tx.commit().await?;
    if req.backup { if let Err(e) = run_backup(&cfg.database_url) { warnings.push(format!("backup failed: {e}")); } }
    Ok(AnamnesisReceipt { operation: req.operation, room: req.room, title: req.title, kind, rep_number, durable: true, authority: "substrate".into(), warnings })
}

#[derive(Debug, Serialize)]
pub struct AnamnesisResult { pub ok: bool, pub mode: String, pub room: String, pub query: String, pub found: bool, pub entries: Vec<serde_json::Value>, pub warnings: Vec<String> }

pub async fn anamnesis(pool: &PgPool, params: AnamnesisParams) -> Result<AnamnesisResult, AppError> {
    let (mode, limit) = params.validate()?;
    let rooms = if params.room == "house" { vec!["house".to_string()] } else { vec![params.room.clone(), "house".to_string()] };
    let rows = if mode == "wake" {
        sqlx::query("SELECT id,room,kind,fidelity,activation,active,title,shape,peak,beginning,ramp,counsel,verify_note,source_paths,canon_links,tags FROM anamnesis WHERE room=ANY($1::text[]) AND ((kind='pillar' AND activation='wake') OR (kind='cycle' AND activation='wake' AND active)) ORDER BY CASE WHEN kind='pillar' THEN 0 ELSE 1 END,updated_at DESC,id DESC LIMIT $2").bind(&rooms).bind(limit as i64).fetch_all(pool).await?
    } else {
        sqlx::query("SELECT id,room,kind,fidelity,activation,active,title,shape,peak,beginning,ramp,counsel,verify_note,source_paths,canon_links,tags FROM anamnesis WHERE room=ANY($1::text[]) AND (body_tsv @@ plainto_tsquery('portuguese',$2) OR lower(title||' '||coalesce(shape,'')||' '||ramp||' '||coalesce(counsel,'')||' '||coalesce(peak,'')||' '||array_to_string(canon_links,' ')||' '||array_to_string(tags,' ')) LIKE '%'||lower($2)||'%') ORDER BY (ts_rank_cd(body_tsv, plainto_tsquery('portuguese',$2)) * 10 + similarity(lower(title||' '||coalesce(shape,'')||' '||ramp||' '||coalesce(counsel,'')||' '||coalesce(peak,'')||' '||array_to_string(canon_links,' ')||' '||array_to_string(tags,' ')), lower($2))) DESC, updated_at DESC,title LIMIT $3").bind(&rooms).bind(&params.query).bind(limit as i64).fetch_all(pool).await?
    };
    let mut entries = Vec::new();
    let mut warnings = Vec::new();
    for row in rows {
        let id: i64 = row.try_get("id")?;
        let kind: String = row.try_get("kind")?;
        let verify: Option<String> = row.try_get("verify_note")?;
        if mode == "wake" && kind == "cycle" && verify.as_deref().unwrap_or("").trim().is_empty() { warnings.push(format!("excluded cycle {id}: blank verify_note")); continue; }
        let reps = sqlx::query("SELECT rep_number,occurred_on,how_it_went,portal_pull,lighter,source_path FROM anamnesis_reps WHERE cabinet_id=$1 ORDER BY occurred_on DESC NULLS LAST,rep_number DESC,id DESC LIMIT 3").bind(id).fetch_all(pool).await?;
        let reps = reps.into_iter().rev().map(|r| serde_json::json!({"rep_number":r.try_get::<i32,_>("rep_number").unwrap_or_default(),"occurred_on":r.try_get::<Option<NaiveDate>,_>("occurred_on").ok().flatten().map(|d|d.to_string()),"how_it_went":r.try_get::<String,_>("how_it_went").unwrap_or_default(),"portal_pull":r.try_get::<Option<String>,_>("portal_pull").ok().flatten(),"lighter":r.try_get::<Option<String>,_>("lighter").ok().flatten(),"source_path":r.try_get::<Option<String>,_>("source_path").ok().flatten()})).collect::<Vec<_>>();
        entries.push(serde_json::json!({"id":id,"room":row.try_get::<String,_>("room")?,"kind":kind,"fidelity":row.try_get::<String,_>("fidelity")?,"activation":row.try_get::<String,_>("activation")?,"active":row.try_get::<bool,_>("active")?,"title":row.try_get::<String,_>("title")?,"shape":row.try_get::<Option<String>,_>("shape")?,"peak":row.try_get::<Option<String>,_>("peak")?,"beginning":row.try_get::<Option<String>,_>("beginning")?,"ramp":row.try_get::<String,_>("ramp")?,"counsel":row.try_get::<Option<String>,_>("counsel")?,"verify_note":verify,"source_paths":row.try_get::<Vec<String>,_>("source_paths")?,"canon_links":row.try_get::<Vec<String>,_>("canon_links")?,"tags":row.try_get::<Vec<String>,_>("tags")?,"reps":reps}));
    }
    let found = entries.len();
    Ok(AnamnesisResult { ok: true, mode, room: params.room, query: params.query, found: !entries.is_empty(), entries, warnings })
}

pub async fn process_request(pool: &PgPool, cfg: &Config, req: RememberRequest) -> Result<RememberReceipt, AppError> { remember(pool,cfg,req).await }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn rejects_bad_room(){let r=RememberRequest{room:"House".into(),kind:"memory".into(),title:"x".into(),body:"y".into(),lesson:None,source_path:None,source_memory_path:None,threads:vec![],supersedes:vec![],shape:None,voice:None,scope:None,project:None,proof_pattern:None,trigger_context:None,tags:vec![],backup:false};assert!(r.validate().is_err());}
    #[test] fn source_is_db_only(){let r=RememberRequest{room:"room".into(),kind:"memory".into(),title:"x".into(),body:"y".into(),lesson:None,source_path:None,source_memory_path:None,threads:vec![],supersedes:vec![],shape:None,voice:None,scope:None,project:None,proof_pattern:None,trigger_context:None,tags:vec![],backup:false};assert!(r.source_path().starts_with("db-only/"));}
    #[test] fn lesson_validation_enforces_project_and_memory_field_boundaries() {
        let mut project = RememberRequest { room:"room".into(), kind:"project-lesson".into(), title:"t".into(), body:"unicode\n多行".into(), lesson:None, source_path:None, source_memory_path:None, threads:vec![], supersedes:vec![], shape:None, voice:None, scope:None, project:None, proof_pattern:Some("proof".into()), trigger_context:None, tags:vec!["a".into()], backup:false };
        assert!(project.validate().is_err());
        project.project = Some("app".into());
        assert!(project.validate().is_ok());
        project.kind = "memory".into();
        assert!(project.validate().is_err());
    }
    #[test] fn lesson_receipt_serializes_typed_identity() {
        let value = serde_json::to_value(RememberReceipt { memory_id:0, lesson_id:7, kind:"writing-lesson".into(), room:"room".into(), source_path:"db-only/x".into(), durable:true, authority:"postgres", warnings:vec![] }).unwrap();
        assert_eq!(value["lesson_id"], 7);
        assert_eq!(value["kind"], "writing-lesson");
        assert!(value.get("memory_id").is_none());
    }
    #[test] fn recall_defaults_and_validation(){let p:RecallParams=serde_json::from_value(serde_json::json!({"room":"room","query":"alpha"})).unwrap();assert_eq!(p.semantic_top_k,8);assert_eq!(p.content_top_k,8);assert!(p.validate().is_ok());}
    #[test] fn anamnesis_accepts_shared_house_but_preserves_slug_rules() {
        let house: AnamnesisParams = serde_json::from_value(serde_json::json!({"room":"house","mode":"wake"})).unwrap();
        assert_eq!(house.validate().unwrap().0, "wake");
        let ordinary = AnamnesisParams { room:"Bad Room".into(), mode:"wake".into(), query:String::new(), limit:None };
        assert!(ordinary.validate().is_err());
    }
    #[test] fn anamnesis_result_serializes_exact_read_envelope() {
        let value = serde_json::to_value(AnamnesisResult {
            ok: true, mode:"consult".into(), room:"house".into(), query:"pattern".into(),
            found: false, entries: vec![], warnings: vec!["excluded cycle 4: blank verify_note".into()],
        }).unwrap();
        assert_eq!(value, serde_json::json!({
            "ok": true, "mode":"consult", "room":"house", "query":"pattern",
            "found":false, "entries":[], "warnings":["excluded cycle 4: blank verify_note"]
        }));
    }
    #[test] fn recall_rejects_unknown_empty_and_bounds(){assert!(serde_json::from_value::<RecallParams>(serde_json::json!({"room":"room","query":"x","extra":1})).is_err());let mut p=RecallParams{room:"room".into(),query:" ".into(),semantic_top_k:8,semantic_min_similarity:0.5,content_top_k:8,content_min_similarity:0.3};assert!(p.validate().is_err());p.query="x".into();p.semantic_min_similarity=f64::NAN;assert!(p.validate().is_err());}
    #[test] fn lexical_evidence_uses_wire_term_names_and_is_deterministic() {
        let terms = query_terms("Alpha 2026-07-22 alpha");
        assert_eq!(terms, vec!["07".to_string(), "2026".to_string(), "22".to_string(), "alpha".to_string()]);
        let (matched, missing) = term_evidence(&terms, &["An alpha memory"]);
        assert_eq!(matched, vec!["alpha".to_string()]);
        assert_eq!(missing, vec!["07".to_string(), "2026".to_string(), "22".to_string()]);
        let candidate = serde_json::json!({"matched_terms": matched, "missing_terms": missing, "body_excerpt": "An alpha memory"});
        assert!(candidate.get("matched_terms").is_some());
        assert!(candidate.get("missing_terms").is_some());
        assert!(candidate.get("body_excerpt").is_some());
        assert!(candidate.get("terms").is_none());
        assert!(candidate.get("excerpt").is_none());
    }
    #[test] fn unicode_chunks_use_utf8_bytes(){let c=chunk_body("éé");assert_eq!((c[0].1,c[0].2), (0,4));assert_eq!(&"éé"[c[0].1..c[0].2],"éé");assert!(token_estimate("é")>0);}
    #[test] fn oversized_chunks_preserve_separator_and_span(){
        let first="a".repeat(2200); let body=format!("{first}\n\né{}", "b".repeat(2500));
        let chunks=chunk_body(&body); let (text,start,end,_)=&chunks[1];
        assert_eq!(text, &body[*start..*end]);
        assert!(text.contains("\n\né"));
    }
    #[test] fn derive_dates_uses_injected_primary_date(){let d=NaiveDate::from_ymd_opt(2026,7,22).unwrap();assert!(derive_dates("db-only/room/no-date",d).contains(&d));}
    #[test] fn backup_target_strips_password(){let (target,password)=backup_target("postgres://alice:secret@db.example/memory").unwrap();assert_eq!(password.as_deref(),Some("secret"));assert!(!target.contains("secret"));assert!(target.contains("db.example/memory"));}
    #[test] fn threads_normalize(){assert_eq!(normalize_threads(&[" a ".into()," ".into(),"a".into()]),vec!["a"]);}
    #[test] fn backup_command_targets_resolved_db_without_password_arg(){let command=build_backup_command("postgres://alice:secret@db.example/memory").unwrap();let envs=command.get_envs().filter_map(|(k,v)|Some((k.to_string_lossy().into_owned(),v?.to_string_lossy().into_owned()))).collect::<std::collections::BTreeMap<_,_>>();assert_eq!(envs.get("SOLARISAEL_BACKUP_DATABASE_URL").map(String::as_str),Some("postgres://alice@db.example/memory"));assert_eq!(envs.get("SOLARISAEL_BACKUP_PASSWORD").map(String::as_str),Some("secret"));}
    #[test] fn bounded_excerpt_is_character_safe_and_limited(){let body="é".repeat(1300);let excerpt=bounded_excerpt(&body);assert!(excerpt.chars().count()<=1201);assert!(excerpt.ends_with('…'));}
    #[test] fn candidate_term_coverage_is_exact(){let (matched,missing,coverage)=candidate_terms(&["alpha".into(),"beta".into()],&["alpha body"]);assert_eq!(matched,vec!["alpha"]);assert_eq!(missing,vec!["beta"]);assert_eq!(coverage,0.5);}
}


/// Cluster maintenance is deliberately owned by Rust.  The pure planner is kept
/// separate from SQL so deterministic behavior can be tested without PostgreSQL.
#[derive(Debug, Clone, Serialize)]
pub struct ClusterMaintenanceResult {
    pub ok: bool,
    pub operation: String,
    #[serde(rename = "dryRun")]
    pub dry_run: bool,
    pub stale: bool,
    pub clusters: usize,
    pub members: usize,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ClusterStaleness {
    pub built_at: Option<chrono::DateTime<chrono::Utc>>,
    pub clusters: i64,
    pub chunks_total: i64,
    pub chunks_since_build: i64,
    pub fraction_unseen: f64,
}

pub fn cluster_is_stale(s: &ClusterStaleness, now: chrono::DateTime<chrono::Utc>) -> bool {
    if s.built_at.is_none() { return true; }
    let count = s.chunks_since_build >= 250;
    let fraction = s.chunks_total > 0 && (s.chunks_since_build as f64 / s.chunks_total as f64) >= 0.05;
    let age = now.signed_duration_since(s.built_at.unwrap()).num_days() >= 7 && s.chunks_since_build > 0;
    count || fraction || age
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if n == 0.0 { return vec![0.0; v.len()]; }
    v.iter().map(|x| (*x as f64 / n) as f32).collect()
}
fn cosine(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x,y)| x*y).sum() }

/// Deterministic spherical k-means (seed 42 is represented by stable farthest
/// initialization; no RNG means equal inputs always produce equal output).
pub fn spherical_kmeans(input: &[(i64, Vec<f32>)], requested_k: usize) -> Vec<(Vec<f32>, Vec<(i64, f64)>)> {
    if input.is_empty() { return Vec::new(); }
    let points: Vec<(i64, Vec<f32>)> = input.iter().map(|(id,v)| (*id, unit(v))).collect();
    let k = requested_k.max(1).min(points.len());
    let mut centers = vec![points[0].1.clone()];
    while centers.len() < k {
        let (idx, _) = points.iter().enumerate().map(|(i,(_,v))| (i, centers.iter().map(|c| cosine(v,c)).fold(-1.0f32, f32::max)))
            .min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))).unwrap();
        centers.push(points[idx].1.clone());
    }
    let mut assignment = vec![usize::MAX; points.len()];
    for _ in 0..32 {
        let next: Vec<usize> = points.iter().map(|(_,v)| centers.iter().enumerate()
            .max_by(|(ia,a),(ib,b)| cosine(v,a).partial_cmp(&cosine(v,b)).unwrap_or(std::cmp::Ordering::Equal).then_with(|| ia.cmp(ib)))
            .map(|(i,_)| i).unwrap()).collect();
        if next == assignment { break; }
        assignment = next;
        for c in 0..k {
            let mut sum = vec![0.0f32; points[0].1.len()];
            for (i,(_,v)) in points.iter().enumerate().filter(|(i,_)| assignment[*i] == c) {
                for (j,x) in v.iter().enumerate() { sum[j] += x; }
            }
            if sum.iter().any(|x| *x != 0.0) { centers[c] = unit(&sum); }
        }
    }
    let mut out = centers.into_iter().map(|c| (c, Vec::new())).collect::<Vec<_>>();
    for (i,(id,v)) in points.iter().enumerate() {
        let c = assignment[i];
        let distance = 1.0 - cosine(v, &out[c].0);
        out[c].1.push((*id, distance as f64));
    }
    out
}

pub async fn cluster_staleness(pool: &PgPool, room: Option<&str>) -> Result<ClusterStaleness, AppError> {
    let built: (Option<chrono::DateTime<chrono::Utc>>, i64) = sqlx::query_as("SELECT max(created_at), count(*) FROM memory_clusters").fetch_one(pool).await?;
    let scope = room.map(|_| " AND m.room = $1").unwrap_or("");
    let sql = format!("SELECT count(*) FROM memory_chunks c JOIN memories m ON m.id=c.memory_id WHERE c.body_embedding IS NOT NULL AND m.archived_at IS NULL AND m.superseded_by IS NULL{scope}");
    let total: i64 = if let Some(r) = room { sqlx::query_scalar(&sql).bind(r).fetch_one(pool).await? } else { sqlx::query_scalar(&sql).fetch_one(pool).await? };
    let since: i64 = if let Some(at) = built.0 {
        if room.is_some() {
            sqlx::query_scalar("SELECT count(*) FROM memory_chunks c JOIN memories m ON m.id=c.memory_id WHERE c.body_embedding IS NOT NULL AND m.archived_at IS NULL AND m.superseded_by IS NULL AND c.embedded_at > $1 AND m.room = $2").bind(at).bind(room.unwrap()).fetch_one(pool).await?
        } else {
            sqlx::query_scalar("SELECT count(*) FROM memory_chunks c JOIN memories m ON m.id=c.memory_id WHERE c.body_embedding IS NOT NULL AND m.archived_at IS NULL AND m.superseded_by IS NULL AND c.embedded_at > $1").bind(at).fetch_one(pool).await?
        }
    } else { total };
    Ok(ClusterStaleness { built_at: built.0, clusters: built.1, chunks_total: total, chunks_since_build: since, fraction_unseen: if total == 0 {0.0} else {since as f64 / total as f64} })
}

pub async fn cluster_maintenance(pool: &PgPool, operation: &str, dry_run: bool, if_stale: bool, k: usize) -> Result<ClusterMaintenanceResult, AppError> {
    if !matches!(operation, "check" | "rebuild") { return Err(AppError::Invalid("operation must be check or rebuild".into())); }
    let stale_info = cluster_staleness(pool, None).await?;
    let stale = cluster_is_stale(&stale_info, chrono::Utc::now());
    if operation == "check" || (if_stale && !stale) {
        return Ok(ClusterMaintenanceResult { ok:true, operation:operation.into(), dry_run, stale, clusters:stale_info.clusters as usize, members:0, warnings:Vec::new() });
    }
    if dry_run { return Ok(ClusterMaintenanceResult { ok:true, operation:operation.into(), dry_run:true, stale, clusters:0, members:0, warnings:Vec::new() }); }
    let mut tx = pool.begin().await?;
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended('solarisael.cluster_maintenance', 42))").execute(&mut *tx).await?;
    let rows = sqlx::query("SELECT c.id,c.body_embedding::text FROM memory_chunks c JOIN memories m ON m.id=c.memory_id WHERE c.body_embedding IS NOT NULL AND m.archived_at IS NULL AND m.superseded_by IS NULL ORDER BY c.id").fetch_all(&mut *tx).await?;
    let mut points = Vec::new();
    for row in rows {
        let text: String = row.try_get("body_embedding")?;
        let vec = text.trim_matches(|c| c=='[' || c==']').split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect::<Vec<_>>();
        if vec.len() != EMBED_DIMENSION { return Err(AppError::Config("cluster embedding dimension is not vector(2048)".into())); }
        points.push((row.try_get::<i64,_>("id")?, vec));
    }
    let groups = spherical_kmeans(&points, k);
    sqlx::query("DELETE FROM memory_clusters").execute(&mut *tx).await?;
    for (center, members) in &groups {
        let center_text = format!("[{}]", center.iter().map(ToString::to_string).collect::<Vec<_>>().join(","));
        let cid: i64 = sqlx::query_scalar("INSERT INTO memory_clusters (label,centroid,member_count,accepted) VALUES ($1,$2::vector,$3,FALSE) RETURNING id").bind("cluster").bind(center_text).bind(members.len() as i32).fetch_one(&mut *tx).await?;
        for (id,distance) in members { sqlx::query("INSERT INTO memory_cluster_members (cluster_id,chunk_id,distance_to_centroid) VALUES ($1,$2,$3)").bind(cid).bind(id).bind(distance).execute(&mut *tx).await?; }
    }
    tx.commit().await?;
    Ok(ClusterMaintenanceResult { ok:true, operation:operation.into(), dry_run:false, stale:true, clusters:groups.len(), members:points.len(), warnings:Vec::new() })
}

#[cfg(test)]
mod cluster_tests {
    use super::*;
    #[test]
    fn stale_policy_boundaries_and_never_built() {
        let now = chrono::Utc::now();
        assert!(cluster_is_stale(&ClusterStaleness { built_at:None, clusters:0, chunks_total:0, chunks_since_build:0, fraction_unseen:0.0 }, now));
        assert!(!cluster_is_stale(&ClusterStaleness { built_at:Some(now), clusters:1, chunks_total:100, chunks_since_build:4, fraction_unseen:0.04 }, now));
        assert!(cluster_is_stale(&ClusterStaleness { built_at:Some(now), clusters:1, chunks_total:100, chunks_since_build:5, fraction_unseen:0.05 }, now));
        assert!(cluster_is_stale(&ClusterStaleness { built_at:Some(now - chrono::Duration::days(8)), clusters:1, chunks_total:1000, chunks_since_build:1, fraction_unseen:0.001 }, now));
    }
    #[test]
    fn kmeans_is_deterministic_and_safe_for_small_inputs() {
        let a = vec![(1, vec![1.0,0.0]), (2, vec![0.9,0.1]), (3, vec![0.0,1.0])];
        let x = spherical_kmeans(&a, 8);
        let y = spherical_kmeans(&a, 8);
        assert_eq!(x, y);
        assert_eq!(x.len(), 3);
        assert_eq!(x.iter().map(|(_,m)| m.len()).sum::<usize>(), 3);
        assert!(spherical_kmeans(&[], 8).is_empty());
    }
}

async fn cluster_resonance(pool: &PgPool, vector_text: &str, rooms: &[String]) -> Result<serde_json::Value, AppError> {
    let rows = sqlx::query("SELECT mc.id,mc.label,COUNT(mm.chunk_id)::bigint AS member_count,(1-(mc.centroid <=> $1::vector))::double precision AS activation FROM memory_clusters mc JOIN memory_cluster_members mm ON mm.cluster_id=mc.id JOIN memory_chunks c ON c.id=mm.chunk_id JOIN memories m ON m.id=c.memory_id WHERE mc.centroid IS NOT NULL AND m.room=ANY($2::text[]) AND m.archived_at IS NULL AND m.superseded_by IS NULL GROUP BY mc.id,mc.label,mc.centroid ORDER BY activation DESC LIMIT 8").bind(vector_text).bind(rooms).fetch_all(pool).await?;
    let mut profile = Vec::new();
    let mut hot = Vec::new();
    for (index, r) in rows.iter().enumerate() {
        let id: i64 = r.try_get("id")?;
        let label: Option<String> = r.try_get("label")?;
        profile.push(serde_json::json!({"cluster_id":id,"label":label,"member_count":r.try_get::<i64,_>("member_count")?,"activation":r.try_get::<f64,_>("activation")?.clamp(-1.0,1.0)}));
        if index < 3 {
            let chunks = sqlx::query("SELECT m.source_path,c.heading_path,(1-(c.body_embedding <=> $1::vector))::double precision AS sim FROM memory_cluster_members mm JOIN memory_chunks c ON c.id=mm.chunk_id JOIN memories m ON m.id=c.memory_id WHERE mm.cluster_id=$2 AND m.room=ANY($3::text[]) AND m.archived_at IS NULL AND m.superseded_by IS NULL ORDER BY sim DESC LIMIT 2").bind(vector_text).bind(id).bind(rooms).fetch_all(pool).await?;
            let pointers = chunks.into_iter().map(|c| serde_json::json!({"source_path":c.try_get::<String,_>("source_path").unwrap_or_default(),"heading_path":c.try_get::<Option<String>,_>("heading_path").ok().flatten(),"sim":c.try_get::<f64,_>("sim").unwrap_or(0.0).clamp(-1.0,1.0)})).collect::<Vec<_>>();
            if !pointers.is_empty() { hot.push(serde_json::json!({"cluster_id":id,"label":label,"chunks":pointers})); }
        }
    }
    Ok(serde_json::json!({"profile":profile,"hot":hot}))
}
