use crate::backup;
use crate::config::{
    AppError, Config, HTTP_CLIENT, PATH_DATE_RE, ROOM_KEY_RE, STITCHED_PATH_DATE_RE,
};
use chrono::{Local, NaiveDate};
use reqwest::Client;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};
use sqlx::PgPool;
use std::collections::BTreeSet;
use uuid::Uuid;

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
pub(crate) fn default_backup() -> bool {
    true
}

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
        if !ROOM_KEY_RE.is_match(&self.room) || self.room == "house" {
            return Err(AppError::Invalid(
                "room must be a lowercase slug and cannot be house".into(),
            ));
        }
        let lessons = [
            "coding-lesson",
            "project-lesson",
            "writing-lesson",
            "audio-lesson",
        ];
        let text = self.lesson.as_deref().unwrap_or(&self.body);
        if self.title.trim().is_empty() {
            return Err(AppError::Invalid("title must not be empty".into()));
        }
        if text.trim().is_empty() {
            return Err(AppError::Invalid("body/lesson must not be empty".into()));
        }
        if self.kind == "memory" {
            if self.lesson.is_some()
                || self.source_memory_path.is_some()
                || self.shape.is_some()
                || self.voice.is_some()
                || self.scope.is_some()
                || self.project.is_some()
                || self.proof_pattern.is_some()
                || self.trigger_context.is_some()
                || !self.tags.is_empty()
            {
                return Err(AppError::Invalid(
                    "lesson-only fields are not valid for memory".into(),
                ));
            }
            if self
                .source_path
                .as_ref()
                .is_some_and(|p| p.trim().is_empty())
            {
                return Err(AppError::Invalid("source_path must not be empty".into()));
            }
            if self.supersedes.iter().any(|id| *id <= 0) {
                return Err(AppError::Invalid(
                    "supersedes must contain positive IDs".into(),
                ));
            }
        } else if lessons.contains(&self.kind.as_str()) {
            if !self.threads.is_empty() || !self.supersedes.is_empty() || self.source_path.is_some()
            {
                return Err(AppError::Invalid(
                    "threads/supersedes/source_path are memory-only".into(),
                ));
            }
            if self
                .source_memory_path
                .as_ref()
                .is_some_and(|p| p.trim().is_empty())
            {
                return Err(AppError::Invalid(
                    "source_memory_path must not be empty".into(),
                ));
            }
            let unsupported = match self.kind.as_str() {
                "coding-lesson" => false,
                "project-lesson" => self.voice.is_some() || self.scope.is_some(),
                "writing-lesson" => {
                    self.scope.is_some() || self.project.is_some() || self.proof_pattern.is_some()
                }
                "audio-lesson" => {
                    self.voice.is_some()
                        || self.scope.is_some()
                        || self.project.is_some()
                        || self.proof_pattern.is_some()
                }
                _ => true,
            };
            if unsupported {
                return Err(AppError::Invalid(
                    "lesson fields are unsupported by this lesson table".into(),
                ));
            }
            if self.kind == "project-lesson"
                && self.project.as_deref().unwrap_or("").trim().is_empty()
            {
                return Err(AppError::Invalid(
                    "project is required for project lessons".into(),
                ));
            }
        } else {
            return Err(AppError::Invalid("unsupported remember kind".into()));
        }
        Ok(())
    }
    pub(crate) fn source_path(&self) -> String {
        self.source_path
            .clone()
            .unwrap_or_else(|| format!("db-only/{}/{}", self.room, Uuid::new_v4()))
    }
    fn lesson_body(&self) -> &str {
        self.lesson.as_deref().unwrap_or(&self.body)
    }
}

pub async fn remember(
    pool: &PgPool,
    cfg: &Config,
    req: RememberRequest,
) -> Result<RememberReceipt, AppError> {
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
    sqlx::query("DELETE FROM memory_threads WHERE memory_id=$1")
        .bind(memory_id)
        .execute(&mut *tx)
        .await?;
    for thread in &threads {
        sqlx::query("INSERT INTO memory_threads (memory_id,thread_key) VALUES ($1,$2)")
            .bind(memory_id)
            .bind(thread)
            .execute(&mut *tx)
            .await?;
    }
    for old_id in req.supersedes.iter().copied().collect::<BTreeSet<_>>() {
        sqlx::query("UPDATE memories SET superseded_by=$1 WHERE id=$2 AND id<>$1")
            .bind(memory_id)
            .bind(old_id)
            .execute(&mut *tx)
            .await?;
    }
    sqlx::query("DELETE FROM memory_chunks WHERE memory_id=$1")
        .bind(memory_id)
        .execute(&mut *tx)
        .await?;
    let chunks = chunk_body(&req.body);
    let mut warnings = Vec::new();
    if cfg.test_embedding_disabled {
        warnings.push("embedding disabled for isolated test; chunks cleared".into());
    } else {
        let url = cfg
            .embed_url
            .as_deref()
            .ok_or_else(|| AppError::Embedding("embedding endpoint is required".into()))?;
        let vectors = embed(
            &HTTP_CLIENT,
            url,
            &cfg.embed_model,
            &chunks,
            cfg.embed_dimension,
        )
        .await?;
        if vectors.len() != chunks.len() {
            return Err(AppError::Embedding("embedding count mismatch".into()));
        }
        for (idx, (text, start, end, heading)) in chunks.iter().enumerate() {
            let vector_text = format!(
                "[{}]",
                vectors[idx]
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            );
            sqlx::query("INSERT INTO memory_chunks (memory_id,chunk_index,heading_path,body,char_start,char_end,token_estimate,body_embedding,embedded_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8::vector,NOW())")
                .bind(memory_id).bind(idx as i32).bind(heading).bind(text).bind(*start as i32).bind(*end as i32).bind(token_estimate(text)).bind(vector_text).execute(&mut *tx).await?;
        }
    }
    tx.commit().await?;
    if req.backup
        && let Err(error) = backup::run_post_write(pool, &cfg.database_url).await
    {
        warnings.push(format!("backup failed: {error}"));
    }
    Ok(RememberReceipt {
        memory_id,
        lesson_id: 0,
        kind: "memory".into(),
        room: req.room,
        source_path,
        durable: true,
        authority: "postgres",
        warnings,
    })
}

async fn remember_lesson(
    pool: &PgPool,
    cfg: &Config,
    req: &RememberRequest,
) -> Result<RememberReceipt, AppError> {
    let text = req.lesson_body();
    let tags = req
        .tags
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
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
    if req.backup
        && matches!(req.kind.as_str(), "project-lesson" | "audio-lesson")
        && let Err(error) = backup::run_post_write(pool, &cfg.database_url).await
    {
        warnings.push(format!("backup failed: {error}"));
    }
    Ok(RememberReceipt {
        memory_id: 0,
        lesson_id: id,
        kind: req.kind.clone(),
        room: req.room.clone(),
        source_path: String::new(),
        durable: true,
        authority: "postgres",
        warnings,
    })
}

pub(crate) fn normalize_threads(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn derive_dates(path: &str, primary_date: NaiveDate) -> Vec<NaiveDate> {
    let mut out = vec![primary_date];
    for captures in PATH_DATE_RE.captures_iter(path) {
        if let (Ok(year), Ok(month), Ok(day)) = (
            captures[1].parse(),
            captures[2].parse(),
            captures[3].parse(),
        ) && let Some(date) = NaiveDate::from_ymd_opt(year, month, day)
        {
            out.push(date);
        }
    }
    for captures in STITCHED_PATH_DATE_RE.captures_iter(path) {
        if let (Ok(year), Ok(month), Ok(day), Ok(hour)) = (
            captures[1].parse::<i32>(),
            captures[2].parse::<u32>(),
            captures[3].parse::<u32>(),
            captures[4].parse::<u32>(),
        ) && hour < 24
            && let Some(date) = NaiveDate::from_ymd_opt(year, month, day)
        {
            out.push(date + chrono::Duration::days(1));
        }
    }
    out.sort();
    out.dedup();
    out
}

pub(crate) fn chunk_body(body: &str) -> Vec<(String, usize, usize, Option<String>)> {
    if body.is_empty() {
        return vec![];
    }
    let chars: Vec<char> = body.chars().collect();
    let mut headings: Vec<(usize, String)> = Vec::new();
    let mut offset = 0usize;
    for line in body.split_inclusive('\n') {
        let content = line.trim_end_matches('\n');
        if content
            .strip_prefix("## ")
            .is_some_and(|h| !h.trim().is_empty())
        {
            headings.push((offset, content.trim().to_string()));
        }
        offset += line.chars().count();
    }
    let mut sections = Vec::new();
    if headings.is_empty() {
        sections.push((0, chars.len(), "__preamble__".to_string()));
    } else {
        if headings[0].0 > 0 {
            sections.push((0, headings[0].0, "__preamble__".to_string()));
        }
        for (i, (start, heading)) in headings.iter().enumerate() {
            sections.push((
                *start,
                headings.get(i + 1).map(|(s, _)| *s).unwrap_or(chars.len()),
                heading.clone(),
            ));
        }
    }
    let byte_at = |char_index: usize| -> usize {
        body.char_indices()
            .nth(char_index)
            .map(|(i, _)| i)
            .unwrap_or(body.len())
    };
    let mut out = Vec::new();
    for (start, end, heading) in sections {
        let text: String = chars[start..end].iter().collect();
        if text.chars().count() <= 4000 {
            if !text.trim().is_empty() {
                out.push((text, byte_at(start), byte_at(end), Some(heading)));
            }
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
            if !piece.trim().is_empty() {
                out.push((
                    piece,
                    byte_at(piece_start),
                    byte_at(piece_end),
                    Some(heading.clone()),
                ));
            }
        }
    }
    out
}
pub(crate) fn token_estimate(text: &str) -> i32 {
    (text.chars().count() / 4).max(1) as i32
}

pub(crate) async fn embed(
    client: &Client,
    url: &str,
    model: &str,
    chunks: &[(String, usize, usize, Option<String>)],
    dim: usize,
) -> Result<Vec<Vec<f32>>, AppError> {
    #[derive(Serialize)]
    struct Input<'a> {
        model: &'a str,
        input: Vec<String>,
    }
    let input = chunks.iter().map(|x| format!("passage: {}", x.0)).collect();
    let value: serde_json::Value = client
        .post(url)
        .json(&Input { model, input })
        .send()
        .await
        .map_err(|e| AppError::Embedding(e.to_string()))?
        .error_for_status()
        .map_err(|e| AppError::Embedding(e.to_string()))?
        .json()
        .await
        .map_err(|e| AppError::Embedding(e.to_string()))?;
    let arr = value
        .get("embeddings")
        .or_else(|| value.get("data"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| AppError::Embedding("response lacks embeddings".into()))?;
    let mut out = Vec::new();
    for item in arr {
        let v = item
            .as_array()
            .or_else(|| item.get("embedding").and_then(|x| x.as_array()))
            .ok_or_else(|| AppError::Embedding("invalid embedding vector".into()))?;
        let row: Vec<f32> = v
            .iter()
            .map(|x| {
                x.as_f64()
                    .map(|n| n as f32)
                    .ok_or_else(|| AppError::Embedding("non-numeric embedding".into()))
            })
            .collect::<Result<_, _>>()?;
        if row.len() != dim {
            return Err(AppError::Embedding(format!(
                "embedding dimension {} != {}",
                row.len(),
                dim
            )));
        }
        out.push(row);
    }
    Ok(out)
}
