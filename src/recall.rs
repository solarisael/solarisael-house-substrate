use crate::cluster::{cluster_resonance, cluster_staleness};
use crate::config::{AppError, Config, EMBED_DIMENSION, HTTP_CLIENT, QUERY_DATE_RE, ROOM_KEY_RE};
use chrono::NaiveDate;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::collections::{BTreeMap, BTreeSet};

fn default_semantic_top_k() -> u32 {
    8
}
fn default_semantic_min_similarity() -> f64 {
    0.50
}
fn default_content_top_k() -> u32 {
    8
}
fn default_content_min_similarity() -> f64 {
    0.30
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecallParams {
    pub room: String,
    pub query: String,
    #[serde(default = "default_semantic_top_k")]
    pub semantic_top_k: u32,
    #[serde(default = "default_semantic_min_similarity")]
    pub semantic_min_similarity: f64,
    #[serde(default = "default_content_top_k")]
    pub content_top_k: u32,
    #[serde(default = "default_content_min_similarity")]
    pub content_min_similarity: f64,
}

impl RecallParams {
    pub fn validate(&self) -> Result<(), AppError> {
        if !ROOM_KEY_RE.is_match(&self.room) || self.room == "house" {
            return Err(AppError::Invalid(
                "room must be a lowercase slug and cannot be house".into(),
            ));
        }
        if self.query.trim().is_empty() {
            return Err(AppError::Invalid("query must not be empty".into()));
        }
        for (name, value) in [
            ("semantic_top_k", self.semantic_top_k),
            ("content_top_k", self.content_top_k),
        ] {
            if value == 0 || value > 1000 {
                return Err(AppError::Invalid(format!(
                    "{name} must be positive and at most 1000"
                )));
            }
        }
        for (name, value) in [
            ("semantic_min_similarity", self.semantic_min_similarity),
            ("content_min_similarity", self.content_min_similarity),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(AppError::Invalid(format!(
                    "{name} must be finite and in [0, 1]"
                )));
            }
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
    #[serde(rename = "retrievalCandidates")]
    pub retrieval_candidates: Vec<serde_json::Value>,
    #[serde(rename = "canonMatches")]
    pub canon_matches: Vec<serde_json::Value>,
    #[serde(rename = "semanticChunks")]
    pub semantic_chunks: Vec<serde_json::Value>,
    #[serde(rename = "contentChunks")]
    pub content_chunks: Vec<serde_json::Value>,
    #[serde(rename = "dateMatches")]
    pub date_matches: Vec<serde_json::Value>,
    #[serde(rename = "queryDates")]
    pub query_dates: Vec<serde_json::Value>,
    pub taxonomy: serde_json::Value,
    #[serde(rename = "clusterStaleness", skip_serializing_if = "Option::is_none")]
    pub cluster_staleness: Option<serde_json::Value>,
    #[serde(rename = "clusterResonance", skip_serializing_if = "Option::is_none")]
    pub cluster_resonance: Option<serde_json::Value>,
}

fn query_dates(query: &str) -> Vec<NaiveDate> {
    QUERY_DATE_RE
        .captures_iter(query)
        .filter_map(|c| {
            NaiveDate::from_ymd_opt(c[1].parse().ok()?, c[2].parse().ok()?, c[3].parse().ok()?)
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
pub(crate) fn query_terms(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .map(|term| term.trim().to_ascii_lowercase())
        .filter(|term| term.len() >= 2)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn term_evidence(terms: &[String], fields: &[&str]) -> (Vec<String>, Vec<String>) {
    let haystack = fields.join(" ").to_ascii_lowercase();
    let matched = terms
        .iter()
        .filter(|term| haystack.contains(term.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let missing = terms
        .iter()
        .filter(|term| !matched.contains(term))
        .cloned()
        .collect::<Vec<_>>();
    (matched, missing)
}

async fn embed_query(
    client: &Client,
    url: &str,
    model: &str,
    query: &str,
    dim: usize,
) -> Result<Vec<f32>, AppError> {
    #[derive(Serialize)]
    struct Input<'a> {
        model: &'a str,
        input: Vec<String>,
    }
    let value: serde_json::Value = client
        .post(url)
        .json(&Input {
            model,
            input: vec![format!("query: {query}")],
        })
        .send()
        .await
        .map_err(|e| AppError::Embedding(e.to_string()))?
        .error_for_status()
        .map_err(|e| AppError::Embedding(e.to_string()))?
        .json()
        .await
        .map_err(|e| AppError::Embedding(e.to_string()))?;
    let item = value
        .get("embeddings")
        .or_else(|| value.get("data"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| AppError::Embedding("response lacks query embedding".into()))?;
    let v = item
        .as_array()
        .or_else(|| item.get("embedding").and_then(|x| x.as_array()))
        .ok_or_else(|| AppError::Embedding("invalid query embedding".into()))?;
    let row = v
        .iter()
        .map(|x| {
            x.as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| AppError::Embedding("non-numeric query embedding".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if row.len() != dim {
        return Err(AppError::Embedding(format!(
            "embedding dimension {} != {}",
            row.len(),
            dim
        )));
    }
    Ok(row)
}

pub(crate) fn bounded_excerpt(body: &str) -> String {
    const MAX: usize = 1200;
    let excerpt: String = body.chars().take(MAX).collect();
    if body.chars().count() > MAX {
        format!("{excerpt}…")
    } else {
        excerpt
    }
}

pub(crate) fn candidate_terms(
    terms: &[String],
    fields: &[&str],
) -> (Vec<String>, Vec<String>, f64) {
    let (matched, missing) = term_evidence(terms, fields);
    let coverage = if terms.is_empty() {
        0.0
    } else {
        matched.len() as f64 / terms.len() as f64
    };
    (matched, missing, coverage)
}
pub async fn recall(
    pool: &PgPool,
    cfg: &Config,
    params: RecallParams,
) -> Result<RecallResult, AppError> {
    params.validate()?;
    let query_dates = query_dates(&params.query);
    let query_terms = query_terms(&params.query);
    let rooms = vec![params.room.clone(), "house".to_string()];
    let mut warnings = Vec::new();
    let vector_text = match (cfg.test_embedding_disabled, cfg.embed_url.as_deref()) {
        (true, _) => {
            warnings.push("semantic lane absent: embedding disabled".to_string());
            None
        }
        (false, Some(url)) => match embed_query(
            &HTTP_CLIENT,
            url,
            &cfg.embed_model,
            &params.query,
            EMBED_DIMENSION,
        )
        .await
        {
            Ok(vector) => Some(format!(
                "[{}]",
                vector
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )),
            Err(e) => {
                warnings.push(format!("semantic lane absent: {e}"));
                None
            }
        },
        (false, None) => {
            warnings.push("semantic lane absent: embedding endpoint is required".to_string());
            None
        }
    };
    let mut semantic_chunks = Vec::new();
    if let Some(vector_text) = vector_text.clone() {
        let semantic_rows = sqlx::query("SELECT m.source_path,coalesce(m.title,'') AS title,coalesce(c.heading_path,'') AS heading_path,c.body,c.char_start,c.char_end,c.chunk_index,(1-(c.body_embedding <=> $1::vector))::double precision AS sim FROM memory_chunks c JOIN memories m ON m.id=c.memory_id WHERE m.room = ANY($2::text[]) AND m.archived_at IS NULL AND m.superseded_by IS NULL AND c.body_embedding IS NOT NULL ORDER BY sim DESC,m.source_path,c.chunk_index LIMIT $3").bind(&vector_text).bind(&rooms).bind(params.semantic_top_k as i64).fetch_all(pool).await?;
        for row in semantic_rows {
            let sim: f64 = row.try_get("sim")?;
            if sim < params.semantic_min_similarity {
                continue;
            }
            let source_path: String = row.try_get("source_path")?;
            let title: Option<String> = row.try_get("title")?;
            let heading_path: Option<String> = row.try_get("heading_path")?;
            let body: String = row.try_get("body")?;
            let (matched_terms, missing_terms, coverage) = candidate_terms(
                &query_terms,
                &[
                    &source_path,
                    title.as_deref().unwrap_or(""),
                    heading_path.as_deref().unwrap_or(""),
                    &body,
                ],
            );
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
        let (matched_terms, missing_terms, coverage) = candidate_terms(
            &query_terms,
            &[
                &source_path,
                title.as_deref().unwrap_or(""),
                heading_path.as_deref().unwrap_or(""),
                &body,
            ],
        );
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
            let (matched_terms, missing_terms, coverage) = candidate_terms(
                &query_terms,
                &[&source_path, title.as_deref().unwrap_or(""), &body],
            );
            date_matches.push(serde_json::json!({"source_path":source_path,"title":title,"body_excerpt":bounded_excerpt(&body),"excerpt":bounded_excerpt(&body),"date":row.try_get::<Option<NaiveDate>,_>("date")?.map(|d|d.to_string()),"dates":dates.into_iter().map(|d|d.to_string()).collect::<Vec<_>>(),"score":1.0,"reason":"date match","matched_terms":matched_terms,"missing_terms":missing_terms,"term_coverage":coverage}));
        }
    }
    let mut fused: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for (rank, c) in semantic_chunks.iter().enumerate() {
        let key = format!(
            "{}#{}",
            c["source_path"].as_str().unwrap_or(""),
            c["chunk_index"].as_i64().unwrap_or(0)
        );
        let score = c["sim"].as_f64().unwrap_or(0.0) * 0.6 + 1.0 / (rank as f64 + 1.0) * 0.4;
        fused.insert(key, serde_json::json!({"source_path":c["source_path"],"title":c["title"],"heading_path":c["heading_path"],"excerpt":c["body"],"sources":[c["source_path"]],"term_coverage":c["term_coverage"],"matched_terms":c["matched_terms"],"missing_terms":c["missing_terms"],"score":score,"semantic_score":c["sim"],"reasons":["semantic cosine similarity"],"source":"semantic","chunk_index":c["chunk_index"]}));
    }
    for (rank, c) in content_chunks.iter().enumerate() {
        let key = format!(
            "{}#{}",
            c["source_path"].as_str().unwrap_or(""),
            c["chunk_index"].as_i64().unwrap_or(0)
        );
        let score = c["ws"].as_f64().unwrap_or(0.0) * 0.6 + 1.0 / (rank as f64 + 1.0) * 0.4;
        if let Some(existing) = fused.get_mut(&key) {
            existing["score"] =
                serde_json::json!(existing["score"].as_f64().unwrap_or(0.0) + score);
            existing["content_score"] = c["ws"].clone();
            existing["source"] = serde_json::json!("semantic+content");
            existing["reasons"] =
                serde_json::json!(["semantic cosine similarity", "lexical word_similarity"]);
        } else {
            fused.insert(key, serde_json::json!({"source_path":c["source_path"],"title":c["title"],"heading_path":c["heading_path"],"excerpt":c["body"],"sources":[c["source_path"]],"term_coverage":c["term_coverage"],"matched_terms":c["matched_terms"],"missing_terms":c["missing_terms"],"score":score,"content_score":c["ws"],"reasons":["lexical word_similarity"],"source":"content","chunk_index":c["chunk_index"]}));
        }
    }
    let mut retrieval_candidates: Vec<_> = fused.into_values().collect();
    retrieval_candidates.sort_by(|a, b| {
        b["score"]
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&a["score"].as_f64().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a["source_path"].as_str().cmp(&b["source_path"].as_str()))
            .then_with(|| a["chunk_index"].as_i64().cmp(&b["chunk_index"].as_i64()))
    });
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
    let cluster_staleness = cluster_staleness(pool, None)
        .await
        .ok()
        .and_then(|s| serde_json::to_value(s).ok());
    let cluster_resonance = if let Some(v) = vector_text.as_deref() {
        cluster_resonance(pool, v, &rooms).await.ok()
    } else {
        None
    };
    Ok(RecallResult {
        ok: true,
        query: params.query,
        found: !retrieval_candidates.is_empty()
            || !canon_matches.is_empty()
            || !date_matches.is_empty(),
        source: "rust-postgres",
        warnings,
        retrieval_candidates,
        canon_matches,
        semantic_chunks,
        content_chunks,
        date_matches,
        query_dates: query_dates
            .into_iter()
            .map(|d| serde_json::json!(d.to_string()))
            .collect(),
        taxonomy,
        cluster_staleness,
        cluster_resonance,
    })
}
