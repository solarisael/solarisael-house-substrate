use crate::cluster::{cluster_resonance, cluster_staleness};
use crate::config::{AppError, Config, EMBED_DIMENSION, HTTP_CLIENT, QUERY_DATE_RE, ROOM_KEY_RE};
use chrono::{DateTime, NaiveDate, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

fn default_semantic_top_k() -> u32 {
    8
}
// Calibration bound to the embedding model, not a strictness knob. On
// Nemotron-3-Embed-1B/Q4 correct rank-1 matches land at 0.42-0.56 and noise tops
// out at 0.24, so the earlier 0.50 sat inside the signal band and cut correct
// hits. Re-measure on any model/quantization change — coding lesson 222.
fn default_semantic_min_similarity() -> f64 {
    0.40
}
fn default_content_top_k() -> u32 {
    8
}
fn default_content_min_similarity() -> f64 {
    0.30
}
fn default_temporal_decay() -> bool {
    false
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
    #[serde(default = "default_temporal_decay")]
    pub temporal_decay: bool,
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
        .map(|term| term.trim().to_lowercase())
        .filter(|term| term.len() >= 2)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn term_evidence(terms: &[String], fields: &[&str]) -> (Vec<String>, Vec<String>) {
    let haystack = fields.join(" ").to_lowercase();
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

// The `query: ` / `passage: ` prefix pair is load-bearing. Do not remove it.
// Without prefixes this model floors unrelated text near 0.50: absolute values
// look higher while separating nothing, and the true/noise gap goes negative.
// The indexer's `passage: ` and this `query: ` are one decision — project
// lesson 126 (the-athanor) carries the measured comparison.
async fn embed_query(
    client: &Client,
    url: &str,
    model: &str,
    query: &str,
    dim: usize,
) -> Result<Vec<f32>, AppError> {
    // The embedder is on the hot path — every user prompt embeds a query. Ollama's
    // default 5 minute keep_alive evicted it during idle gaps and charged a cold
    // reload on the next turn, which is felt directly as latency. Residency costs
    // 2.1 GB of VRAM; measured 2026-07-25.
    #[derive(Serialize)]
    struct Input<'a> {
        model: &'a str,
        input: Vec<String>,
        keep_alive: &'a str,
    }
    let value: serde_json::Value = client
        .post(url)
        .timeout(Duration::from_secs(20))
        .json(&Input {
            model,
            input: vec![format!("query: {query}")],
            keep_alive: "30m",
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

fn giga_temporal_factor(meta: &serde_json::Value, now: DateTime<Utc>) -> (Option<f64>, f64) {
    let Some(object) = meta.as_object() else {
        return (None, 1.0);
    };
    if object.get("origin").and_then(|value| value.as_str()) != Some("giga-promotion") {
        return (None, 1.0);
    }
    let Some(giga) = object.get("giga").and_then(|value| value.as_object()) else {
        return (None, 1.0);
    };
    let Some(durability) = giga.get("durability").and_then(|value| value.as_f64()) else {
        return (None, 1.0);
    };
    if !durability.is_finite()
        || !(0.0..=1.0).contains(&durability)
        || giga.get("decay_anchor").and_then(|value| value.as_str()) != Some("candidate_created_at")
    {
        return (None, 1.0);
    }
    let Some(anchor) = giga
        .get("decay_anchor_at")
        .and_then(|value| value.as_str())
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.with_timezone(&Utc))
    else {
        return (None, 1.0);
    };
    let age_days = (now - anchor).num_seconds() as f64 / 86_400.0;
    if age_days <= 0.0 || durability >= 1.0 {
        return (Some(durability), 1.0);
    }
    let factor = (-std::f64::consts::LN_2 * age_days * (1.0 - durability).powi(2) / 7.0).exp();
    (
        Some(durability),
        if factor.is_finite() { factor } else { 1.0 },
    )
}

fn weighted_lane_score(chunk: &serde_json::Value, score_field: &str) -> f64 {
    chunk[score_field].as_f64().unwrap_or(0.0) * chunk["temporal_weight"].as_f64().unwrap_or(1.0)
}

fn compare_weighted_lane(
    left: &serde_json::Value,
    right: &serde_json::Value,
    score_field: &str,
) -> std::cmp::Ordering {
    weighted_lane_score(right, score_field)
        .partial_cmp(&weighted_lane_score(left, score_field))
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| {
            left["source_path"]
                .as_str()
                .cmp(&right["source_path"].as_str())
        })
        .then_with(|| {
            left["chunk_index"]
                .as_i64()
                .cmp(&right["chunk_index"].as_i64())
        })
}
async fn load_thread_neighbors(
    pool: &PgPool,
    memory_ids: &[i64],
) -> Result<BTreeMap<i64, Vec<serde_json::Value>>, AppError> {
    if memory_ids.is_empty() {
        return Ok(BTreeMap::new());
    }
    let rows = sqlx::query(
        r#"WITH neighbor_edges AS (
             SELECT current_event.memory_id AS candidate_memory_id,
                    t.thread_key,
                    'previous'::text AS direction,
                    0 AS direction_order,
                    neighbor_event.id AS neighbor_event_id,
                    neighbor.id,coalesce(neighbor.title,'') AS title,
                    neighbor.source_path,left(coalesce(neighbor.body,''),1200) AS body,
                    CASE
                      WHEN neighbor.archived_at IS NULL AND neighbor.superseded_by IS NULL
                      THEN 'active' ELSE 'historical'
                    END AS authority_state,
                    neighbor.superseded_by
             FROM thread_events current_event
             JOIN threads t ON t.id=current_event.thread_id
             JOIN thread_event_links link
               ON link.thread_id=current_event.thread_id
              AND link.next_event_id=current_event.id
             JOIN thread_events neighbor_event
               ON neighbor_event.thread_id=link.thread_id
              AND neighbor_event.id=link.previous_event_id
             JOIN memories neighbor ON neighbor.id=neighbor_event.memory_id
             WHERE current_event.memory_id = ANY($1::bigint[])
             UNION ALL
             SELECT current_event.memory_id AS candidate_memory_id,
                    t.thread_key,
                    'next'::text AS direction,
                    1 AS direction_order,
                    neighbor_event.id AS neighbor_event_id,
                    neighbor.id,coalesce(neighbor.title,'') AS title,
                    neighbor.source_path,left(coalesce(neighbor.body,''),1200) AS body,
                    CASE
                      WHEN neighbor.archived_at IS NULL AND neighbor.superseded_by IS NULL
                      THEN 'active' ELSE 'historical'
                    END AS authority_state,
                    neighbor.superseded_by
             FROM thread_events current_event
             JOIN threads t ON t.id=current_event.thread_id
             JOIN thread_event_links link
               ON link.thread_id=current_event.thread_id
              AND link.previous_event_id=current_event.id
             JOIN thread_events neighbor_event
               ON neighbor_event.thread_id=link.thread_id
              AND neighbor_event.id=link.next_event_id
             JOIN memories neighbor ON neighbor.id=neighbor_event.memory_id
             WHERE current_event.memory_id = ANY($1::bigint[])
           ), ranked AS (
             SELECT *,
                    row_number() OVER (
                      PARTITION BY candidate_memory_id
                      ORDER BY thread_key,direction_order,neighbor_event_id,id
                    ) AS ordinal
             FROM neighbor_edges
           )
           SELECT candidate_memory_id,thread_key,direction,id,title,source_path,body,
                  authority_state,superseded_by
           FROM ranked WHERE ordinal <= 6
           ORDER BY candidate_memory_id,ordinal"#,
    )
    .bind(memory_ids)
    .fetch_all(pool)
    .await?;
    let mut by_memory = BTreeMap::new();
    for row in rows {
        let candidate_memory_id: i64 = row.try_get("candidate_memory_id")?;
        let body: String = row.try_get("body")?;
        by_memory
            .entry(candidate_memory_id)
            .or_insert_with(Vec::new)
            .push(serde_json::json!({
                "thread": row.try_get::<String, _>("thread_key")?,
                "direction": row.try_get::<String, _>("direction")?,
                "id": row.try_get::<i64, _>("id")?,
                "title": row.try_get::<String, _>("title")?,
                "source_path": row.try_get::<String, _>("source_path")?,
                "excerpt": bounded_excerpt(&body),
                "authority_state": row.try_get::<String, _>("authority_state")?,
                "superseded_by": row.try_get::<Option<i64>, _>("superseded_by")?,
            }));
    }
    Ok(by_memory)
}

pub async fn recall(
    pool: &PgPool,
    cfg: &Config,
    params: RecallParams,
) -> Result<RecallResult, AppError> {
    params.validate()?;
    let query_dates = query_dates(&params.query);
    let query_terms = query_terms(&params.query);
    let content_patterns = query_terms
        .iter()
        .map(|term| format!("%{term}%"))
        .collect::<Vec<_>>();
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
    let decay_now = Utc::now();
    let semantic_fetch_limit = (!params.temporal_decay).then_some(i64::from(params.semantic_top_k));
    let content_fetch_limit = (!params.temporal_decay).then_some(i64::from(params.content_top_k));
    let mut semantic_chunks = Vec::new();
    if let Some(vector_text) = vector_text.clone() {
        let semantic_rows = sqlx::query(
            r#"SELECT memory_id,source_path,title,heading_path,body,char_start,char_end,chunk_index,meta,sim
               FROM (
                 SELECT m.source_path,
                        m.id AS memory_id,
                        coalesce(m.title,'') AS title,
                        coalesce(c.heading_path,'') AS heading_path,
                        c.body,c.char_start,c.char_end,c.chunk_index,m.meta AS meta,
                        (1-(c.body_embedding <=> $1::vector))::double precision AS sim
                 FROM memory_chunks c
                 JOIN memories m ON m.id=c.memory_id
                 WHERE m.room = ANY($2::text[])
                   AND m.archived_at IS NULL
                   AND m.superseded_by IS NULL
                   AND c.body_embedding IS NOT NULL
               ) ranked
               WHERE sim >= $3
               ORDER BY sim DESC,source_path,chunk_index
               LIMIT $4"#,
        )
        .bind(&vector_text)
        .bind(&rooms)
        .bind(params.semantic_min_similarity)
        .bind(semantic_fetch_limit)
        .fetch_all(pool)
        .await?;
        for row in semantic_rows {
            let sim: f64 = row.try_get("sim")?;
            if sim < params.semantic_min_similarity {
                continue;
            }
            let source_path: String = row.try_get("source_path")?;
            let title: Option<String> = row.try_get("title")?;
            let heading_path: Option<String> = row.try_get("heading_path")?;
            let body: String = row.try_get("body")?;
            let meta: serde_json::Value = row.try_get("meta")?;
            let (durability, temporal_weight) = if params.temporal_decay {
                giga_temporal_factor(&meta, decay_now)
            } else {
                (None, 1.0)
            };
            let (matched_terms, missing_terms, coverage) = candidate_terms(
                &query_terms,
                &[
                    &source_path,
                    title.as_deref().unwrap_or(""),
                    heading_path.as_deref().unwrap_or(""),
                    &body,
                ],
            );
            semantic_chunks.push(serde_json::json!({"memory_id":row.try_get::<i64,_>("memory_id")?,"source_path":source_path,"title":title,"heading_path":heading_path,"body":bounded_excerpt(&body),"char_start":row.try_get::<i32,_>("char_start")?,"char_end":row.try_get::<i32,_>("char_end")?,"chunk_index":row.try_get::<i32,_>("chunk_index")?,"sim":sim,"durability":durability,"temporal_weight":temporal_weight,"matched_terms":matched_terms,"missing_terms":missing_terms,"term_coverage":coverage,"evidence":"semantic cosine similarity"}));
        }
    }
    let content_rows = sqlx::query("SELECT m.id AS memory_id,m.source_path,coalesce(m.title,'') AS title,coalesce(c.heading_path,'') AS heading_path,c.body,c.char_start,c.char_end,c.chunk_index,m.meta AS meta,word_similarity($1,c.body)::double precision AS sim FROM memory_chunks c JOIN memories m ON m.id=c.memory_id WHERE m.room = ANY($2::text[]) AND m.archived_at IS NULL AND m.superseded_by IS NULL AND ($5::text[] = '{}'::text[] OR c.body ILIKE ANY($5::text[])) AND word_similarity($1,c.body) >= $3 ORDER BY sim DESC,m.source_path,c.chunk_index LIMIT $4")
        .bind(&params.query).bind(&rooms).bind(params.content_min_similarity).bind(content_fetch_limit).bind(&content_patterns).fetch_all(pool).await?;
    let mut content_chunks = Vec::new();
    for row in content_rows {
        let sim: f64 = row.try_get("sim")?;
        let source_path: String = row.try_get("source_path")?;
        let title: Option<String> = row.try_get("title")?;
        let heading_path: Option<String> = row.try_get("heading_path")?;
        let body: String = row.try_get("body")?;
        let meta: serde_json::Value = row.try_get("meta")?;
        let (durability, temporal_weight) = if params.temporal_decay {
            giga_temporal_factor(&meta, decay_now)
        } else {
            (None, 1.0)
        };
        let (matched_terms, missing_terms, coverage) = candidate_terms(
            &query_terms,
            &[
                &source_path,
                title.as_deref().unwrap_or(""),
                heading_path.as_deref().unwrap_or(""),
                &body,
            ],
        );
        content_chunks.push(serde_json::json!({"memory_id":row.try_get::<i64,_>("memory_id")?,"source_path":source_path,"title":title,"heading_path":heading_path,"body":bounded_excerpt(&body),"char_start":row.try_get::<i32,_>("char_start")?,"char_end":row.try_get::<i32,_>("char_end")?,"chunk_index":row.try_get::<i32,_>("chunk_index")?,"ws":sim,"durability":durability,"temporal_weight":temporal_weight,"matched_terms":matched_terms,"missing_terms":missing_terms,"term_coverage":coverage,"evidence":"lexical word_similarity"}));
    }
    if params.temporal_decay {
        semantic_chunks.sort_by(|left, right| compare_weighted_lane(left, right, "sim"));
        content_chunks.sort_by(|left, right| compare_weighted_lane(left, right, "ws"));
    }
    semantic_chunks.truncate(params.semantic_top_k as usize);
    content_chunks.truncate(params.content_top_k as usize);
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
    let thread_rows = sqlx::query(
        "SELECT DISTINCT ON (m.id) m.id AS memory_id,m.source_path,
                coalesce(m.title,'') AS title,t.thread_key,left(m.body,1200) AS body,
                GREATEST(similarity(t.thread_key,$2),
                         similarity(coalesce(ref.context,''),$2))::double precision AS rank
         FROM memory_thread_refs ref
         JOIN thread_events event ON event.id=ref.event_id
         JOIN threads t ON t.id=event.thread_id
         JOIN memories m ON m.id=event.memory_id
         WHERE m.room = ANY($1::text[]) AND m.archived_at IS NULL
           AND m.superseded_by IS NULL
           AND (t.thread_key ILIKE ANY($3::text[])
                OR ref.context ILIKE ANY($3::text[])
                OR m.source_path ILIKE ANY($3::text[]))
         ORDER BY m.id,rank DESC,t.thread_key
         LIMIT 8",
    )
    .bind(&rooms)
    .bind(&params.query)
    .bind(&content_patterns)
    .fetch_all(pool)
    .await?;
    let mut fused: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for (rank, c) in semantic_chunks.iter().enumerate() {
        let key = format!(
            "{}#{}",
            c["memory_id"].as_i64().unwrap_or(0),
            c["chunk_index"].as_i64().unwrap_or(0)
        );
        let score = (c["sim"].as_f64().unwrap_or(0.0) * 0.6 + 1.0 / (rank as f64 + 1.0) * 0.4)
            * c["temporal_weight"].as_f64().unwrap_or(1.0);
        let mut reasons = vec!["semantic cosine similarity"];
        if c["temporal_weight"].as_f64().unwrap_or(1.0) < 1.0 {
            reasons.push("durability-shaped temporal decay");
        }
        fused.insert(key, serde_json::json!({"memory_id":c["memory_id"],"source_path":c["source_path"],"title":c["title"],"heading_path":c["heading_path"],"excerpt":c["body"],"sources":[c["source_path"]],"term_coverage":c["term_coverage"],"matched_terms":c["matched_terms"],"missing_terms":c["missing_terms"],"score":score,"semantic_score":c["sim"],"durability":c["durability"],"temporal_weight":c["temporal_weight"],"reasons":reasons,"source":"semantic","chunk_index":c["chunk_index"]}));
    }
    for (rank, c) in content_chunks.iter().enumerate() {
        let key = format!(
            "{}#{}",
            c["memory_id"].as_i64().unwrap_or(0),
            c["chunk_index"].as_i64().unwrap_or(0)
        );
        let score = (c["ws"].as_f64().unwrap_or(0.0) * 0.6 + 1.0 / (rank as f64 + 1.0) * 0.4)
            * c["temporal_weight"].as_f64().unwrap_or(1.0);
        let decayed = c["temporal_weight"].as_f64().unwrap_or(1.0) < 1.0;
        let decay_reason = if decayed {
            vec!["durability-shaped temporal decay"]
        } else {
            Vec::new()
        };
        if let Some(existing) = fused.get_mut(&key) {
            existing["score"] =
                serde_json::json!(existing["score"].as_f64().unwrap_or(0.0) + score);
            existing["content_score"] = c["ws"].clone();
            existing["source"] = serde_json::json!("semantic+content");
            existing["durability"] = c["durability"].clone();
            existing["temporal_weight"] = c["temporal_weight"].clone();
            let mut reasons = vec!["semantic cosine similarity", "lexical word_similarity"];
            if decayed {
                reasons.push("durability-shaped temporal decay");
            }
            existing["reasons"] = serde_json::json!(reasons);
        } else {
            fused.insert(key, serde_json::json!({"memory_id":c["memory_id"],"source_path":c["source_path"],"title":c["title"],"heading_path":c["heading_path"],"excerpt":c["body"],"sources":[c["source_path"]],"term_coverage":c["term_coverage"],"matched_terms":c["matched_terms"],"missing_terms":c["missing_terms"],"score":score,"content_score":c["ws"],"durability":c["durability"],"temporal_weight":c["temporal_weight"],"reasons":if decay_reason.is_empty() { serde_json::json!(["lexical word_similarity"]) } else { serde_json::json!(["lexical word_similarity","durability-shaped temporal decay"]) },"source":"content","chunk_index":c["chunk_index"]}));
        }
    }
    for row in &thread_rows {
        let memory_id: i64 = row.try_get("memory_id")?;
        let source_path: String = row.try_get("source_path")?;
        let thread_key: String = row.try_get("thread_key")?;
        let title: String = row.try_get("title")?;
        let body: String = row.try_get("body")?;
        let rank: f64 = row.try_get("rank")?;
        // A named thread is a deliberate authoring act, so it carries weight even when
        // trigram similarity is low; the floor keeps a real key from scoring as noise.
        let score = 0.35 + rank * 0.55;
        if let Some(existing) = fused
            .values_mut()
            .find(|entry| entry["memory_id"].as_i64() == Some(memory_id))
        {
            existing["score"] =
                serde_json::json!(existing["score"].as_f64().unwrap_or(0.0) + score);
            existing["thread_key"] = serde_json::json!(thread_key);
            if let Some(reasons) = existing["reasons"].as_array_mut() {
                reasons.push(serde_json::json!("lexical thread key"));
            }
            continue;
        }
        let (matched_terms, missing_terms, coverage) =
            candidate_terms(&query_terms, &[&source_path, &title, &thread_key, &body]);
        fused.insert(
            format!("{memory_id}#thread"),
            serde_json::json!({"memory_id":memory_id,"source_path":source_path.clone(),"title":title,"heading_path":"","excerpt":bounded_excerpt(&body),"sources":[source_path],"term_coverage":coverage,"matched_terms":matched_terms,"missing_terms":missing_terms,"score":score,"thread_key":thread_key,"reasons":["lexical thread key"],"source":"thread","chunk_index":0}),
        );
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
            .then_with(|| a["memory_id"].as_i64().cmp(&b["memory_id"].as_i64()))
    });
    retrieval_candidates.truncate(params.semantic_top_k.max(params.content_top_k) as usize);
    let memory_ids = retrieval_candidates
        .iter()
        .filter_map(|candidate| candidate["memory_id"].as_i64())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut neighbors = load_thread_neighbors(pool, &memory_ids).await?;
    for candidate in &mut retrieval_candidates {
        let memory_id = candidate["memory_id"].as_i64().unwrap_or_default();
        candidate["thread_neighbors"] =
            serde_json::Value::Array(neighbors.remove(&memory_id).unwrap_or_default());
    }
    // Canon lookup matches three ways, ranked. Tokens alone can never match a
    // multi-word name or a hyphenated alias, which is how 42 of 109 rows went
    // dark; widening to ILIKE/tsvector then lets fuzzy hits evict the row the
    // caller literally named. So: whole-phrase exact, then token exact, then
    // similarity — and only the last tier competes for leftover LIMIT slots.
    let query_phrase = params.query.trim().to_lowercase();
    let canon_rows = sqlx::query("SELECT name,kind,summary,aliases,weighty,pointer_files, (CASE WHEN lower(name) = $5 OR EXISTS (SELECT 1 FROM unnest(aliases) a1 WHERE lower(a1) = $5) THEN 0 WHEN lower(name) = ANY($2::text[]) OR EXISTS (SELECT 1 FROM unnest(aliases) a2 WHERE lower(a2) = ANY($2::text[])) THEN 1 ELSE 2 END) AS exactness FROM named_entities, websearch_to_tsquery('portuguese', $3) AS tsq WHERE room = ANY($1::text[]) AND (lower(name) = $5 OR EXISTS (SELECT 1 FROM unnest(aliases) alias WHERE lower(alias) = $5) OR lower(name) = ANY($2::text[]) OR EXISTS (SELECT 1 FROM unnest(aliases) alias WHERE lower(alias) = ANY($2::text[])) OR name ILIKE ANY($4::text[]) OR EXISTS (SELECT 1 FROM unnest(aliases) alias WHERE alias ILIKE ANY($4::text[])) OR summary_tsv @@ tsq) ORDER BY exactness, weighty DESC, name LIMIT 12")
        .bind(&rooms).bind(&query_terms).bind(&params.query).bind(&content_patterns).bind(&query_phrase).fetch_all(pool).await?;
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
    let thread_keys: Vec<String> = sqlx::query_scalar("SELECT DISTINCT t.thread_key FROM threads t JOIN thread_events event ON event.thread_id=t.id JOIN memories m ON m.id=event.memory_id WHERE m.room = ANY($1::text[]) AND m.archived_at IS NULL AND m.superseded_by IS NULL ORDER BY t.thread_key LIMIT 20").bind(&rooms).fetch_all(pool).await?;
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

#[cfg(test)]
mod tests {
    use super::giga_temporal_factor;
    use chrono::{Duration, TimeZone, Utc};
    use serde_json::{Value, json};

    fn giga_meta(durability: Value, anchor: &str) -> Value {
        json!({
            "origin": "giga-promotion",
            "giga": {
                "durability": durability,
                "decay_anchor": "candidate_created_at",
                "decay_anchor_at": anchor,
            },
        })
    }

    #[test]
    fn temporal_factor_treats_unsafe_metadata_as_legacy_weight() {
        let now = Utc.with_ymd_and_hms(2030, 2, 1, 0, 0, 0).single().unwrap();
        let anchor = (now - Duration::days(7)).to_rfc3339();
        let cases = [
            json!({}),
            giga_meta(json!("not-a-number"), &anchor),
            json!({
                "origin": "manual",
                "giga": {
                    "durability": 0.0,
                    "decay_anchor": "candidate_created_at",
                    "decay_anchor_at": anchor,
                },
            }),
        ];

        for meta in cases {
            assert_eq!(giga_temporal_factor(&meta, now), (None, 1.0));
        }
    }

    #[test]
    fn temporal_factor_follows_the_durability_shaped_half_life() {
        let now = Utc.with_ymd_and_hms(2030, 2, 1, 0, 0, 0).single().unwrap();
        let seven_days_ago = (now - Duration::days(7)).to_rfc3339();
        let twenty_eight_days_ago = (now - Duration::days(28)).to_rfc3339();

        let (durability_zero, factor_zero) =
            giga_temporal_factor(&giga_meta(json!(0.0), &seven_days_ago), now);
        assert_eq!(durability_zero, Some(0.0));
        assert!((factor_zero - 0.5).abs() < 1e-12);

        let (durability_half, factor_half) =
            giga_temporal_factor(&giga_meta(json!(0.5), &twenty_eight_days_ago), now);
        assert_eq!(durability_half, Some(0.5));
        assert!((factor_half - 0.5).abs() < 1e-12);
    }

    #[test]
    fn temporal_factor_never_decays_permanent_or_future_memories() {
        let now = Utc.with_ymd_and_hms(2030, 2, 1, 0, 0, 0).single().unwrap();
        let old_anchor = (now - Duration::days(365)).to_rfc3339();
        let future_anchor = (now + Duration::days(1)).to_rfc3339();

        assert_eq!(
            giga_temporal_factor(&giga_meta(json!(1.0), &old_anchor), now),
            (Some(1.0), 1.0)
        );
        assert_eq!(
            giga_temporal_factor(&giga_meta(json!(0.0), &future_anchor), now),
            (Some(0.0), 1.0)
        );
    }
}
