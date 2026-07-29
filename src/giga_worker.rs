use crate::{
    AppError,
    config::HTTP_CLIENT,
    giga::{database_now, event_from_store, giga_candidate_store_and_finish, giga_event_finish},
};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use house_core::{
    GIGA_MAX_EVENT_ATTEMPTS, GigaAuthority, GigaCandidate, GigaCandidateKind,
    GigaClassifierIdentity, GigaEvent, GigaEventFinishOutcome, GigaEventFinishRequest,
    GigaEventType, GigaProcessRequest, GigaReviewState, GigaScope, GigaScores, GigaSourceRef,
    GigaSourceType, GigaVisibility,
};
use house_protocol::{GigaClassifierHealthResult, GigaProcessResult, RequiredNullable};
use reqwest::{RequestBuilder, Url};
use serde::{ser::SerializeMap, Deserialize, Serialize};
use serde_json::{value::RawValue, Value, json};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Row};
use std::{env, sync::LazyLock, time::Duration};
use tokio::time::sleep;
use uuid::Uuid;

pub const GIGA_PROMPT_VERSION: &str = "agents-a1-two-pass-v1";
pub const GIGA_MODEL_TAG: &str = "hf.co/InternScience/Agents-A1-4B-Q4_K_M-GGUF:latest";
pub const GIGA_MODEL_MANIFEST_DIGEST: &str =
    "96ca1ea02b302bf5cd1118d637f12a5af7c2a5aa465837532448bd6e54db4ceb";
const GIGA_DEFAULT_OLLAMA_ENDPOINT: &str = "http://127.0.0.1:11435";
const GIGA_LEASE_SECONDS: i64 = 300;
const GIGA_RETRY_DELAY_SECONDS: u32 = 30;
const GIGA_MODEL_TIMEOUT: Duration = Duration::from_secs(60);
const GIGA_MAX_OLLAMA_RESPONSE_BYTES: usize = 64 * 1024;
const GIGA_MAX_MODEL_RATIONALE_BYTES: usize = 4 * 1024;
const GIGA_MAX_STORED_RATIONALE_BYTES: usize = 1_200;
const GIGA_RATIONALE_TRUNCATION_MARKER: &str =
    "\n[truncated: classifier rationale is diagnostic, not evidence]";
const GIGA_MAX_MESSAGE_BYTES: usize = 16 * 1024;
// Measured on this box 2026-07-25: KV cache costs ~41 MB per 1k tokens for this
// model (num_ctx 4096 -> 3.19 GB VRAM, 16384 -> 3.69 GB). 32768 lands near 4.4 GB
// and sits comfortably beside the 2.1 GB embedder on a 16 GB card.
//
// 4096 was unusable and produced GigaClassifierOutputError on every large event:
// the adapter may send up to GIGA_MAX_WINDOW_BYTES (24_000 bytes, roughly 6-7k
// tokens) plus the system prompt plus 768 reserved output tokens, against a 4096
// window. The two budgets contradicted each other. Proof that day: an 8-turn kodo
// event failed three attempts while a 1-turn kintsu event succeeded first try.
//
// The headroom above the window is deliberate. Classification is meant to receive
// retrieved neighbour memories so that `novelty` is measured against what the
// House already holds rather than asserted by a model with no past.
const GIGA_NUM_CTX: u32 = 32_768;
// Ollama defaults to a 5 minute keep_alive, which evicted both models during idle
// gaps and charged a cold reload on the next turn. Lower this if the card is
// needed elsewhere; residency is a comfort setting, not a correctness one.
const GIGA_KEEP_ALIVE: &str = "30m";
const GIGA_MAX_SELECTED_SOURCES: usize = 4;
const GIGA_MAX_RETRIEVAL_TERMS: usize = 8;
const GIGA_MAX_DIAGNOSTIC_CLASS_BYTES: usize = 128;

pub const GIGA_GATE_PROMPT: &str = concat!(
    "You are Hippocampus's first-pass durability gate. Classify the single strongest ",
    "source-grounded durable change in the supplied exact conversation window. Return ",
    "kind=none only when no explicit durable change exists. Return memory for a new ",
    "decision, commitment, consent boundary, preference, meaningful first, relationship ",
    "meaning, or current-state change. Affection, Eros, play, and intimacy are load-bearing ",
    "when they establish one of those changes; do not clinicalize them. Return coding_lesson ",
    "for a transferable engineering rule with an observed result and no explicit project key. ",
    "Return project_lesson for a stable rule tied to one supplied project key; when a project ",
    "key exists and the source explicitly states a stable project rule, project_lesson wins ",
    "unless the source separately states both an old current state and a new replacement state. ",
    "Return correction for a direct correction of an earlier interpretation. Return supersession ",
    "only when a newer current-state claim explicitly replaces an older current-state claim. ",
    "Ordinary continuity with no new meaning, greetings, repeated status, tool noise, unsupported ",
    "opinions, and abandoned hypotheticals are none. Use only supplied source_id values. For ",
    "kind=none source_ids must be empty; otherwise include the minimal exact supporting IDs. ",
    "Do not emit reasoning outside the JSON.\n\n",
    "Few-shot examples:\n",
    "1. New consent boundary -> memory.\n",
    "2. Explicit project rule with project key -> project_lesson.\n",
    "3. Direct rejection/correction of earlier interpretation -> correction.\n",
    "4. Explicit old canonical state + new replacement + supersede instruction -> supersession.\n",
    "5. Tool exit status + acknowledgment -> none.\n",
    "6. Test failure caused by inherited variable + rule to sanitize + passing proof -> coding_lesson."
);

pub const GIGA_EXTRACTION_PROMPT: &str = concat!(
    "You are Hippocampus's second-pass candidate extractor. The durability gate already chose ",
    "one candidate kind and exact supporting turns. Produce one compact source-grounded candidate ",
    "proposal. Generated title, rationale, gist, scores, and retrieval terms are navigation aids, ",
    "never authority or evidence. Preserve the human meaning and voice without clinicalizing intimacy. Do not ",
    "invent facts, project keys, targets, entities, threads, or source IDs. Lessons must state a ",
    "reusable rule and observed proof. Corrections must describe what reading was corrected. ",
    "Supersessions must describe old and new state but must not claim that the old record was ",
    "already changed. Use the minimal exact source IDs. Every numeric score must be finite and ",
    "between 0.0 and 1.0 inclusive. Put classifier reasoning in rationale; do not emit reasoning outside JSON."
);

static GIGA_WORKER_ID: LazyLock<String> =
    LazyLock::new(|| format!("rust-hippocampus:{}", Uuid::new_v4()));

#[derive(Clone, Copy, Debug)]
struct WorkerFailure {
    class: &'static str,
    retryable: bool,
}

impl WorkerFailure {
    const fn new(class: &'static str, retryable: bool) -> Self {
        Self { class, retryable }
    }
}

#[derive(Clone)]
struct OllamaConfig {
    endpoint: Url,
}

#[derive(Clone)]
struct ResolvedSource {
    source: GigaSourceRef,
    text: String,
}

#[derive(Serialize)]
struct ModelSource<'a> {
    source_id: &'a str,
    role: &'a str,
    timestamp: &'a str,
    text: &'a str,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum GateKind {
    None,
    Memory,
    CodingLesson,
    ProjectLesson,
    Correction,
    Supersession,
}

impl GateKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Memory => "memory",
            Self::CodingLesson => "coding_lesson",
            Self::ProjectLesson => "project_lesson",
            Self::Correction => "correction",
            Self::Supersession => "supersession",
        }
    }

    fn candidate_kind(self) -> Result<GigaCandidateKind, WorkerFailure> {
        match self {
            Self::None => Err(WorkerFailure::new("GigaClassifierOutputError", true)),
            Self::Memory => Ok(GigaCandidateKind::Memory),
            Self::CodingLesson => Ok(GigaCandidateKind::CodingLesson),
            Self::ProjectLesson => Ok(GigaCandidateKind::ProjectLesson),
            Self::Correction => Ok(GigaCandidateKind::Correction),
            Self::Supersession => Ok(GigaCandidateKind::Supersession),
        }
    }

    const fn requires_proof(self) -> bool {
        !matches!(self, Self::None | Self::Memory)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GateOutput {
    kind: GateKind,
    source_ids: Vec<String>,
    reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtractionOutput {
    source_ids: Vec<String>,
    proof_source_ids: Vec<String>,
    proposed_title: String,
    rationale: String,
    gist: String,
    priority: f64,
    novelty: f64,
    durability: f64,
    confidence: f64,
    retrieval_terms: Vec<String>,
}

enum ClaimState {
    Claimed {
        event: GigaEvent,
        attempt_count: u32,
    },
    WaitUntil(DateTime<Utc>),
    Terminal {
        event: GigaEvent,
        result: GigaProcessResult,
    },
}

fn domain_failure() -> WorkerFailure {
    WorkerFailure::new("GigaClassifierOutputError", true)
}

fn classifier_enabled() -> bool {
    env::var("SOLARISAEL_GIGA_ENABLED").ok().as_deref() == Some("1")
        && env::var("SOLARISAEL_HIPPOCAMPUS_ENABLED").ok().as_deref() == Some("1")
        && env::var("SOLARISAEL_REPLAY_MODE").ok().as_deref() != Some("1")
}

pub(crate) fn giga_classifier_enabled() -> bool {
    classifier_enabled()
}

pub(crate) fn giga_classifier_health(
    last_error_class: Option<String>,
    last_error_at: Option<String>,
    consecutive_failures: u64,
) -> GigaClassifierHealthResult {
    let raw = env::var("SOLARISAEL_HIPPOCAMPUS_OLLAMA_ENDPOINT")
        .unwrap_or_else(|_| GIGA_DEFAULT_OLLAMA_ENDPOINT.into());
    let endpoint_scope = Url::parse(&raw)
        .ok()
        .map(|endpoint| {
            if is_loopback(&endpoint) {
                "loopback"
            } else {
                "remote"
            }
        })
        .unwrap_or("invalid");
    GigaClassifierHealthResult {
        provider_type: "ollama".into(),
        model: GIGA_MODEL_TAG.into(),
        model_digest: GIGA_MODEL_MANIFEST_DIGEST.into(),
        prompt_version: GIGA_PROMPT_VERSION.into(),
        endpoint_scope: endpoint_scope.into(),
        // This is historical sticky context; consecutive_failures is the live signal.
        last_error_class: RequiredNullable(safe_error_class(last_error_class)),
        last_error_at: RequiredNullable(last_error_at),
        consecutive_failures,
    }
}

fn is_loopback(endpoint: &Url) -> bool {
    let Some(host) = endpoint.host_str() else {
        return false;
    };
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn ollama_config() -> Result<OllamaConfig, WorkerFailure> {
    if !classifier_enabled() {
        return Err(WorkerFailure::new("GigaClassifierDisabled", false));
    }
    let raw = env::var("SOLARISAEL_HIPPOCAMPUS_OLLAMA_ENDPOINT")
        .unwrap_or_else(|_| GIGA_DEFAULT_OLLAMA_ENDPOINT.into());
    let mut endpoint =
        Url::parse(&raw).map_err(|_| WorkerFailure::new("GigaOllamaConfigurationError", false))?;
    if !matches!(endpoint.scheme(), "http" | "https")
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
        || endpoint.host_str().is_none()
        || (!is_loopback(&endpoint)
            && env::var("SOLARISAEL_HIPPOCAMPUS_REMOTE_CONSENT")
                .ok()
                .as_deref()
                != Some("1"))
    {
        return Err(WorkerFailure::new("GigaOllamaConfigurationError", false));
    }
    let normalized_path = endpoint.path().trim_end_matches('/').to_owned();
    endpoint.set_path(if normalized_path.is_empty() {
        "/"
    } else {
        &normalized_path
    });
    Ok(OllamaConfig { endpoint })
}

fn ollama_url(config: &OllamaConfig, suffix: &str) -> Result<Url, WorkerFailure> {
    let base = config.endpoint.as_str().trim_end_matches('/');
    Url::parse(&format!("{base}/{}", suffix.trim_start_matches('/')))
        .map_err(|_| WorkerFailure::new("GigaOllamaConfigurationError", false))
}

async fn bounded_response(request: RequestBuilder) -> Result<String, WorkerFailure> {
    let mut response = request
        .timeout(GIGA_MODEL_TIMEOUT)
        .send()
        .await
        .map_err(|_| WorkerFailure::new("GigaOllamaTransportError", true))?;
    if !response.status().is_success() {
        return Err(WorkerFailure::new("GigaOllamaTransportError", true));
    }
    if response
        .content_length()
        .is_some_and(|length| length > GIGA_MAX_OLLAMA_RESPONSE_BYTES as u64)
    {
        return Err(WorkerFailure::new("GigaOllamaResponseError", true));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| WorkerFailure::new("GigaOllamaTransportError", true))?
    {
        if body.len().saturating_add(chunk.len()) > GIGA_MAX_OLLAMA_RESPONSE_BYTES {
            return Err(WorkerFailure::new("GigaOllamaResponseError", true));
        }
        body.extend_from_slice(&chunk);
    }
    if body.is_empty() {
        return Err(WorkerFailure::new("GigaOllamaResponseError", true));
    }
    String::from_utf8(body).map_err(|_| WorkerFailure::new("GigaOllamaResponseError", true))
}

async fn verify_ollama_model(config: &OllamaConfig) -> Result<(), WorkerFailure> {
    let body = bounded_response(HTTP_CLIENT.get(ollama_url(config, "/api/tags")?)).await?;
    let value: Value = serde_json::from_str(&body)
        .map_err(|_| WorkerFailure::new("GigaOllamaResponseError", true))?;
    let models = value
        .as_object()
        .and_then(|object| object.get("models"))
        .and_then(Value::as_array)
        .ok_or_else(|| WorkerFailure::new("GigaOllamaResponseError", true))?;
    let matching = models
        .iter()
        .filter(|model| {
            model.as_object().is_some_and(|object| {
                object.get("name").and_then(Value::as_str) == Some(GIGA_MODEL_TAG)
                    || object.get("model").and_then(Value::as_str) == Some(GIGA_MODEL_TAG)
            })
        })
        .collect::<Vec<_>>();
    if matching.len() != 1
        || matching[0]
            .as_object()
            .and_then(|object| object.get("digest"))
            .and_then(Value::as_str)
            != Some(GIGA_MODEL_MANIFEST_DIGEST)
    {
        return Err(WorkerFailure::new("GigaOllamaModelIdentityError", false));
    }
    Ok(())
}

fn gate_schema(source_ids: &[String]) -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["kind", "source_ids", "reason"],
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["none", "memory", "coding_lesson", "project_lesson", "correction", "supersession"]
            },
            "source_ids": {
                "type": "array",
                "maxItems": GIGA_MAX_SELECTED_SOURCES,
                "uniqueItems": true,
                "items": { "type": "string", "enum": source_ids }
            },
            "reason": { "type": "string", "minLength": 1 }
        }
    })
}

// Ollama 0.32.4 rejects maxLength during JSON-Schema-to-GBNF conversion. Keep
// wire schemas shape-focused; bounded_trimmed and truncate_with_marker enforce limits locally.
#[derive(Serialize)]
struct OrderedExtractionSchema {
    #[serde(rename = "type")]
    schema_type: &'static str,
    #[serde(rename = "additionalProperties")]
    additional_properties: bool,
    required: [&'static str; 10],
    properties: OrderedExtractionProperties,
}

struct OrderedExtractionProperties {
    source_ids: Value,
    proof_source_ids: Value,
    proposed_title: Value,
    rationale: Value,
    gist: Value,
    priority: Value,
    novelty: Value,
    durability: Value,
    confidence: Value,
    retrieval_terms: Value,
}

impl Serialize for OrderedExtractionProperties {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Keep this ordering local: serde_json::preserve_order would alter every
        // Value serialization, including the hash inputs used for GIGA identity.
        let mut properties = serializer.serialize_map(Some(10))?;
        properties.serialize_entry("source_ids", &self.source_ids)?;
        properties.serialize_entry("proof_source_ids", &self.proof_source_ids)?;
        properties.serialize_entry("proposed_title", &self.proposed_title)?;
        properties.serialize_entry("rationale", &self.rationale)?;
        properties.serialize_entry("gist", &self.gist)?;
        properties.serialize_entry("priority", &self.priority)?;
        properties.serialize_entry("novelty", &self.novelty)?;
        properties.serialize_entry("durability", &self.durability)?;
        properties.serialize_entry("confidence", &self.confidence)?;
        properties.serialize_entry("retrieval_terms", &self.retrieval_terms)?;
        properties.end()
    }
}

fn extraction_schema(
    source_ids: &[String],
    proof_required: bool,
) -> Result<String, WorkerFailure> {
    serde_json::to_string(&OrderedExtractionSchema {
        schema_type: "object",
        additional_properties: false,
        required: [
            "source_ids",
            "proof_source_ids",
            "proposed_title",
            "rationale",
            "gist",
            "priority",
            "novelty",
            "durability",
            "confidence",
            "retrieval_terms",
        ],
        properties: OrderedExtractionProperties {
            source_ids: json!({
                "type": "array",
                "minItems": 1,
                "maxItems": GIGA_MAX_SELECTED_SOURCES,
                "uniqueItems": true,
                "items": { "type": "string", "enum": source_ids }
            }),
            proof_source_ids: json!({
                "type": "array",
                "minItems": if proof_required { 1 } else { 0 },
                "maxItems": GIGA_MAX_SELECTED_SOURCES,
                "uniqueItems": true,
                "items": { "type": "string", "enum": source_ids }
            }),
            proposed_title: json!({ "type": "string", "minLength": 1 }),
            rationale: json!({ "type": "string", "minLength": 1 }),
            gist: json!({ "type": "string", "minLength": 1 }),
            priority: json!({ "type": "number", "minimum": 0, "maximum": 1 }),
            novelty: json!({ "type": "number", "minimum": 0, "maximum": 1 }),
            durability: json!({ "type": "number", "minimum": 0, "maximum": 1 }),
            confidence: json!({ "type": "number", "minimum": 0, "maximum": 1 }),
            retrieval_terms: json!({
                "type": "array",
                "maxItems": GIGA_MAX_RETRIEVAL_TERMS,
                "uniqueItems": true,
                "items": { "type": "string", "minLength": 1 }
            }),
        },
    })
    .map_err(|_| WorkerFailure::new("GigaClassifierRequestError", false))
}

fn schema_json(value: Value) -> Result<String, WorkerFailure> {
    serde_json::to_string(&value)
        .map_err(|_| WorkerFailure::new("GigaClassifierRequestError", false))
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: u8,
    seed: u32,
    num_ctx: u32,
    num_predict: u32,
}

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'static str,
    messages: [Value; 2],
    stream: bool,
    think: bool,
    format: &'a RawValue,
    keep_alive: &'static str,
    options: OllamaOptions,
}
async fn request_ollama_structured<T: for<'de> Deserialize<'de>>(
    config: &OllamaConfig,
    system_prompt: &str,
    user_payload: Value,
    schema: String,
    num_predict: u32,
) -> Result<T, WorkerFailure> {
    let user_content = serde_json::to_string(&user_payload)
        .map_err(|_| WorkerFailure::new("GigaClassifierRequestError", false))?;
    let schema = RawValue::from_string(schema)
        .map_err(|_| WorkerFailure::new("GigaClassifierRequestError", false))?;
    let request = OllamaRequest {
        model: GIGA_MODEL_TAG,
        messages: [
            json!({ "role": "system", "content": system_prompt }),
            json!({ "role": "user", "content": user_content }),
        ],
        stream: false,
        think: false,
        format: schema.as_ref(),
        keep_alive: GIGA_KEEP_ALIVE,
        options: OllamaOptions {
            temperature: 0,
            seed: 42,
            num_ctx: GIGA_NUM_CTX,
            num_predict,
        },
    };
    let body = bounded_response(
        HTTP_CLIENT
            .post(ollama_url(config, "/api/chat")?)
            .header("content-type", "application/json")
            .json(&request),
    )
    .await?;
    let envelope: Value = serde_json::from_str(&body)
        .map_err(|_| WorkerFailure::new("GigaOllamaResponseError", true))?;
    let object = envelope
        .as_object()
        .ok_or_else(|| WorkerFailure::new("GigaOllamaResponseError", true))?;
    let message = object
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| WorkerFailure::new("GigaOllamaResponseError", true))?;
    let content = message
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| WorkerFailure::new("GigaOllamaResponseError", true))?;
    if object.get("done").and_then(Value::as_bool) != Some(true)
        || object.get("done_reason").and_then(Value::as_str) != Some("stop")
        || object.get("model").and_then(Value::as_str) != Some(GIGA_MODEL_TAG)
        || message.get("role").and_then(Value::as_str) != Some("assistant")
        || content.is_empty()
        || content.len() > GIGA_MAX_MESSAGE_BYTES
        || message
            .get("thinking")
            .and_then(Value::as_str)
            .is_some_and(|thinking| !thinking.is_empty())
    {
        return Err(WorkerFailure::new("GigaOllamaResponseError", true));
    }
    serde_json::from_str(content).map_err(|_| WorkerFailure::new("GigaClassifierOutputError", true))
}

fn exact_unique_subset(values: &[String], allowed: &[String], allow_empty: bool) -> bool {
    (allow_empty || !values.is_empty())
        && values.len() <= GIGA_MAX_SELECTED_SOURCES
        && values
            .iter()
            .enumerate()
            .all(|(index, value)| allowed.contains(value) && !values[..index].contains(value))
}

fn bounded_trimmed(value: &str, maximum: usize) -> bool {
    !value.is_empty() && value.len() <= maximum && value.trim() == value
}

fn truncate_with_marker(value: &str, maximum: usize, marker: &str) -> String {
    if value.len() <= maximum {
        return value.to_owned();
    }
    if marker.len() >= maximum {
        return marker.chars().take(maximum).collect();
    }
    let mut content_end = maximum - marker.len();
    while !value.is_char_boundary(content_end) {
        content_end -= 1;
    }
    let mut truncated = String::with_capacity(maximum);
    truncated.push_str(&value[..content_end]);
    truncated.push_str(marker);
    truncated
}

fn validate_gate(gate: &GateOutput, event: &GigaEvent) -> Result<(), WorkerFailure> {
    let source_ids = event
        .source_refs()
        .iter()
        .map(|source| source.source_id().to_owned())
        .collect::<Vec<_>>();
    if !bounded_trimmed(&gate.reason, 160)
        || !exact_unique_subset(&gate.source_ids, &source_ids, gate.kind == GateKind::None)
        || (gate.kind == GateKind::None && !gate.source_ids.is_empty())
        || (gate.kind != GateKind::None && gate.source_ids.is_empty())
        || (gate.kind == GateKind::ProjectLesson && event.project_keys().len() != 1)
        || (gate.kind == GateKind::CodingLesson && !event.project_keys().is_empty())
    {
        return Err(domain_failure());
    }
    Ok(())
}
fn finite_score(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn validate_extraction(
    extraction: &ExtractionOutput,
    gate: &GateOutput,
) -> Result<(), WorkerFailure> {
    if !exact_unique_subset(&extraction.source_ids, &gate.source_ids, false)
        || !exact_unique_subset(
            &extraction.proof_source_ids,
            &extraction.source_ids,
            !gate.kind.requires_proof(),
        )
        || (gate.kind.requires_proof() && extraction.proof_source_ids.is_empty())
        || !bounded_trimmed(&extraction.proposed_title, 160)
        || !bounded_trimmed(&extraction.gist, 1_200)
        || !bounded_trimmed(&extraction.rationale, GIGA_MAX_MODEL_RATIONALE_BYTES)
        || !finite_score(extraction.priority)
        || !finite_score(extraction.novelty)
        || !finite_score(extraction.durability)
        || !finite_score(extraction.confidence)
        || extraction.retrieval_terms.len() > GIGA_MAX_RETRIEVAL_TERMS
        || extraction
            .retrieval_terms
            .iter()
            .enumerate()
            .any(|(index, term)| {
                !bounded_trimmed(term, 80) || extraction.retrieval_terms[..index].contains(term)
            })
    {
        return Err(domain_failure());
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_json(value: &Value) -> Result<String, WorkerFailure> {
    serde_json::to_vec(value)
        .map(|bytes| sha256_bytes(&bytes))
        .map_err(|_| WorkerFailure::new("GigaClassifierRequestError", false))
}

fn configuration_digest(config: &OllamaConfig) -> Result<String, WorkerFailure> {
    sha256_json(&json!({
        "provider": "ollama",
        "endpoint_base_url": config.endpoint.as_str(),
        "model": GIGA_MODEL_TAG,
        "model_digest": GIGA_MODEL_MANIFEST_DIGEST,
        "prompt_version": GIGA_PROMPT_VERSION,
        "gate_prompt_digest": sha256_bytes(GIGA_GATE_PROMPT.as_bytes()),
        "extraction_prompt_digest": sha256_bytes(GIGA_EXTRACTION_PROMPT.as_bytes()),
        "temperature": 0,
        "seed": 42,
        "num_ctx": GIGA_NUM_CTX,
        "gate_num_predict": 256,
        "extraction_num_predict": 768
    }))
}

fn candidate_id(
    event: &GigaEvent,
    kind: GateKind,
    source_ids: &[String],
) -> Result<String, WorkerFailure> {
    let mut sorted = source_ids.to_vec();
    sorted.sort();
    sha256_json(&json!([
        event.event_id(),
        GIGA_PROMPT_VERSION,
        GIGA_MODEL_MANIFEST_DIGEST,
        kind.as_str(),
        sorted
    ]))
}

fn validate_source_bundle(
    event: &GigaEvent,
    request: &GigaProcessRequest,
) -> Result<Vec<ResolvedSource>, WorkerFailure> {
    if event.event_type() != GigaEventType::ConversationWindow
        || event.event_id() != request.event_id()
        || event.project_keys().len() > 1
        || event.source_refs().len() != request.sources().len()
    {
        return Err(WorkerFailure::new("GigaSourceVerificationError", false));
    }
    let expected_project = event.project_keys().first().map(String::as_str);
    for (index, source) in event.source_refs().iter().enumerate() {
        if event.source_refs()[..index]
            .iter()
            .any(|known| known.source_id() == source.source_id())
            || source.source_type() != GigaSourceType::Turn
            || !matches!(source.role(), "user" | "assistant")
            || source.scope().visibility() != GigaVisibility::Private
            || source.scope().room() != Some(event.room())
            || source.scope().project() != expected_project
            || !source.scope().publication_review_required()
            || source.range().is_some()
        {
            return Err(WorkerFailure::new("GigaSourceVerificationError", false));
        }
    }
    request
        .sources()
        .iter()
        .map(|provided| {
            let source = event
                .source_refs()
                .iter()
                .find(|source| source.source_id() == provided.source_id())
                .ok_or_else(|| WorkerFailure::new("GigaSourceVerificationError", false))?;
            if sha256_bytes(provided.text().as_bytes()) != source.content_hash() {
                return Err(WorkerFailure::new("GigaSourceVerificationError", false));
            }
            Ok(ResolvedSource {
                source: source.clone(),
                text: provided.text().to_owned(),
            })
        })
        .collect()
}

async fn classify_event(
    event: &GigaEvent,
    sources: &[ResolvedSource],
) -> Result<Option<GigaCandidate>, WorkerFailure> {
    let config = ollama_config()?;
    verify_ollama_model(&config).await?;
    let model_sources = sources
        .iter()
        .map(|resolved| ModelSource {
            source_id: resolved.source.source_id(),
            role: resolved.source.role(),
            timestamp: resolved.source.timestamp(),
            text: &resolved.text,
        })
        .collect::<Vec<_>>();
    let all_source_ids = model_sources
        .iter()
        .map(|source| source.source_id.to_owned())
        .collect::<Vec<_>>();
    let gate: GateOutput = request_ollama_structured(
        &config,
        GIGA_GATE_PROMPT,
        json!({
            "project_keys": event.project_keys(),
            "sources": model_sources
        }),
        schema_json(gate_schema(&all_source_ids))?,
        256,
    )
    .await?;
    validate_gate(&gate, event)?;
    if gate.kind == GateKind::None {
        return Ok(None);
    }
    let extraction: ExtractionOutput = request_ollama_structured(
        &config,
        GIGA_EXTRACTION_PROMPT,
        json!({
            "fixed_kind": gate.kind.as_str(),
            "gate_source_ids": gate.source_ids,
            "project_keys": event.project_keys(),
            "sources": model_sources
        }),
        extraction_schema(&gate.source_ids, gate.kind.requires_proof())?,
        768,
    )
    .await?;
    validate_extraction(&extraction, &gate)?;
    let rationale = truncate_with_marker(
        &extraction.rationale,
        GIGA_MAX_STORED_RATIONALE_BYTES,
        GIGA_RATIONALE_TRUNCATION_MARKER,
    );
    let selected = sources
        .iter()
        .filter(|resolved| {
            extraction
                .source_ids
                .iter()
                .any(|source_id| source_id == resolved.source.source_id())
        })
        .map(|resolved| resolved.source.clone())
        .collect::<Vec<_>>();
    if selected.len() != extraction.source_ids.len() {
        return Err(domain_failure());
    }
    let completed_at = Utc::now().to_rfc3339();
    let classifier = GigaClassifierIdentity::new(
        GIGA_MODEL_TAG.into(),
        "ollama".into(),
        GIGA_MODEL_MANIFEST_DIGEST.into(),
        GIGA_PROMPT_VERSION.into(),
        configuration_digest(&config)?,
        Uuid::new_v4().to_string(),
        completed_at.clone(),
    )
    .map_err(|_| domain_failure())?;
    let scores = GigaScores::new(
        extraction.priority,
        extraction.novelty,
        extraction.durability,
        extraction.confidence,
    )
    .map_err(|_| domain_failure())?;
    let project = event.project_keys().first().cloned();
    let scope = GigaScope::new(
        Some(event.room().to_string()),
        project,
        GigaVisibility::Private,
        true,
    )
    .map_err(|_| domain_failure())?;
    GigaCandidate::new(
        candidate_id(event, gate.kind, &extraction.source_ids)?,
        event.event_id().into(),
        event.room().clone(),
        event.session_id().into(),
        gate.kind.candidate_kind()?,
        selected,
        extraction.proof_source_ids,
        scores,
        event.project_keys().to_vec(),
        Vec::new(),
        Vec::new(),
        extraction.retrieval_terms,
        extraction.proposed_title,
        extraction.gist,
        rationale,
        scope,
        GigaAuthority::PointerOnly,
        GigaReviewState::Unreviewed,
        classifier,
        completed_at,
        None,
        Vec::new(),
    )
    .map(Some)
    .map_err(|_| domain_failure())
}

fn safe_error_class(value: Option<String>) -> Option<String> {
    value.filter(|class| {
        !class.is_empty()
            && class.len() <= GIGA_MAX_DIAGNOSTIC_CLASS_BYTES
            && class
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    })
}

fn terminal_result(
    event_id: &str,
    state: &str,
    candidate_count: i32,
    attempt_count: i32,
    error_class: Option<String>,
) -> Result<GigaProcessResult, AppError> {
    let outcome = match state {
        "succeeded" => GigaEventFinishOutcome::Succeeded.as_str(),
        "failed" => GigaEventFinishOutcome::Failed.as_str(),
        _ => {
            return Err(AppError::Invalid(
                "stored GIGA queue state is invalid".into(),
            ));
        }
    };
    Ok(GigaProcessResult {
        event_id: event_id.into(),
        outcome: outcome.into(),
        candidate_count: u32::try_from(candidate_count)
            .map_err(|_| AppError::Invalid("stored GIGA candidate count is invalid".into()))?,
        attempt_count: u32::try_from(attempt_count)
            .map_err(|_| AppError::Invalid("stored GIGA attempt count is invalid".into()))?,
        error_class: RequiredNullable(safe_error_class(error_class)),
    })
}

async fn claim_exact(pool: &PgPool, event_id: &str) -> Result<ClaimState, AppError> {
    let mut tx = pool.begin().await?;
    let row = sqlx::query(
        "SELECT room,queue_state,available_at,locked_by,lease_expires_at,replay_count,
                attempt_count,candidate_count,last_error
         FROM giga_events WHERE event_id=$1 FOR UPDATE",
    )
    .bind(event_id)
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(|| AppError::Invalid("GIGA process event does not exist".into()))?;
    let claimed_at = database_now(&mut tx).await?;
    let room: String = row
        .try_get::<Option<String>, _>("room")?
        .ok_or_else(|| AppError::Invalid("GIGA process event has no room".into()))?;
    let state: String = row.try_get("queue_state")?;
    let previous_attempt: i32 = row.try_get("attempt_count")?;
    let candidate_count: i32 = row.try_get("candidate_count")?;
    if matches!(state.as_str(), "succeeded" | "failed") {
        let result = terminal_result(
            event_id,
            &state,
            candidate_count,
            previous_attempt,
            row.try_get("last_error")?,
        )?;
        let event = event_from_store(&mut tx, event_id).await?;
        tx.commit().await?;
        return Ok(ClaimState::Terminal { event, result });
    }
    let replay_count: i32 = row.try_get("replay_count")?;
    let previous_running = state == "running";
    if previous_running {
        let lease_expires_at = row
            .try_get::<Option<DateTime<Utc>>, _>("lease_expires_at")?
            .ok_or_else(|| AppError::Invalid("running GIGA event has no lease expiry".into()))?;
        if lease_expires_at > claimed_at {
            tx.commit().await?;
            return Ok(ClaimState::WaitUntil(lease_expires_at));
        }
        sqlx::query(
            "UPDATE giga_event_attempts
             SET outcome='lease_expired',error_class=$4,finished_at=$5
             WHERE event_id=$1 AND replay_count=$2 AND attempt_count=$3 AND finished_at IS NULL",
        )
        .bind(event_id)
        .bind(replay_count)
        .bind(previous_attempt)
        .bind(if previous_attempt >= GIGA_MAX_EVENT_ATTEMPTS as i32 {
            "GigaLeaseExpiredRetryExhausted"
        } else {
            "GigaLeaseExpired"
        })
        .bind(claimed_at)
        .execute(&mut *tx)
        .await?;
    } else if state != "pending" {
        return Err(AppError::Invalid(
            "stored GIGA queue state is invalid".into(),
        ));
    } else {
        let available_at: DateTime<Utc> = row.try_get("available_at")?;
        if available_at > claimed_at {
            tx.commit().await?;
            return Ok(ClaimState::WaitUntil(available_at));
        }
    }
    if previous_attempt >= GIGA_MAX_EVENT_ATTEMPTS as i32 {
        sqlx::query(
            "UPDATE giga_events
             SET queue_state='failed',locked_by=NULL,locked_at=NULL,lease_expires_at=NULL,
                 last_error='GigaRetryExhausted',processed_at=$2,last_finished_at=$2,updated_at=$2
             WHERE event_id=$1",
        )
        .bind(event_id)
        .bind(claimed_at)
        .execute(&mut *tx)
        .await?;
        let event = event_from_store(&mut tx, event_id).await?;
        let result = terminal_result(
            event_id,
            "failed",
            candidate_count,
            previous_attempt,
            Some("GigaRetryExhausted".into()),
        )?;
        tx.commit().await?;
        return Ok(ClaimState::Terminal { event, result });
    }
    let attempt_count = previous_attempt + 1;
    let lease_expires_at = claimed_at + ChronoDuration::seconds(GIGA_LEASE_SECONDS);
    sqlx::query(
        "UPDATE giga_events
         SET queue_state='running',attempt_count=$2,
             retry_count=retry_count+CASE WHEN $3 THEN 1 ELSE 0 END,
             locked_by=$4,locked_at=$5,lease_expires_at=$6,candidate_count=0,
             processed_at=NULL,last_finished_at=NULL,updated_at=$5
         WHERE event_id=$1",
    )
    .bind(event_id)
    .bind(attempt_count)
    .bind(previous_running)
    .bind(GIGA_WORKER_ID.as_str())
    .bind(claimed_at)
    .bind(lease_expires_at)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "INSERT INTO giga_event_attempts
         (event_id,replay_count,attempt_count,room,worker_id,claimed_at,lease_expires_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7)",
    )
    .bind(event_id)
    .bind(replay_count)
    .bind(attempt_count)
    .bind(room)
    .bind(GIGA_WORKER_ID.as_str())
    .bind(claimed_at)
    .bind(lease_expires_at)
    .execute(&mut *tx)
    .await?;
    let event = event_from_store(&mut tx, event_id).await?;
    tx.commit().await?;
    Ok(ClaimState::Claimed {
        event,
        attempt_count: u32::try_from(attempt_count)
            .map_err(|_| AppError::Invalid("GIGA attempt count is invalid".into()))?,
    })
}

fn finish_request(
    event: &GigaEvent,
    outcome: GigaEventFinishOutcome,
    candidate_count: u32,
    error_class: Option<&str>,
) -> Result<GigaEventFinishRequest, AppError> {
    GigaEventFinishRequest::new(
        event.room().clone(),
        event.event_id().into(),
        GIGA_WORKER_ID.to_string(),
        outcome,
        candidate_count,
        error_class.map(str::to_owned),
        (outcome == GigaEventFinishOutcome::Retry).then_some(GIGA_RETRY_DELAY_SECONDS),
    )
    .map_err(|error| AppError::Invalid(error.to_string()))
}

fn process_result(
    event: &GigaEvent,
    attempt_count: u32,
    outcome: GigaEventFinishOutcome,
    candidate_count: u32,
    error_class: Option<&str>,
) -> GigaProcessResult {
    GigaProcessResult {
        event_id: event.event_id().into(),
        outcome: outcome.as_str().into(),
        candidate_count,
        attempt_count,
        error_class: RequiredNullable(error_class.map(str::to_owned)),
    }
}

async fn finish_attempt(
    pool: &PgPool,
    event: &GigaEvent,
    attempt_count: u32,
    outcome: GigaEventFinishOutcome,
    candidate_count: u32,
    error_class: Option<&str>,
) -> Result<GigaProcessResult, AppError> {
    let request = finish_request(event, outcome, candidate_count, error_class)?;
    giga_event_finish(pool, request).await?;
    Ok(process_result(
        event,
        attempt_count,
        outcome,
        candidate_count,
        error_class,
    ))
}

async fn store_candidate_and_finish(
    pool: &PgPool,
    event: &GigaEvent,
    attempt_count: u32,
    candidate: GigaCandidate,
) -> Result<GigaProcessResult, AppError> {
    let request = finish_request(event, GigaEventFinishOutcome::Succeeded, 1, None)?;
    giga_candidate_store_and_finish(pool, candidate, request).await?;
    Ok(process_result(
        event,
        attempt_count,
        GigaEventFinishOutcome::Succeeded,
        1,
        None,
    ))
}

fn source_digest(event: &GigaEvent) -> String {
    let mut digest = Sha256::new();
    for source in event.source_refs() {
        digest.update(source.source_id().as_bytes());
        digest.update([0_u8]);
        digest.update(source.content_hash().as_bytes());
        digest.update([0xff_u8]);
    }
    format!("{:x}", digest.finalize())
}

async fn wait_until(target: DateTime<Utc>) {
    let delay = (target - Utc::now())
        .to_std()
        .unwrap_or_else(|_| Duration::from_millis(1));
    sleep(delay.min(Duration::from_secs(GIGA_LEASE_SECONDS as u64))).await;
}

pub async fn giga_process(
    pool: &PgPool,
    request: GigaProcessRequest,
) -> Result<GigaProcessResult, AppError> {
    loop {
        let (event, attempt_count) = match claim_exact(pool, request.event_id()).await? {
            ClaimState::Terminal { event, result } => {
                validate_source_bundle(&event, &request).map_err(|failure| {
                    AppError::Invalid(format!("GIGA process failed: {}", failure.class))
                })?;
                return Ok(result);
            }
            ClaimState::WaitUntil(target) => {
                wait_until(target).await;
                continue;
            }
            ClaimState::Claimed {
                event,
                attempt_count,
            } => (event, attempt_count),
        };
        let source_hash = source_digest(&event);
        let event_hash = sha256_bytes(event.event_id().as_bytes());
        let result = validate_source_bundle(&event, &request);
        let classified = match result {
            Ok(sources) => {
                let existing_candidates: i64 = sqlx::query_scalar(
                    "SELECT COUNT(*)::bigint FROM giga_candidates WHERE event_id=$1",
                )
                .bind(event.event_id())
                .fetch_one(pool)
                .await?;
                match existing_candidates {
                    0 => classify_event(&event, &sources).await,
                    1 => {
                        tracing::info!(
                            operation = "giga_process",
                            event_hash = %event_hash,
                            source_hash = %source_hash,
                            source_count = event.source_refs().len(),
                            candidate_count = 1,
                            outcome = "succeeded",
                            recovery = "existing_candidate",
                            model = GIGA_MODEL_TAG,
                            model_digest = GIGA_MODEL_MANIFEST_DIGEST,
                            prompt_version = GIGA_PROMPT_VERSION,
                        );
                        return finish_attempt(
                            pool,
                            &event,
                            attempt_count,
                            GigaEventFinishOutcome::Succeeded,
                            1,
                            None,
                        )
                        .await;
                    }
                    _ => {
                        return Err(AppError::Invalid(
                            "GIGA event has more than one durable candidate".into(),
                        ));
                    }
                }
            }
            Err(failure) => Err(failure),
        };
        match classified {
            Ok(None) => {
                let result = finish_attempt(
                    pool,
                    &event,
                    attempt_count,
                    GigaEventFinishOutcome::Succeeded,
                    0,
                    None,
                )
                .await?;
                tracing::info!(
                    operation = "giga_process",
                    event_hash = %event_hash,
                    source_hash = %source_hash,
                    source_count = event.source_refs().len(),
                    candidate_count = 0,
                    outcome = "succeeded",
                    model = GIGA_MODEL_TAG,
                    model_digest = GIGA_MODEL_MANIFEST_DIGEST,
                    prompt_version = GIGA_PROMPT_VERSION,
                );
                return Ok(result);
            }
            Ok(Some(candidate)) => {
                let result =
                    store_candidate_and_finish(pool, &event, attempt_count, candidate).await?;
                tracing::info!(
                    operation = "giga_process",
                    event_hash = %event_hash,
                    source_hash = %source_hash,
                    source_count = event.source_refs().len(),
                    candidate_count = 1,
                    outcome = "succeeded",
                    model = GIGA_MODEL_TAG,
                    model_digest = GIGA_MODEL_MANIFEST_DIGEST,
                    prompt_version = GIGA_PROMPT_VERSION,
                );
                return Ok(result);
            }
            Err(failure) => {
                let retry = failure.retryable && attempt_count < GIGA_MAX_EVENT_ATTEMPTS;
                let outcome = if retry {
                    GigaEventFinishOutcome::Retry
                } else {
                    GigaEventFinishOutcome::Failed
                };
                tracing::warn!(
                    operation = "giga_process",
                    event_hash = %event_hash,
                    source_hash = %source_hash,
                    source_count = event.source_refs().len(),
                    candidate_count = 0,
                    outcome = outcome.as_str(),
                    error_class = failure.class,
                    model = GIGA_MODEL_TAG,
                    model_digest = GIGA_MODEL_MANIFEST_DIGEST,
                    prompt_version = GIGA_PROMPT_VERSION,
                );
                let finished =
                    finish_attempt(pool, &event, attempt_count, outcome, 0, Some(failure.class))
                        .await?;
                if retry {
                    continue;
                }
                return Ok(finished);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use house_core::{GigaLifecycle, GigaProcessSource, RoomKey};

    fn conversation_event(
        room: &RoomKey,
        project: Option<&str>,
        texts: &[(&str, &str, &str)],
    ) -> GigaEvent {
        let scope = GigaScope::new(
            Some(room.to_string()),
            project.map(str::to_owned),
            GigaVisibility::Private,
            true,
        )
        .unwrap();
        let sources = texts
            .iter()
            .enumerate()
            .map(|(index, (source_id, role, text))| {
                GigaSourceRef::new(
                    GigaSourceType::Turn,
                    (*source_id).into(),
                    (*role).into(),
                    format!("2026-07-24T12:00:0{index}Z"),
                    sha256_bytes(text.as_bytes()),
                    scope.clone(),
                    None,
                )
                .unwrap()
            })
            .collect();
        GigaEvent::new(
            "event-1".into(),
            GigaEventType::ConversationWindow,
            room.clone(),
            "session-1".into(),
            project.into_iter().map(str::to_owned).collect(),
            sources,
            GigaLifecycle::conversation_window(),
            "2026-07-24T12:00:10Z".into(),
        )
        .unwrap()
    }

    #[test]
    fn exact_source_bundle_recomputes_hashes_and_preserves_turn_order() {
        let room = RoomKey::new("lab").unwrap();
        let event = conversation_event(
            &room,
            None,
            &[
                ("turn-1", "user", "exact user text"),
                ("turn-2", "assistant", "exact assistant text"),
            ],
        );
        let request = GigaProcessRequest::new(
            "event-1".into(),
            vec![
                GigaProcessSource::new("turn-2".into(), "exact assistant text".into()).unwrap(),
                GigaProcessSource::new("turn-1".into(), "exact user text".into()).unwrap(),
            ],
        )
        .unwrap();
        let resolved = validate_source_bundle(&event, &request).unwrap();
        assert_eq!(resolved[0].source.source_id(), "turn-2");
        assert_eq!(resolved[1].source.source_id(), "turn-1");

        let tampered = GigaProcessRequest::new(
            "event-1".into(),
            vec![
                GigaProcessSource::new("turn-1".into(), "changed".into()).unwrap(),
                GigaProcessSource::new("turn-2".into(), "exact assistant text".into()).unwrap(),
            ],
        )
        .unwrap();
        assert!(validate_source_bundle(&event, &tampered).is_err());
    }

    #[test]
    fn semantic_gate_and_extractor_validation_rejects_invalid_kinds_proofs_and_scores() {
        let room = RoomKey::new("lab").unwrap();
        let event = conversation_event(
            &room,
            None,
            &[("turn-1", "user", "rule"), ("turn-2", "assistant", "proof")],
        );
        let valid_gate = GateOutput {
            kind: GateKind::CodingLesson,
            source_ids: vec!["turn-1".into(), "turn-2".into()],
            reason: "Explicit reusable rule with proof".into(),
        };
        validate_gate(&valid_gate, &event).unwrap();
        assert!(
            validate_gate(
                &GateOutput {
                    kind: GateKind::ProjectLesson,
                    source_ids: vec!["turn-1".into()],
                    reason: "No project exists".into(),
                },
                &event,
            )
            .is_err()
        );
        assert!(
            validate_gate(
                &GateOutput {
                    kind: GateKind::None,
                    source_ids: vec!["turn-1".into()],
                    reason: "none".into(),
                },
                &event,
            )
            .is_err()
        );

        let invalid_extraction = ExtractionOutput {
            source_ids: vec!["turn-1".into(), "turn-2".into()],
            proof_source_ids: Vec::new(),
            proposed_title: "Rule".into(),
            gist: "Apply the rule".into(),
            rationale: "Observed proof".into(),
            priority: 2.0,
            novelty: 0.5,
            durability: 0.8,
            confidence: 0.9,
            retrieval_terms: vec!["rule".into()],
        };
        assert!(validate_extraction(&invalid_extraction, &valid_gate).is_err());
    }

    #[test]
    fn candidate_identity_is_deterministic_across_source_order() {
        let room = RoomKey::new("lab").unwrap();
        let event = conversation_event(
            &room,
            None,
            &[("turn-1", "user", "rule"), ("turn-2", "assistant", "proof")],
        );
        let forward = candidate_id(
            &event,
            GateKind::CodingLesson,
            &["turn-1".into(), "turn-2".into()],
        )
        .unwrap();
        let reverse = candidate_id(
            &event,
            GateKind::CodingLesson,
            &["turn-2".into(), "turn-1".into()],
        )
        .unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(forward.len(), 64);
    }

    #[test]
    fn classifier_configuration_digest_includes_normalized_provider_base_path() {
        let first = OllamaConfig {
            endpoint: Url::parse("http://127.0.0.1:11435/route-a").unwrap(),
        };
        let second = OllamaConfig {
            endpoint: Url::parse("http://127.0.0.1:11435/route-b").unwrap(),
        };
        assert_ne!(
            configuration_digest(&first).unwrap(),
            configuration_digest(&second).unwrap()
        );
    }
    #[test]
    fn extraction_schema_keeps_rationale_before_gist() {
        let schema = extraction_schema(&["turn-1".into()], true).unwrap();
        let properties = schema.split("\"properties\":").nth(1).unwrap();
        assert!(
            properties.find("\"rationale\"").unwrap() < properties.find("\"gist\"").unwrap()
        );
    }

    #[test]
    fn classifier_rationale_is_bounded_with_an_explicit_marker() {
        let source = "r".repeat(GIGA_MAX_STORED_RATIONALE_BYTES + 32);
        let stored = truncate_with_marker(
            &source,
            GIGA_MAX_STORED_RATIONALE_BYTES,
            GIGA_RATIONALE_TRUNCATION_MARKER,
        );
        assert!(stored.len() <= GIGA_MAX_STORED_RATIONALE_BYTES);
        assert!(stored.ends_with(GIGA_RATIONALE_TRUNCATION_MARKER));
    }
}
