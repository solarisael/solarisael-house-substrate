use serde::{Deserialize, Serialize};
use serde_json::Value;
use solarisael_house_substrate::{process_request, recall, AppError, Config, RecallParams, RememberRequest};
use std::collections::BTreeSet;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::error;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    protocol: Option<u64>,
    id: Option<String>,
    method: Option<String>,
    params: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProtocolRememberRequest {
    room: String,
    kind: String,
    title: String,
    #[serde(default)]
    body: String,
    #[serde(default)]
    lesson: Option<String>,
    #[serde(default)]
    source_path: Option<String>,
    #[serde(default)]
    source_memory_path: Option<String>,
    #[serde(default)]
    threads: Vec<String>,
    #[serde(default)]
    supersedes: Vec<String>,
    #[serde(default)]
    shape: Option<String>,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    project: Option<String>,
    #[serde(default, alias = "proofPattern")]
    proof_pattern: Option<String>,
    #[serde(default, alias = "triggerContext")]
    trigger_context: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default = "default_backup")]
    backup: bool,
}

#[derive(Debug)]
enum ProtocolRequest {
    Remember(RememberRequest),
    Recall(RecallParams),
}

fn default_backup() -> bool { true }

#[derive(Debug, Serialize)]
struct Response<'a, T: Serialize> {
    protocol: u64,
    id: String,
    #[serde(flatten)]
    body: Body<'a, T>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Body<'a, T: Serialize> {
    Result { result: &'a T },
    Error { error: ErrorBody },
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    code: String,
    message: String,
    retryable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<Value>,
}

fn error_response(id: String, code: &str, message: impl Into<String>, retryable: bool) -> String {
    serde_json::to_string(&Response::<Value> {
        protocol: 1,
        id,
        body: Body::Error {
            error: ErrorBody {
                code: code.into(),
                message: message.into(),
                retryable,
                details: None,
            },
        },
    })
    .expect("response serialization cannot fail")
}

fn app_error(id: String, e: AppError) -> String {
    let (code, retryable) = match &e {
        AppError::Invalid(_) => ("invalid_params", false),
        AppError::Config(_) => ("configuration", false),
        AppError::Protocol(_) => ("protocol", false),
        AppError::Embedding(_) => ("embedding", true),
        AppError::Database(_) => ("database", true),
        AppError::Io(_) => ("io", true),
    };
    error_response(id, code, e.to_string(), retryable)
}
fn parse_request(value: Value) -> Result<ProtocolRequest, String> {
    let request: ProtocolRememberRequest =
        serde_json::from_value(value).map_err(|e| format!("params must be a remember request: {e}"))?;
    let mut supersedes = BTreeSet::new();
    for raw in request.supersedes {
        if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) { return Err("supersedes must contain positive decimal strings".into()); }
        let id = raw.parse::<i64>().map_err(|_| "supersedes ID is out of range".to_string())?;
        if id <= 0 { return Err("supersedes must contain positive IDs".into()); }
        supersedes.insert(id);
    }
    Ok(ProtocolRequest::Remember(RememberRequest {
        room: request.room,
        kind: request.kind,
        title: request.title,
        body: request.body,
        lesson: request.lesson,
        source_path: request.source_path,
        source_memory_path: request.source_memory_path,
        threads: request.threads,
        supersedes: supersedes.into_iter().collect(),
        shape: request.shape,
        voice: request.voice,
        scope: request.scope,
        project: request.project,
        proof_pattern: request.proof_pattern,
        trigger_context: request.trigger_context,
        tags: request.tags,
        backup: request.backup,
    }))
}

fn parse_recall(value: Value) -> Result<ProtocolRequest, String> {
    serde_json::from_value(value)
        .map(ProtocolRequest::Recall)
        .map_err(|e| format!("params must be recall parameters: {e}"))
}

fn decode_line(line: &str) -> (String, Result<ProtocolRequest, String>) {
    let parsed: Result<Envelope, _> = serde_json::from_str(line);
    let env = match parsed {
        Ok(env) => env,
        Err(e) => return ("unknown".into(), Err(format!("malformed request: {e}"))),
    };
    let Some(raw_id) = env.id.clone() else {
        return ("unknown".into(), Err("id is required".into()));
    };
    let id = raw_id;
    if id.trim().is_empty() {
        return (id, Err("id must be non-empty".into()));
    }
    if env.protocol != Some(1) {
        return (id, Err("protocol must be 1".into()));
    }
    let Some(method) = env.method.as_deref() else {
        return (id, Err("method is required".into()));
    };
    let Some(params) = env.params else {
        return (id, Err("params are required".into()));
    };
    match method {
        "remember" => (id, parse_request(params)),
        "recall" => (id, parse_recall(params)),
        _ => (id, Err("method is not supported".into())),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_writer(std::io::stderr).with_env_filter("warn").init();
    let cfg = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            error!("configuration failed: {e}");
            return Err(e.into());
        }
    };
    let pool = cfg.pool().await?;
    let stdin = BufReader::new(io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = io::BufWriter::new(io::stdout());
    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        let (id, request) = decode_line(trimmed);
        let response = match request {
            Ok(ProtocolRequest::Remember(req)) => match process_request(&pool, &cfg, req).await {
                Ok(receipt) => serde_json::to_string(&Response {
                    protocol: 1,
                    id: id.clone(),
                    body: Body::Result { result: &receipt },
                })?,
                Err(e) => app_error(id.clone(), e),
            },
            Ok(ProtocolRequest::Recall(params)) => match recall(&pool, &cfg, params).await {
                Ok(result) => serde_json::to_string(&Response {
                    protocol: 1,
                    id: id.clone(),
                    body: Body::Result { result: &result },
                })?,
                Err(e) => app_error(id.clone(), e),
            },
            Err(message) => {
                let code = if message.starts_with("malformed request") || message.starts_with("id ") {
                    "malformed_request"
                } else if message == "protocol must be 1" {
                    "protocol_mismatch"
                } else if message == "method is not supported" {
                    "unknown_method"
                } else {
                    "invalid_params"
                };
                error_response(id, code, message, false)
            }
        };
        stdout.write_all(format!("{response}\n").as_bytes()).await?;
        stdout.flush().await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_supersedes_strings_are_positive_and_deduplicated() {
        let req = parse_request(serde_json::json!({
            "room": "room",
            "kind": "memory",
            "title": "title",
            "body": "body",
            "supersedes": ["12", "3", "12"]
        })).unwrap();
        match req { ProtocolRequest::Remember(req) => assert_eq!(req.supersedes, vec![3, 12]), _ => panic!("expected remember") }
    }

    #[test]
    fn rejects_non_decimal_supersedes() {
        let err = parse_request(serde_json::json!({
            "room": "room", "kind": "memory", "title": "title", "body": "body",
            "supersedes": ["+1"]
        })).unwrap_err();
        assert!(err.contains("decimal"));
    }

    #[test]
    fn recall_protocol_params_are_strict() {
        let (id, decoded) = decode_line(r#"{"protocol":1,"id":"r1","method":"recall","params":{"room":"room","query":"needle","unexpected":true}}"#);
        assert_eq!(id, "r1");
        assert!(decoded.unwrap_err().contains("recall parameters"));
        let (_, valid) = decode_line(r#"{"protocol":1,"id":"r2","method":"recall","params":{"room":"room","query":"needle"}}"#);
        assert!(matches!(valid.unwrap(), ProtocolRequest::Recall(_)));
    }

    #[test]
    fn protocol_errors_have_current_codes() {
        let (id, result) = decode_line(r#"{"protocol":2,"id":"x","method":"remember","params":{}}"#);
        assert_eq!(id, "x");
        assert_eq!(result.unwrap_err(), "protocol must be 1");
    }
}
