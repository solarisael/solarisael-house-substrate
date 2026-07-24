use regex::Regex;
use reqwest::Client;
use sqlx::{
    PgPool,
    postgres::{PgConnectOptions, PgPoolOptions},
};
use std::{env, fs, io, path::Path, str::FromStr, sync::LazyLock, time::Duration};
use thiserror::Error;

const DEFAULT_EMBED_URL: &str = "http://127.0.0.1:11435/api/embed";
const DEFAULT_EMBED_MODEL: &str = "hf.co/zenmagnets/Nemotron-3-Embed-1B-Q4_K_M-GGUF:latest";
pub(crate) const EMBED_DIMENSION: usize = 2048;
pub(crate) static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(Client::new);
pub(crate) static ROOM_KEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[a-z0-9]+(?:-[a-z0-9]+)*$").expect("room key regex must compile")
});
pub(crate) static PATH_DATE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(20\d{2})[-_](\d{2})[-_](\d{2})").expect("path date regex must compile")
});
pub(crate) static STITCHED_PATH_DATE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(20\d{2})[-_](\d{2})[-_](\d{2})[_-](\d{2})")
        .expect("stitched path date regex must compile")
});
pub(crate) static QUERY_DATE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(20\d{2})-(\d{2})-(\d{2})\b").expect("query date regex must compile")
});

#[derive(Debug, Error)]
pub enum AppError {
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("configuration error: {0}")]
    Config(String),
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("io error: {0}")]
    Io(#[from] io::Error),
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
    if let Ok(v) = env::var(key) {
        return Some(v);
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(".env");
    let text = fs::read_to_string(path).ok()?;
    text.lines().find_map(|line| {
        let line = line.trim();
        let (k, v) = line.split_once('=')?;
        if k.trim() == key {
            Some(v.trim().trim_matches('"').trim_matches('\'').to_string())
        } else {
            None
        }
    })
}

impl Config {
    pub fn from_env() -> Result<Self, AppError> {
        let database_url = env::var("SOLARISAEL_SUBSTRATE_TEST_DATABASE_URL")
            .ok()
            .or_else(|| dotenv_value("DATABASE_URL"))
            .or_else(|| {
                let host = dotenv_value("PGHOST")?;
                let port = dotenv_value("PGPORT")
                    .unwrap_or_else(|| "5432".into())
                    .parse()
                    .ok()?;
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
            .ok_or_else(|| {
                AppError::Config("DATABASE_URL or complete PG* variables required".into())
            })?;
        let embed_url =
            Some(dotenv_value("SOLARISAEL_EMBED_URL").unwrap_or_else(|| DEFAULT_EMBED_URL.into()));
        let embed_dimension = dotenv_value("SOLARISAEL_EMBED_DIMENSION")
            .unwrap_or_else(|| EMBED_DIMENSION.to_string())
            .parse()
            .map_err(|_| {
                AppError::Config("SOLARISAEL_EMBED_DIMENSION must be an integer".into())
            })?;
        if embed_dimension != EMBED_DIMENSION {
            return Err(AppError::Config(
                "embedding dimension must be 2048 for migration 0002".into(),
            ));
        }
        let test_embedding_disabled =
            dotenv_value("SOLARISAEL_TEST_DISABLE_EMBEDDING").as_deref() == Some("1");
        Ok(Self {
            database_url,
            embed_model: dotenv_value("SOLARISAEL_EMBED_MODEL")
                .unwrap_or_else(|| DEFAULT_EMBED_MODEL.into()),
            embed_dimension,
            embed_required: !test_embedding_disabled,
            test_embedding_disabled,
            embed_url,
        })
    }
    pub async fn pool(&self) -> Result<PgPool, AppError> {
        let options = PgConnectOptions::from_str(&self.database_url)
            .map_err(|e| AppError::Config(format!("invalid database configuration: {e}")))?;
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .acquire_timeout(Duration::from_secs(120))
            .connect_with(options)
            .await?;
        let shape: String = sqlx::query_scalar("SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid WHERE c.relname='memory_chunks' AND a.attname='body_embedding' AND NOT a.attisdropped")
        .fetch_optional(&pool).await?.ok_or_else(|| AppError::Config("memory_chunks.body_embedding is missing; apply migration 0002".into()))?;
        if shape != "vector(2048)" {
            return Err(AppError::Config(format!(
                "incompatible embedding schema: {shape}"
            )));
        }
        Ok(pool)
    }
}
