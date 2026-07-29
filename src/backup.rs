use chrono::{DateTime, Utc};
use house_protocol::{
    DiagnosticCategory, DiagnosticDetails, DiagnosticEvidence, DiagnosticExecution,
    DiagnosticNextCheck, DiagnosticOwner, DiagnosticRetry, DiagnosticStage, DiagnosticTarget,
    DiagnosticTargetKind, DiagnosticWriteOutcome,
};
use percent_encoding::percent_decode_str;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Row};
use std::{
    env, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
use uuid::Uuid;

const CONSOLIDATED_MIGRATIONS: &[&str] = &["1", "2", "3", "4", "5", "6", "7"];
const LEGACY_MIGRATIONS: &[&str] = &[
    "0001_create_memories",
    "0002_memory_threads_pivot",
    "0003_named_entities",
    "0004_coding_lessons",
    "0005_discord_chat",
    "0006_channel_summaries",
    "0007_continuity_rails",
    "0008_coding_lessons_voice_negation",
    "0009_bot_decision_rows",
    "0009_memories_dates_array",
    "0010_gym_walk_ledger",
    "0011_wake_triggers",
    "0012_project_lessons",
    "0013_coding_lessons_intention_alignment",
    "0014_coding_lessons_long_running_processes",
    "0015_coding_lessons_powershell_encoding",
    "0016_pgvector_and_chunks_8b",
    "0017_memory_chunks_4b",
    "0018_coding_lessons_semantic_duplication",
    "0019_writing_lessons",
    "0020_anamnesis_cabinet",
    "0021_coding_lessons_always_on",
    "0022_memory_clusters_live_space",
    "0023_memory_clusters_centroid",
    "0024_memory_erasure",
    "0025_nemotron_2048",
];

fn known_migration_lineage(versions: &[String]) -> bool {
    !versions.is_empty()
        && [CONSOLIDATED_MIGRATIONS, LEGACY_MIGRATIONS]
            .iter()
            .any(|lineage| {
                versions.len() <= lineage.len()
                    && versions
                        .iter()
                        .zip(lineage.iter())
                        .all(|(actual, expected)| actual == expected)
            })
}

#[derive(Debug, thiserror::Error)]
pub enum BackupError {
    #[error("backup configuration: {0}")]
    Config(String),
    #[error("backup io: {0}")]
    Io(#[from] io::Error),
    #[error("backup command failed: {0}")]
    Command(String),
    #[error("backup manifest: {0}")]
    Manifest(String),
}
impl BackupError {
    pub fn diagnostics(&self, operation: &str) -> DiagnosticDetails {
        let (failure, retry, write_outcome, target) = match self {
            Self::Config(_) => (
                "configuration_invalid",
                DiagnosticRetry::AfterChange,
                DiagnosticWriteOutcome::NotStarted,
                DiagnosticTarget::new(DiagnosticTargetKind::File, "src/backup.rs"),
            ),
            Self::Io(_) => (
                "filesystem_error",
                DiagnosticRetry::ReconcileFirst,
                DiagnosticWriteOutcome::Unknown,
                DiagnosticTarget::new(DiagnosticTargetKind::File, "backup output directory"),
            ),
            Self::Command(_) => (
                "postgres_command_failed",
                DiagnosticRetry::ReconcileFirst,
                DiagnosticWriteOutcome::Unknown,
                DiagnosticTarget::new(DiagnosticTargetKind::Service, "pg_dump or pg_restore"),
            ),
            Self::Manifest(_) => (
                "manifest_invalid",
                DiagnosticRetry::AfterChange,
                DiagnosticWriteOutcome::NotStarted,
                DiagnosticTarget::new(DiagnosticTargetKind::File, "backup manifest"),
            ),
        };
        let observed = match self {
            Self::Io(error) => serde_json::json!({
                "failure": failure,
                "io_error_kind": error.kind().to_string(),
            }),
            _ => serde_json::json!({"failure": failure}),
        };
        DiagnosticDetails::new(DiagnosticCategory::Backup, DiagnosticStage::Backup)
            .operation(operation)
            .owner(
                DiagnosticOwner::new("solarisael-house-substrate")
                    .path("src/backup.rs")
                    .symbol(match operation {
                        "restore" => "restore_checked",
                        _ => "backup_with_migrations",
                    }),
            )
            .expected(match operation {
                "restore" => serde_json::json!({
                    "restore": "validated manifest and confirmed target database",
                }),
                _ => serde_json::json!({
                    "backup": "durable custom-format dump and manifest",
                }),
            })
            .observed(observed.clone())
            .evidence(
                DiagnosticEvidence::new("backup_failure")
                    .summary("Backup diagnostics omit command stderr, database URLs, and passwords")
                    .data(observed),
            )
            .target(DiagnosticTarget::new(
                DiagnosticTargetKind::File,
                "src/backup.rs",
            ))
            .target(target.clone())
            .next_check(
                DiagnosticNextCheck::new("inspect_backup_target")
                    .target(target)
                    .expected(serde_json::json!({"failure_resolved": failure})),
            )
            .next_check(
                DiagnosticNextCheck::new(if retry == DiagnosticRetry::ReconcileFirst {
                    "reconcile_backup_or_restore"
                } else {
                    "retry_backup_or_restore"
                })
                .expected(serde_json::json!({"safe_retry": retry == DiagnosticRetry::SafeNow})),
            )
            .execution(DiagnosticExecution::new(
                operation == "restore",
                write_outcome,
                retry,
            ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub database: String,
    pub created_at: String,
    pub size: u64,
    pub sha256: String,
    pub format: String,
    pub schema_migrations: Vec<String>,
    pub pg_dump_version: String,
    pub dump: String,
}

fn use_wsl_pg() -> bool {
    cfg!(windows) && env::var("SOLARISAEL_PG_WSL").as_deref() == Ok("1")
}

fn pg_command(name: &str) -> Command {
    if use_wsl_pg() {
        let mut command = Command::new("wsl.exe");
        let mut wslenv = env::var("WSLENV").unwrap_or_default();
        if !wslenv.split(':').any(|entry| entry == "PGPASSWORD/u") {
            if !wslenv.is_empty() {
                wslenv.push(':');
            }
            wslenv.push_str("PGPASSWORD/u");
        }
        command.env("WSLENV", wslenv);
        command.args(["--exec", name]);
        return command;
    }
    let executable = env::var_os("PG_BIN_DIR")
        .map(PathBuf::from)
        .map(|dir| {
            dir.join(if cfg!(windows) {
                format!("{name}.exe")
            } else {
                name.into()
            })
        })
        .unwrap_or_else(|| {
            PathBuf::from(if cfg!(windows) {
                format!("{name}.exe")
            } else {
                name.into()
            })
        });
    Command::new(executable)
}

fn pg_path(path: &Path) -> Result<String, BackupError> {
    if !use_wsl_pg() {
        return Ok(path.to_string_lossy().into_owned());
    }
    let output = Command::new("wsl.exe")
        .args(["--exec", "wslpath", "-a"])
        .arg(path)
        .output()
        .map_err(BackupError::Io)?;
    if !output.status.success() {
        return Err(BackupError::Command(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
fn db_parts(raw: &str) -> Result<(String, Option<String>, String), BackupError> {
    let u =
        Url::parse(raw).map_err(|e| BackupError::Config(format!("invalid database URL: {e}")))?;
    let forbidden = [
        "dbname", "database", "service", "host", "hostaddr", "port", "user", "username", "password",
    ];
    for (k, _) in u.query_pairs() {
        if forbidden.iter().any(|x| k.eq_ignore_ascii_case(x)) {
            return Err(BackupError::Config(format!(
                "database URL query key overrides identity: {k}"
            )));
        }
    }
    let db = u.path().trim_matches('/').to_string();
    if db.is_empty() {
        return Err(BackupError::Config(
            "database URL has no database name".into(),
        ));
    }
    let password = u
        .password()
        .map(|p| percent_decode_str(p).decode_utf8_lossy().into_owned());
    let mut safe = u.clone();
    if password.is_some() {
        safe.set_password(None)
            .map_err(|_| BackupError::Config("invalid database URL password".into()))?;
    }
    Ok((db, password, safe.to_string()))
}
fn run(mut c: Command) -> Result<std::process::Output, BackupError> {
    c.stdin(Stdio::null())
        .output()
        .map_err(BackupError::Io)
        .and_then(|o| {
            if o.status.success() {
                Ok(o)
            } else {
                Err(BackupError::Command(
                    String::from_utf8_lossy(&o.stderr).trim().to_string(),
                ))
            }
        })
}
fn version(name: &str) -> String {
    pg_command(name)
        .arg("--version")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}
fn hash(path: &Path) -> Result<(u64, String), BackupError> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut n = 0;
    let mut b = [0; 8192];
    loop {
        let k = f.read(&mut b)?;
        if k == 0 {
            break;
        }
        n += k as u64;
        h.update(&b[..k]);
    }
    Ok((n, format!("{:x}", h.finalize())))
}
fn validate_dump(path: &Path, url: &str, password: Option<&str>) -> Result<(), BackupError> {
    let mut c = pg_command("pg_restore");
    c.args(["--list"])
        .arg(pg_path(path)?)
        .args(["--dbname"])
        .arg(url);
    if let Some(p) = password {
        c.env("PGPASSWORD", p);
    }
    run(c).map(|_| ())
}
fn rotate(dir: &Path, db: &str, keep: usize) -> Result<(), BackupError> {
    let mut pairs = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            let n = p.file_name()?.to_str()?.to_owned();
            if n.starts_with(&format!("{db}-")) && n.ends_with(".manifest.json") {
                let m: Manifest = serde_json::from_slice(&fs::read(&p).ok()?).ok()?;
                let t = DateTime::parse_from_rfc3339(&m.created_at)
                    .ok()?
                    .with_timezone(&Utc);
                Some((p, m, t))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| b.0.cmp(&a.0)));
    for (mp, m, _) in pairs.into_iter().skip(keep) {
        let dp = mp.parent().unwrap_or(dir).join(m.dump);
        if dp.exists() {
            fs::remove_file(dp)?;
        }
        fs::remove_file(mp)?;
    }
    Ok(())
}

pub fn backup_with_migrations(
    database_url: &str,
    output_dir: &Path,
    keep: usize,
    source: Vec<String>,
) -> Result<Manifest, BackupError> {
    if keep == 0 {
        return Err(BackupError::Config("keep must be at least 1".into()));
    }
    if !known_migration_lineage(&source) {
        return Err(BackupError::Manifest(
            "database schema migrations are unsupported".into(),
        ));
    }
    fs::create_dir_all(output_dir)?;
    let (db, password, safe) = db_parts(database_url)?;
    let stem = format!("{db}-{}", Uuid::new_v4());
    let dump_name = format!("{stem}.dump");
    let dump = output_dir.join(&dump_name);
    let tmp = output_dir.join(format!(".{stem}.tmp"));
    let mut c = pg_command("pg_dump");
    c.args(["--format=custom", "--no-owner", "--no-acl", "--file"])
        .arg(pg_path(&tmp)?)
        .args(["--dbname"])
        .arg(&safe);
    if let Some(p) = password.as_deref() {
        c.env("PGPASSWORD", p);
    }
    if let Err(e) = run(c) {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    let f = fs::OpenOptions::new().write(true).open(&tmp)?;
    f.sync_all()?;
    drop(f);
    validate_dump(&tmp, &safe, password.as_deref())?;
    fs::rename(&tmp, &dump)?;
    let (size, sha) = hash(&dump)?;
    let manifest = Manifest {
        database: db.clone(),
        created_at: Utc::now().to_rfc3339(),
        size,
        sha256: sha,
        format: "custom".into(),
        schema_migrations: source,
        pg_dump_version: version("pg_dump"),
        dump: dump_name,
    };
    let mp = output_dir.join(format!("{stem}.manifest.json"));
    let mt = output_dir.join(format!(".{stem}.manifest.tmp"));
    let data =
        serde_json::to_vec_pretty(&manifest).map_err(|e| BackupError::Manifest(e.to_string()))?;
    {
        let mut f = fs::File::create(&mt)?;
        f.write_all(&data)?;
        f.sync_all()?;
    }
    fs::rename(&mt, &mp)?;
    rotate(output_dir, &db, keep)?;
    Ok(manifest)
}
pub fn backup(database_url: &str, output_dir: &Path, keep: usize) -> Result<Manifest, BackupError> {
    backup_with_migrations(
        database_url,
        output_dir,
        keep,
        CONSOLIDATED_MIGRATIONS
            .iter()
            .map(|x| x.to_string())
            .collect(),
    )
}
pub async fn source_migrations(pool: &PgPool) -> Result<Vec<String>, BackupError> {
    let rows = sqlx::query("SELECT version::text FROM schema_migrations ORDER BY version")
        .fetch_all(pool)
        .await
        .map_err(|e| BackupError::Command(format!("migration query: {e}")))?;
    Ok(rows.into_iter().map(|r| r.get::<String, _>(0)).collect())
}
pub async fn restore_checked(
    pool: &PgPool,
    database_url: &str,
    manifest_path: &Path,
    confirm: &str,
) -> Result<(), BackupError> {
    let shape:Option<String>=sqlx::query_scalar("SELECT format_type(a.atttypid,a.atttypmod) FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid WHERE c.relname='memory_chunks' AND a.attname='body_embedding' AND NOT a.attisdropped").fetch_optional(pool).await.map_err(|e|BackupError::Command(format!("schema preflight: {e}")))?;
    if shape.as_deref() != Some("vector(2048)") {
        return Err(BackupError::Config(format!(
            "incompatible embedding schema: {}",
            shape.unwrap_or_else(|| "missing".into())
        )));
    }
    let versions = source_migrations(pool).await?;
    if !known_migration_lineage(&versions) {
        return Err(BackupError::Config(
            "schema migration versions are incompatible".into(),
        ));
    }
    restore(database_url, manifest_path, confirm)
}
pub fn restore(database_url: &str, manifest_path: &Path, confirm: &str) -> Result<(), BackupError> {
    let (db, password, safe) = db_parts(database_url)?;
    if confirm != db {
        return Err(BackupError::Config(
            "database confirmation does not match target database".into(),
        ));
    }
    let m: Manifest = serde_json::from_slice(&fs::read(manifest_path)?)
        .map_err(|e| BackupError::Manifest(e.to_string()))?;
    if m.database != db || m.format != "custom" || !known_migration_lineage(&m.schema_migrations) {
        return Err(BackupError::Manifest(
            "manifest database, format, or migrations mismatch".into(),
        ));
    }
    if Path::new(&m.dump).file_name().and_then(|x| x.to_str()) != Some(m.dump.as_str()) {
        return Err(BackupError::Manifest("manifest dump path is unsafe".into()));
    }
    let dump = manifest_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(&m.dump);
    if !dump.exists() {
        return Err(BackupError::Manifest("dump is missing".into()));
    }
    let (size, sha) = hash(&dump)?;
    if size != m.size || sha != m.sha256 {
        return Err(BackupError::Manifest(
            "dump checksum or size mismatch".into(),
        ));
    }
    validate_dump(&dump, &safe, password.as_deref())?;
    let mut c = pg_command("pg_restore");
    c.args([
        "--clean",
        "--if-exists",
        "--no-owner",
        "--no-acl",
        "--single-transaction",
        "--dbname",
    ])
    .arg(&safe)
    .arg(pg_path(&dump)?);
    if let Some(p) = password {
        c.env("PGPASSWORD", p);
    }
    run(c).map(|_| ())
}
pub fn default_backup_dir() -> PathBuf {
    env::var_os("SOLARISAEL_BACKUP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.join("backups")))
                .unwrap_or_else(|| PathBuf::from("backups"))
        })
}
pub async fn run_post_write(pool: &PgPool, database_url: &str) -> Result<(), BackupError> {
    let keep = env::var("SOLARISAEL_BACKUP_KEEP")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(3);
    let source = source_migrations(pool).await?;
    backup_with_migrations(database_url, &default_backup_dir(), keep, source).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encoded_password() {
        let (_, p, s) = db_parts("postgres://u:p%40ss%2Fword@host/db").unwrap();
        assert_eq!(p.as_deref(), Some("p@ss/word"));
        assert!(!s.contains("p%40"));
    }
    #[test]
    fn query_identity_rejected() {
        assert!(db_parts("postgres://host/db?dbname=other").is_err());
    }
    #[test]
    fn accepts_known_migration_lineage_prefixes_only() {
        let strings = |lineage: &[&str]| {
            lineage
                .iter()
                .map(|version| (*version).to_owned())
                .collect::<Vec<_>>()
        };
        assert!(known_migration_lineage(&strings(CONSOLIDATED_MIGRATIONS)));
        assert!(known_migration_lineage(&strings(LEGACY_MIGRATIONS)));
        let mut previous_consolidated = strings(CONSOLIDATED_MIGRATIONS);
        previous_consolidated.pop();
        assert!(known_migration_lineage(&previous_consolidated));
        assert!(!known_migration_lineage(&[]));
        assert!(!known_migration_lineage(&["0001".into(), "0002".into()]));
        assert!(!known_migration_lineage(&["1".into(), "3".into()]));
    }

    #[test]
    fn keep_rejects_zero() {
        assert!(backup("postgres://host/db", Path::new("target/nope"), 0).is_err());
    }
}
