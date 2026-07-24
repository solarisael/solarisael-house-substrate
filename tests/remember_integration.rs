use house_protocol::{ClusterMaintenanceResultWire, ResponseEnvelope};
use solarisael_house_substrate::{
    Config, RecallParams, RememberRequest, backup::source_migrations, recall, remember,
};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use std::str::FromStr;
use uuid::Uuid;

fn isolated_database_url() -> String {
    let url = std::env::var("SOLARISAEL_SUBSTRATE_TEST_DATABASE_URL")
        .expect("dedicated test database URL must be configured when this proof is run");
    let lower = url.to_ascii_lowercase();
    assert!(
        !lower.contains("solarisael_memory"),
        "refusing the live/default database"
    );
    assert!(
        !lower.contains("solarisael-house"),
        "refusing a production-looking database"
    );
    url
}

fn migration_database_scope() -> (String, Option<String>) {
    let Ok(schema) = std::env::var("SOLARISAEL_SUBSTRATE_TEST_SCHEMA") else {
        return (isolated_database_url(), None);
    };
    assert!(schema.starts_with("solarisael_tuner_test_"));
    assert!(
        schema
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    );
    let url = std::env::var("SOLARISAEL_SUBSTRATE_TEST_DATABASE_URL")
        .expect("the schema proof requires a PostgreSQL database URL");
    (url, Some(schema))
}

#[tokio::test]
#[ignore = "requires SOLARISAEL_SUBSTRATE_TEST_DATABASE_URL and an isolated PostgreSQL database"]
async fn isolated_database_guard() {
    let url = isolated_database_url();
    let options = PgConnectOptions::from_str(&url).expect("dedicated test URL must be valid");
    let pool = PgPoolOptions::new()
        .max_connections(2)
        .connect_with(options)
        .await
        .expect("isolated database must be reachable");
    sqlx::query("SELECT 1")
        .execute(&pool)
        .await
        .expect("isolated database health check");

    let cfg = Config {
        database_url: url,
        embed_url: None,
        embed_model: "disabled".into(),
        embed_dimension: 2048,
        embed_required: false,
        test_embedding_disabled: true,
    };
    let source_path = format!("isolated-test/{}", Uuid::new_v4());
    let body = "This mutation proves the dedicated PostgreSQL authority path.";
    let receipt = remember(
        &pool,
        &cfg,
        RememberRequest {
            room: "isolated-test".into(),
            kind: "memory".into(),
            title: "isolated integration proof".into(),
            body: body.into(),
            lesson: None,
            source_path: Some(source_path.clone()),
            source_memory_path: None,
            threads: vec!["integration".into()],
            supersedes: vec![],
            shape: None,
            voice: None,
            scope: None,
            project: None,
            proof_pattern: None,
            trigger_context: None,
            tags: vec![],
            backup: false,
        },
    )
    .await
    .expect("remember mutation must commit");
    assert_eq!(receipt.authority, "postgres");
    assert!(receipt.durable);
    let count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM memories WHERE room=$1 AND source_path=$2")
            .bind("isolated-test")
            .bind(&source_path)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 1);
    let lexical_chunks: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM memory_chunks WHERE memory_id=$1 AND body_embedding IS NULL",
    )
    .bind(receipt.memory_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(lexical_chunks > 0);
    let recalled = recall(
        &pool,
        &cfg,
        RecallParams {
            room: "isolated-test".into(),
            query: body.into(),
            semantic_top_k: 1,
            semantic_min_similarity: 0.5,
            content_top_k: 8,
            content_min_similarity: 0.3,
        },
    )
    .await
    .expect("lexical recall must succeed with embeddings disabled");
    assert!(recalled.found);
    assert!(
        recalled.content_chunks.iter().any(|chunk| {
            chunk.get("source_path").and_then(serde_json::Value::as_str)
                == Some(source_path.as_str())
                && chunk.get("body").and_then(serde_json::Value::as_str) == Some(body)
        }),
        "lexical recall must return the exact written body for the written source path"
    );
    sqlx::query("DELETE FROM memories WHERE room=$1 AND source_path=$2")
        .bind("isolated-test")
        .bind(source_path)
        .execute(&pool)
        .await
        .unwrap();
    pool.close().await;
}

#[tokio::test]
#[ignore = "requires an isolated PostgreSQL database or schema"]
async fn migrations_reapply_without_clearing_current_embeddings() {
    let (url, schema) = migration_database_scope();
    let options = PgConnectOptions::from_str(&url).expect("test database URL must be valid");
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await
        .expect("test database must be reachable");
    if let Some(schema) = &schema {
        sqlx::query(&format!("CREATE SCHEMA {schema}"))
            .execute(&pool)
            .await
            .expect("isolated schema must create");
        sqlx::query(&format!("SET search_path TO {schema}, public"))
            .execute(&pool)
            .await
            .expect("isolated schema must become active");
    }
    let initial = include_str!("../migrations/0001_initial.sql");
    let nemotron = include_str!("../migrations/0002_nemotron_2048.sql");
    sqlx::raw_sql(initial)
        .execute(&pool)
        .await
        .expect("initial migration must apply");
    sqlx::raw_sql(nemotron)
        .execute(&pool)
        .await
        .expect("Nemotron migration must apply");

    let source_path = format!("migration-reapply/{}", Uuid::new_v4());
    let memory_id: i64 = sqlx::query_scalar(
        "INSERT INTO memories (room,type,title,source_path,body) VALUES ('isolated-test','memory','migration reapply',$1,'sentinel') RETURNING id",
    )
    .bind(&source_path)
    .fetch_one(&pool)
    .await
    .expect("sentinel memory must insert");
    let vector = format!("[{}]", vec!["0"; 2048].join(","));
    sqlx::query(
        "INSERT INTO memory_chunks (memory_id,chunk_index,body,char_start,char_end,body_embedding,embedded_at) VALUES ($1,0,'sentinel',0,8,$2::vector,NOW())",
    )
    .bind(memory_id)
    .bind(vector)
    .execute(&pool)
    .await
    .expect("sentinel embedding must insert");

    sqlx::raw_sql(initial)
        .execute(&pool)
        .await
        .expect("initial migration must reapply");
    sqlx::raw_sql(nemotron)
        .execute(&pool)
        .await
        .expect("Nemotron migration must reapply");
    let embedded: bool = sqlx::query_scalar(
        "SELECT body_embedding IS NOT NULL FROM memory_chunks WHERE memory_id=$1",
    )
    .bind(memory_id)
    .fetch_one(&pool)
    .await
    .expect("sentinel embedding must remain queryable");
    assert!(embedded);
    let versions: i64 =
        sqlx::query_scalar("SELECT count(*) FROM schema_migrations WHERE version IN (1,2)")
            .fetch_one(&pool)
            .await
            .expect("migration versions must remain queryable");
    assert_eq!(versions, 2);

    sqlx::query("DELETE FROM memories WHERE id=$1")
        .bind(memory_id)
        .execute(&pool)
        .await
        .expect("sentinel cleanup must succeed");
    if let Some(schema) = &schema {
        sqlx::query("SET search_path TO public")
            .execute(&pool)
            .await
            .expect("public schema must become active for cleanup");
        sqlx::query(&format!("DROP SCHEMA {schema} CASCADE"))
            .execute(&pool)
            .await
            .expect("isolated schema cleanup must succeed");
    }
    pool.close().await;
}

#[tokio::test]
#[ignore = "requires an isolated PostgreSQL schema"]
async fn source_migrations_accepts_text_version_columns() {
    let (url, schema) = migration_database_scope();
    let schema = schema.expect("text-version proof requires SOLARISAEL_SUBSTRATE_TEST_SCHEMA");
    let options = PgConnectOptions::from_str(&url).expect("test database URL must be valid");
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await
        .expect("test database must be reachable");
    sqlx::query(&format!("CREATE SCHEMA {schema}"))
        .execute(&pool)
        .await
        .expect("isolated schema must create");
    sqlx::query(&format!("SET search_path TO {schema}, public"))
        .execute(&pool)
        .await
        .expect("isolated schema must become active");
    sqlx::query("CREATE TABLE schema_migrations (version TEXT PRIMARY KEY)")
        .execute(&pool)
        .await
        .expect("text migration table must create");
    sqlx::query("INSERT INTO schema_migrations(version) VALUES ('1'), ('2')")
        .execute(&pool)
        .await
        .expect("text migration versions must insert");

    assert_eq!(source_migrations(&pool).await.unwrap(), ["1", "2"]);

    sqlx::query("SET search_path TO public")
        .execute(&pool)
        .await
        .expect("public schema must become active for cleanup");
    sqlx::query(&format!("DROP SCHEMA {schema} CASCADE"))
        .execute(&pool)
        .await
        .expect("isolated schema cleanup must succeed");
    pool.close().await;
}

#[test]
fn cluster_wire_fixture_deserializes_through_shared_protocol() {
    let fixture = r#"{
        "protocol": 1,
        "id": "cluster-1",
        "result": {
            "ok": true,
            "operation": "check",
            "dryRun": false,
            "rebuilt": false,
            "status": {
                "stale": true,
                "reason": "never_built",
                "staleness": {
                    "builtAt": null,
                    "clusters": 0,
                    "chunksTotal": 3,
                    "chunksSinceBuild": 3,
                    "fractionUnseen": 1.0
                }
            },
            "clusters": [{"clusterId":1,"label":"cluster","memberCount":3,"accepted":false}]
        }
    }"#;
    let response: ResponseEnvelope<ClusterMaintenanceResultWire> =
        serde_json::from_str(fixture).expect("substrate fixture must match shared wire types");
    let house_protocol::ResponsePayload::Result { result } = response.payload else {
        panic!("expected result payload");
    };
    assert_eq!(result.status.reason, "never_built");
    assert_eq!(result.status.staleness.chunks_since_build, 3);
}

#[test]
fn spherical_kmeans_k1_uses_normalized_mean_centroid() {
    let groups = solarisael_house_substrate::spherical_kmeans(
        &[(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0])],
        1,
    );
    assert_eq!(groups.len(), 1);
    let centroid = &groups[0].0;
    let expected = 2.0_f32.sqrt().recip();
    assert!((centroid[0] - expected).abs() < 1e-6);
    assert!((centroid[1] - expected).abs() < 1e-6);
    assert_eq!(groups[0].1.len(), 2);
}
