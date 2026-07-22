use solarisael_house_substrate::{process_request, Config, RememberRequest};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use std::str::FromStr;
use uuid::Uuid;

#[tokio::test]
#[ignore = "requires SOLARISAEL_SUBSTRATE_TEST_DATABASE_URL and an isolated PostgreSQL database"]
async fn isolated_database_guard() {
    let url = std::env::var("SOLARISAEL_SUBSTRATE_TEST_DATABASE_URL")
        .expect("dedicated test database URL must be configured when this proof is run");
    let lower = url.to_ascii_lowercase();
    assert!(!lower.contains("solarisael_memory"), "refusing the live/default database");
    assert!(!lower.contains("solarisael-house"), "refusing a production-looking database");
    let options = PgConnectOptions::from_str(&url).expect("dedicated test URL must be valid");
    let pool = PgPoolOptions::new().max_connections(2).connect_with(options).await
        .expect("isolated database must be reachable");
    sqlx::query("SELECT 1").execute(&pool).await.expect("isolated database health check");

    let cfg = Config {
        database_url: url,
        embed_url: None,
        embed_model: "disabled".into(),
        embed_dimension: 2048,
        embed_required: false,
        test_embedding_disabled: true,
    };
    let source_path = format!("isolated-test/{}", Uuid::new_v4());
    let receipt = process_request(&pool, &cfg, RememberRequest {
        room: "isolated-test".into(),
        kind: "memory".into(),
        title: "isolated integration proof".into(),
        body: "This mutation proves the dedicated PostgreSQL authority path.".into(),
        source_path: Some(source_path.clone()),
        threads: vec!["integration".into()],
        supersedes: vec![],
        backup: false,
    }).await.expect("remember mutation must commit");
    assert_eq!(receipt.authority, "postgres");
    assert!(receipt.durable);
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM memories WHERE room=$1 AND source_path=$2")
        .bind("isolated-test").bind(&source_path).fetch_one(&pool).await.unwrap();
    assert_eq!(count, 1);
    sqlx::query("DELETE FROM memories WHERE room=$1 AND source_path=$2")
        .bind("isolated-test").bind(source_path).execute(&pool).await.unwrap();
    pool.close().await;
}
