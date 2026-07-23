use house_protocol::{ClusterMaintenanceResultWire, ResponseEnvelope};
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
