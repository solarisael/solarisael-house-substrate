#!/usr/bin/env python3
"""Return one explicit health verdict for the full House substrate."""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

try:
    import psycopg2
except Exception as exc:  # pragma: no cover - exercised by installations
    psycopg2 = None
    PSYCOPG_ERROR = str(exc)
else:
    PSYCOPG_ERROR = None

REQUIRED_SCRIPTS = (
    "record_memory.py",
    "catch_boat.py",
    "record_coding_lesson.py",
    "record_project_lesson.py",
    "record_writing_lesson.py",
    "record_audio_lesson.py",
    "record_cabinet_entry.py",
)
REQUIRED_TABLES = (
    "memories",
    "memory_threads",
    "memory_chunks",
    "named_entities",
    "coding_lessons",
    "project_lessons",
    "writing_lessons",
    "audio_lessons",
    "anamnesis",
    "anamnesis_reps",
)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def probe_embedding(timeout: float) -> dict:
    url = os.environ.get("SOLARISAEL_EMBED_URL", "http://127.0.0.1:11435/api/embed")
    model = os.environ.get("SOLARISAEL_EMBED_MODEL", "hf.co/zenmagnets/Nemotron-3-Embed-1B-Q4_K_M-GGUF:latest")
    raw_expected = os.environ.get("SOLARISAEL_EMBED_DIMENSION", "2048")
    try:
        expected = int(raw_expected)
    except ValueError:
        return {
            "ok": False,
            "url": url,
            "model": model,
            "error": f"SOLARISAEL_EMBED_DIMENSION must be an integer, got {raw_expected!r}",
        }
    payload = json.dumps({"model": model, "input": "passage: solarisael house health"}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        vectors = body.get("embeddings") or [item.get("embedding") for item in body.get("data", [])]
        vector = vectors[0] if vectors else None
        dimension = len(vector) if isinstance(vector, list) else None
        return {
            "ok": dimension == expected,
            "url": url,
            "model": model,
            "dimension": dimension,
            "expectedDimension": expected,
            **({"error": f"embedding dimension is {dimension}, expected {expected}"} if dimension != expected else {}),
        }
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {"ok": False, "url": url, "model": model, "expectedDimension": expected, "error": str(exc)}


def probe_database() -> dict:
    if psycopg2 is None:
        return {"ok": False, "error": f"psycopg2 unavailable: {PSYCOPG_ERROR}"}
    try:
        conn = psycopg2.connect(
            host=os.environ.get("PGHOST", "127.0.0.1"),
            port=os.environ.get("PGPORT", "5432"),
            user=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
            dbname=os.environ["PGDATABASE"],
            connect_timeout=3,
        )
        with conn, conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user")
            database, user = cur.fetchone()
            cur.execute("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm') ORDER BY extname")
            extensions = [row[0] for row in cur.fetchall()]
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema() AND table_name = ANY(%s)", (list(REQUIRED_TABLES),))
            tables = {row[0] for row in cur.fetchall()}
            cur.execute("SELECT coalesce(max((substring(version::text from '^[0-9]+'))::integer), 0) FROM schema_migrations")
            schema_version = cur.fetchone()[0]
        conn.close()
        missing = sorted(set(REQUIRED_TABLES) - tables)
        ok = not missing and {"vector", "pg_trgm"}.issubset(extensions) and schema_version >= 1
        return {
            "ok": ok,
            "database": database,
            "user": user,
            "schemaVersion": schema_version,
            "extensions": extensions,
            "missingTables": missing,
            **({"error": "database schema is incomplete"} if not ok else {}),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(Path(__file__).with_name(".env")))
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    load_dotenv(Path(args.env_file))

    missing_scripts = [name for name in REQUIRED_SCRIPTS if not (root / name).is_file()]
    scripts = {"ok": not missing_scripts, "missing": missing_scripts}
    database = probe_database()
    embedding = {"ok": None, "skipped": True} if args.skip_embedding else probe_embedding(args.timeout)
    reasons = []
    if not scripts["ok"]:
        reasons.append("required substrate scripts are missing")
    if not database["ok"]:
        reasons.append("PostgreSQL substrate is unavailable or incomplete")
    if embedding.get("ok") is False:
        reasons.append("embedding service is unavailable or incompatible")
    mode = "full" if not reasons else "degraded"
    result = {"ok": not reasons, "mode": mode, "substrateApi": 1, "scripts": scripts, "database": database, "embedding": embedding, "degradedReasons": reasons}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
