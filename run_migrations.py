#!/usr/bin/env python3
"""Apply ordered Solarisael House substrate migrations exactly once."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import psycopg2


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def connect(database: str | None = None):
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "127.0.0.1"),
        port=os.environ.get("PGPORT", "5432"),
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        dbname=database or os.environ["PGDATABASE"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database")
    parser.add_argument("--env-file", default=str(Path(__file__).with_name(".env")))
    args = parser.parse_args()
    load_dotenv(Path(args.env_file))

    migrations = sorted(Path(__file__).with_name("migrations").glob("[0-9][0-9][0-9][0-9]_*.sql"))
    if not migrations:
        raise SystemExit("no migrations found")

    with connect(args.database) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())")
            cur.execute("SELECT version FROM schema_migrations")
            applied = {row[0] for row in cur.fetchall()}
        for migration in migrations:
            version = int(migration.name[:4])
            if version in applied:
                print(f"skip {migration.name}")
                continue
            sql = migration.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(sql)
            print(f"applied {migration.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
