# Solarisael House Substrate

The reproducible PostgreSQL, pgvector, and local-embedding backend for [Solarisael House](https://github.com/solarisael/solarisael-house).

This repository is the **optional Full House backend**. The Base House does not require it. It supplies durable memory writes, hybrid lexical/semantic retrieval storage, typed lesson stores, paper boats, Anamnesis drawers, backups, and health reporting.

## Supported path

The verified integration path is:

- Windows 10/11 running OMP
- WSL 2 with Ubuntu
- PostgreSQL 16 inside WSL
- `pgvector` and `pg_trgm`
- Python 3.11+
- Ollama with `qwen3-embedding:4b`

All substrate Python processes must run **inside WSL**. Do not invoke long writes from Windows Python across the PostgreSQL bridge; a dropped mirrored-network connection can abort a vector update after embedding work has completed.

## Install

Inside WSL:

```bash
sudo apt update
sudo apt install postgresql-16 postgresql-16-pgvector python3-venv
sudo -u postgres psql -c "CREATE ROLE solarisael LOGIN PASSWORD 'replace-me';"
sudo -u postgres createdb -O solarisael solarisael_memory
sudo -u postgres psql -d solarisael_memory -c "CREATE EXTENSION IF NOT EXISTS vector;"

cd /mnt/c/path/to/solarisael-house-substrate
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
cp .env.example .env
# edit .env; never commit it
python3 run_migrations.py
python3 health.py
```

Configure OMP on Windows with:

```text
SOLARISAEL_SUBSTRATE=C:\path\to\solarisael-house-substrate
```

The OMP adapter translates that path and launches every substrate process through `wsl.exe`.

## Health contract

`python3 health.py` exits zero only when:

- every script required by the OMP lifecycle exists;
- PostgreSQL is reachable;
- `vector` and `pg_trgm` are enabled;
- the required schema is present; and
- the embedding endpoint returns the configured vector dimension.

It emits one JSON object with `mode: "full"` or `mode: "degraded"`. Degradation is a visible state, not a silent absence of results.

Use `--skip-embedding` only to isolate database setup. A Full House acceptance check must exercise the embedder.

## Lifecycle smoke

After migration:

```bash
python3 tests/lifecycle_smoke.py
```

The smoke writes a disposable memory, verifies its chunk and vector, reads the latest paper boat, and removes the disposable row. It uses the configured database; use a throwaway database for installation testing.

## Data and backups

- `.env`, dumps, logs, and backups are ignored.
- `record_memory.py` writes and embeds through one transaction.
- Successful write helpers call `backup.sh` unless `--no-backup` is explicit.
- `restore.sh` restores custom-format dumps.
- Memory supersession and archival preserve historical rows while removing stale rows from normal retrieval authority.

## Compatibility

`compatibility.json` declares `substrateApi: 1` and schema version `1`. The House core and adapters verify these values before claiming Full House compatibility.

## Limits

- The default vector space is `qwen3-embedding:4b`, dimension 2560.
- Changing embedding models requires a full reindex; changing dimensions also requires a schema migration.
- Native Linux adapters are not yet wired. The substrate itself runs on Linux, but the public OMP adapter currently enters it through WSL.
- Fail-open retrieval may still fall back to Base House files. The adapter must report that degradation explicitly.
