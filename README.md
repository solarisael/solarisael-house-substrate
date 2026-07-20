# Solarisael House Substrate

The reproducible PostgreSQL, pgvector, and local-embedding backend for
[Solarisael House](https://github.com/solarisael/solarisael-house).  The public
repository is
[`solarisael/solarisael-house-substrate`](https://github.com/solarisael/solarisael-house-substrate).

This is the optional **Full House** backend.  **Base House** is the
file-backed operation and does not require this repository.  **Full House**
means PostgreSQL, the required schema, and the configured embedder are all
healthy.  **Degraded** is explicit: `health.py` reports `mode: "degraded"` and
exits non-zero; the House may continue with Base House files, but the adapter
must surface that the substrate is unavailable.

## Supported path and prerequisites

The supported integration path is:

- Windows 10/11 running OMP
- WSL 2 with Ubuntu; run every substrate Python command inside WSL
- PostgreSQL 16 with pgvector **0.7 or newer** and `pg_trgm`
- Python 3.11+
- Ollama with `qwen3-embedding:4b`

The external commands used below are `bash`, `sudo`, `curl`, `gpg`, `git`,
`make`, `gcc`, `psql`, `pg_dump`, `pg_restore`, `python3`, `pip`, and
`ollama`.  `wsl.exe` is required on the Windows side for the OMP bridge.  Do
not run long substrate writes from Windows Python across the PostgreSQL bridge.

Configure the Windows OMP adapter with the Windows path:

```text
SOLARISAEL_SUBSTRATE=C:\Projects\solarisael-house-substrate
```

The adapter enters the repository through WSL.  In WSL, the same repository is
normally `/mnt/c/Projects/solarisael-house-substrate`.

## Install a fresh WSL database

Run these commands in WSL.  The PGDG repository supplies PostgreSQL 16; the
source build pins pgvector to the first 0.7 release, which satisfies the
substrate's `halfvec` requirement.  `pg_trgm` comes from PostgreSQL contrib.

```bash
sudo apt update
sudo apt install -y curl ca-certificates gnupg lsb-release git build-essential

curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/postgresql.gpg
. /etc/os-release
echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] https://apt.postgresql.org/pub/repos/apt ${VERSION_CODENAME}-pgdg main" \
  | sudo tee /etc/apt/sources.list.d/pgdg.list >/dev/null
sudo apt update
sudo apt install -y postgresql-16 postgresql-server-dev-16 postgresql-contrib-16 python3-venv
sudo service postgresql start

rm -rf /tmp/pgvector
git clone --branch v0.7.0 --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector
make -C /tmp/pgvector
sudo make -C /tmp/pgvector install
sudo -u postgres psql -d postgres -c \
  "SELECT name, default_version FROM pg_available_extensions WHERE name IN ('vector', 'pg_trgm') ORDER BY name;"
```

Create the login and database.  This block is for a fresh install; choose a
real password before using it outside a local workstation.

```bash
sudo -u postgres psql -v ON_ERROR_STOP=1 <<'SQL'
CREATE ROLE solarisael LOGIN PASSWORD 'replace-me';
CREATE DATABASE solarisael_memory OWNER solarisael;
SQL
sudo -u postgres psql -d solarisael_memory -v ON_ERROR_STOP=1 -c \
  'CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;'
```

Install the Python package and create its environment:

```bash
cd /mnt/c/Projects/solarisael-house-substrate
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
# Edit .env: set the database password and any non-default host/port/model.
```

Install and start Ollama in WSL.  Keep `ollama serve` running in one terminal
and use another WSL terminal for the pull and substrate commands:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

In the second terminal:

```bash
cd /mnt/c/Projects/solarisael-house-substrate
. .venv/bin/activate
ollama pull qwen3-embedding:4b
python3 run_migrations.py
```

`run_migrations.py` records applied versions in `schema_migrations`; rerunning
it is idempotent.  The migration also creates `vector` and `pg_trgm` if they
are not already present.

## Health, embed, record, and wake

```bash
python3 health.py
```

Health prints one JSON object.  It exits zero only for `mode: "full"`:
required scripts, PostgreSQL, both extensions, schema version 1, and the
configured embedding endpoint/dimension must all pass.  For database-only
setup diagnostics, use `python3 health.py --skip-embedding`; that is not a
Full House check.  `--timeout SECONDS` changes the embedding probe timeout.

`record_memory.py` embeds inline by default.  Use `--no-embed` for a batch
write, then fill missing chunks with the embedding pass:

```bash
printf '%s\n' 'A substrate memory body.' \
  | python3 record_memory.py \
      --room room-key --type session --title 'A substrate memory' \
      --source-path db-only/example.md --body-stdin \
      --thread 'substrate / operation / install' --no-embed --no-backup
python3 embed_4b_pass.py --rooms room-key --batch 16
python3 catch_boat.py --room room-key
```

`catch_boat.py` is the wake/read helper and returns the latest paper boat for
the requested room.  `record_memory.py` accepts `--body-file` instead of
`--body-stdin`, and supports repeatable `--thread`, `--also-date`,
`--supersedes`, `--canon-touches`, `--meta-kv`, and `--meta-bool`.  The embed
pass also accepts `--rooms`, `--batch`, `--limit`, `--dry-run`, and
`--env-file`.

## Typed writers and query CLIs

The typed write helpers are:

```bash
python3 record_coding_lesson.py \
  --title 'Keep names precise' --lesson 'Name the actual boundary.' \
  --scope shared --no-backup
python3 record_project_lesson.py \
  --project solarisael-house --title 'Project rule' \
  --lesson 'Keep the project contract explicit.' --no-backup
python3 record_writing_lesson.py \
  --title 'Prefer concrete verbs' --lesson 'Use the smallest true verb.' \
  --voice general --no-backup
python3 record_audio_lesson.py \
  --title 'Check the room' --lesson 'Listen before changing gain.' \
  --stage mix --no-backup
python3 record_cabinet_entry.py --room room-key --no-backup add \
  --kind pillar --fidelity record --activation fork \
  --title 'A durable cabinet drawer' --counsel 'Keep the path visible.'
```

For any writer that accepts lesson text, `--lesson-stdin` can replace
`--lesson`.  Cabinet entries use `add` or `append-rep`; see each script's
`--help` for its remaining typed fields.

The shipped query CLIs are project and audio only:

```bash
python3 query_project_lessons.py --project solarisael-house 'contract' --limit 8
python3 query_audio_lessons.py 'gain' --stage mix --limit 12
python3 query_audio_lessons.py --spine
```

No coding-lesson or writing-lesson query CLI is shipped.  Do not document or
invoke one.

## Backups and restore

Create a custom-format dump using the configured `.env` database:

```bash
bash backup.sh
```

It writes `backups/${PGDATABASE}_TIMESTAMP.dump` and retains the newest
`KEEP` files.  A successful write helper invokes this script unless
`--no-backup` is supplied.  `--no-backup` is supported by
`record_memory.py`, `record_coding_lesson.py`, `record_project_lesson.py`,
`record_writing_lesson.py`, `record_audio_lesson.py`,
`record_cabinet_entry.py`, `import_markdown.py`, and
`import_named_entities.py`.  `backup.sh` itself has no `--no-backup` option.

The write is not rolled back when its post-write backup fails: helper scripts
print a `WARN: backup failed ...` message and return the write result.  A
direct `bash backup.sh` failure is a real command failure and should be fixed.

Restore into an **already existing** target database; `restore.sh` does not
create a database and its `--clean --if-exists` restore replaces matching
objects in that target:

```bash
sudo -u postgres createdb -O solarisael restore_target
sudo -u postgres psql -d restore_target -v ON_ERROR_STOP=1 -c \
  'CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;'
PGDATABASE=restore_target bash restore.sh backups/solarisael_memory_2026-01-01_120000.dump
PGDATABASE=restore_target python3 health.py
```

Use a target database with PostgreSQL 16 and pgvector 0.7+; do not restore
over a live target without confirming the dump and the destructive
`--clean` behavior.

## Lifecycle smoke

After migration and with Ollama serving:

```bash
python3 tests/lifecycle_smoke.py
```

The smoke uses the configured database, writes a disposable memory with inline
embedding, checks its chunk/vector and wake result, then deletes the
disposable row.  Use a throwaway database for installation testing.

## Compatibility values

Keep these values exact when integrating the public contract:

```text
substrateApi=1
coreApi=1
adapterApi=1
schemaVersion=1
```

The default embedding contract is `qwen3-embedding:4b` with dimension `2560`.
Changing the model requires a full reindex; changing dimensions also requires
a schema migration.
