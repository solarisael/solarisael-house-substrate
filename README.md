# The Athanor AKASHA Substrate

**the backend that makes House memory stick — start at [The Athanor](https://github.com/solarisael/the-athanor).**
This repository is the optional **AKASHA** backend for [The Athanor](https://github.com/solarisael/the-athanor): Rust, PostgreSQL, pgvector, and local embeddings.
The public source is [`solarisael/solarisael-house-substrate`](https://github.com/solarisael/solarisael-house-substrate).

This is the optional **AKASHA** backend. **Vault** is file-backed and
does not require this repository. **AKASHA** means the release Rust
executable, PostgreSQL schema, and configured embedder are all healthy.
**Degraded** is explicit: `health.py` reports `mode: "degraded"` and exits
non-zero; the House may continue with Vault files, but the adapter must
surface that the substrate is unavailable.

## Public distribution contract

This repository is the complete public AKASHA distribution. It includes the Rust runtime, migrations, `.env.example`, Python support environment, health probe, lifecycle smoke, import and maintenance tools, and Rust backup/restore authority. An installing AI must not ask the maintainer for a private substrate bundle or infer that these artifacts are absent from the Vault archive. Clone this repository beside the House core, supply local credentials, build the release executable, adapt host commands when necessary, and verify the mounted OMP lifecycle.

Vault and AKASHA are deployment modes, not a required trial sequence. Choose from the operator's requested continuity and the host's verified capabilities. Additional setup work is not, by itself, a reason to recommend Vault.

Current AKASHA memory recall combines native BM25F, PostgreSQL FTS, `pg_trgm`,
structured date/thread/canon rails, pgvector semantic retrieval, and a controlled
Nemotron-to-BM25F lexical bridge. Schema 12 maintains a room-scoped vocabulary
from authoritative named entities, active threads, and lesson metadata, adds
the unified design-lesson family, and introduces the design-system document
catalogue. Recall
selects at most three concepts, caps normalized expansion terms, keeps exact
BM25F precedence, attributes the separate lane, and fails open when vocabulary
vectors are absent or stale. No additional model, service, or extension is used.

## Supported path and prerequisites

The supported integration path is:

- Windows 10/11 running OMP
- stable Rust with the MSVC toolchain on Windows
- WSL 2 with Ubuntu for PostgreSQL, migrations, health, and Python support tools
- PostgreSQL 16 with pgvector **0.7 or newer** and `pg_trgm`
- Python 3.11+
- Ollama with `hf.co/zenmagnets/Nemotron-3-Embed-1B-Q4_K_M-GGUF:latest`

The mounted OMP path is:

```text
OMP TypeScript adapter -> Windows solarisael-house-substrate.exe -> WSL PostgreSQL and Ollama
```

The external commands used below are `cargo`, `bash`, `sudo`, `curl`, `gpg`,
`git`, `make`, `gcc`, `psql`, `pg_dump`, `pg_restore`, `python3`, `pip`, and
`ollama`. `wsl.exe` is required on the Windows side. Python tools remain the
supported migration, health, import, and maintenance surface; ordinary mounted
memory calls go through the long-lived Rust process.

Configure the Windows OMP process with absolute Windows paths:

```text
SOLARISAEL_SUBSTRATE=C:\Projects\solarisael-house-substrate
SOLARISAEL_HOUSE_RUST=C:\Projects\solarisael-house-substrate\target\release\solarisael-house-substrate.exe
SOLARISAEL_PG_WSL=1
```

Restart OMP after setting them. The Rust executable reads `.env` from the
substrate repository at startup.

## Install a fresh WSL database

Run these commands in WSL.  The PGDG repository supplies PostgreSQL 16; the
source build pins pgvector to the first 0.7 release, which satisfies the
substrate's current vector requirements. `pg_trgm` comes from PostgreSQL contrib.

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

Build the authoritative Windows runtime from PowerShell or another Windows
terminal:

```text
cd C:\Projects\solarisael-house-substrate
cargo build --release
```

The adapter must point `SOLARISAEL_HOUSE_RUST` at:

```text
C:\Projects\solarisael-house-substrate\target\release\solarisael-house-substrate.exe
```

### Upgrade a running local installation

Use the canonical PowerShell 7 deployment path instead of building over a
locked live executable:

```text
cd C:\Projects\solarisael-house-substrate
pwsh -File .\deploy-local.ps1
```

The script tests the Athanor core/protocol and substrate, builds a staged
release, takes a PostgreSQL backup, stops only substrate workers running from
the exact configured `SOLARISAEL_HOUSE_RUST` path, replaces that executable,
applies ordered migrations, and requires an AKASHA health result (`mode: "full"` for compatibility). It restores
the prior executable if migration fails.

`-SkipTests` and `-SkipBackup` exist for diagnosed recovery only; the safe
default performs both. Restart OMP once after success so its long-lived Rust
transport and TypeScript tool schemas reload.

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
ollama pull hf.co/zenmagnets/Nemotron-3-Embed-1B-Q4_K_M-GGUF:latest
python3 run_migrations.py
```

`run_migrations.py` records applied versions in `schema_migrations`; rerunning
it is idempotent.  The migration also creates `vector` and `pg_trgm` if they
are not already present.

## Verify the Rust runtime

`health.py` verifies dependencies and schema. It does not prove that OMP is
using the Rust transport. After health passes and OMP is restarted, use the
registered `remember` and `recall` tools. A successful recall must report
`source: rust-postgres`. Leave the same mounted process idle for at least 75
seconds, then repeat the write and recall to exercise process lifetime and WSL
keepalive behavior.

The executable also owns guarded backup and restore:

```text
solarisael-house-substrate.exe backup --output-dir C:\path\to\backups --keep 14
solarisael-house-substrate.exe restore --manifest C:\path\to\manifest.json --confirm-database solarisael_memory
```

Restore is destructive. Inspect the manifest and target database before running
the second command.

## Health, embed, record, and wake

```bash
python3 health.py
```

Health prints one JSON object. It exits zero only for `mode: "full"`: required
scripts, PostgreSQL, both extensions, schema version 12, and the configured
embedding endpoint/dimension must all pass. For database-only setup diagnostics,
use `python3 health.py --skip-embedding`; that is not an AKASHA check.
`--timeout SECONDS` changes the embedding probe timeout.

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

`--continues` accepts a repeatable JSON object with `thread` and
`previousMemoryId`. The thread must also be present in `--thread`; continuation
records chronology and never replaces `--supersedes` authority semantics.

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
python3 record_design_lesson.py \
  --title 'Keep focus visible' --lesson 'Use the focus token for every control.' \
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

The design-system catalogue uses `design-docs.py` to query and `design-doc-write.py` to supersede rather than mutate entries.

## Starter coding lessons

This repository includes a privacy-safe library with 117 reusable coding lessons and no project lessons.

Preview the pack before import:

```bash
python3 import_coding_lessons.py --dry-run
```

Import the pack after AKASHA health passes:

```bash
python3 import_coding_lessons.py
```

The default import preserves existing lessons with the same scope, project, and title.

Use `--update-existing` only when the operator chooses to replace those lesson fields with the bundled versions.

The pack preserves 12 formal lesson negations by title. The importer resolves those links after all rows exist.

The importer runs one transaction. A successful write runs the normal backup unless `--no-backup` is present.

Use `--pack PATH` to validate and import another pack that follows schema version 1.


The shipped query CLIs are project and audio only:

```bash
python3 query_project_lessons.py --project solarisael-house 'contract' --limit 8
python3 query_audio_lessons.py 'gain' --stage mix --limit 12
python3 query_audio_lessons.py --spine
```

No coding-lesson, writing-lesson, or design-lesson query CLI is shipped.  Do not
document or invoke one.

## Backups and restore

Create a custom-format dump using the configured `.env` database:

```bash
bash backup.sh
```

It writes `backups/${PGDATABASE}_TIMESTAMP.dump` and retains the newest
`KEEP` files.  A successful write helper invokes this script unless
`--no-backup` is supplied.  `--no-backup` is supported by
`record_memory.py`, `record_coding_lesson.py`, `record_project_lesson.py`,
`record_writing_lesson.py`, `record_design_lesson.py`, `record_audio_lesson.py`,
`record_cabinet_entry.py`, `import_markdown.py`, `import_named_entities.py`,
and `import_coding_lessons.py`. `backup.sh` itself has no `--no-backup` option.

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
schemaVersion=12
```

The default embedding contract is `hf.co/zenmagnets/Nemotron-3-Embed-1B-Q4_K_M-GGUF:latest`
with dimension `2048`. Changing the model requires a full reindex; changing
dimensions also requires a schema migration.
