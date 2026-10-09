# MIRA — Local Setup from Scratch

> **Runtime note:** this guide's model-download instructions describe an older version of
> MIRA. The current backend uses the authenticated remote embedding API in
> `backend/app/services/matching/embeddings.py`. Set `MIRA_MODEL_SERVER_URL` and
> `MIRA_API_KEY` in `backend/.env`, and run the service described in
> [DEPLOYMENT.md](DEPLOYMENT.md) before uploading or matching. The local-model sections
> below have not yet been rewritten for this runtime.

**Audience:** an evaluator starting on a clean machine. Nothing is assumed to be installed.
**Platform:** Windows (PowerShell) first; macOS/Linux notes are given where they differ.
**Time:** about 30–45 minutes, most of it downloads. Add 5–10 minutes for the first model download.

> **Important expectation.** On the very first upload, MIRA fetches a 753 MB embedding
> model from Hugging Face and verifies it. The UI will sit on "Uploading…" for roughly
> 30–60 seconds (fast connection) while that happens. This is normal, it happens once
> per machine, and every later upload is fast. There is no GPU requirement.

---

## 0. What you need before you start

| Software | Version | Check with | Get it from |
|---|---|---|---|
| Git | any | `git --version` | https://git-scm.com/downloads |
| Python | 3.11 – 3.13 (3.13 verified) | `python --version` | https://www.python.org/downloads/ — on Windows tick **Add python.exe to PATH** |
| Node.js | **20.19+ or 22.12+** (the frontend uses Vite 8) | `node --version` | https://nodejs.org |
| PostgreSQL | 14 or newer (16/17 fine) | `psql --version` | https://www.postgresql.org/download/ — **remember the password you set for `postgres`** |
| Docker Desktop | any recent | `docker --version` | https://www.docker.com/products/docker-desktop/ — needed for the Milvus vector store |

Also needed:

- **~6 GB free disk space** (about 3 GB of Python packages, ~1.5 GB for the model, the rest for Docker images)
- **Internet access** for the package installs and the model download
- **Access to the repository** — it is private. Ask for a collaborator invite, or use the ZIP you were given.

Quick pre-flight (all four should print a version, no errors):

```powershell
git --version
python --version
node --version
psql --version
docker --version
```

---

## 1. Get the code

```powershell
cd "$env:USERPROFILE\Downloads"
git clone https://github.com/AshIndian-Coder/mira.git mira
cd mira
```

If you were given a ZIP instead, unzip it and `cd` into the folder — the rest of this guide is identical.

**Check it is the right code:**

```powershell
dir backend\mira_full_schema.sql, backend\app\main.py, frontend\package.json, data\sample, Original_company_records_no_synthetic.csv
```

All five must appear. If `mira_full_schema.sql` is missing, you have the wrong folder.

---

## 2. Database (PostgreSQL)

MIRA uses PostgreSQL as its system of record. The application does **not** create tables
itself — the schema must be applied once, before the first start.

### 2.1 Create the database

```powershell
psql -U postgres -c "CREATE DATABASE mira;"
```

It will prompt for the `postgres` password. Expected output: `CREATE DATABASE`.

> If you get `psql: command not found`, PostgreSQL's `bin` folder is not on your PATH.
> Either add it, or use the full path, e.g.
> `"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -c "CREATE DATABASE mira;"`.

### 2.2 Apply the schema

Run this **from the repository root** (the folder containing `backend`):

```powershell
psql -U postgres -d mira -f backend/mira_full_schema.sql
```

Expected: a series of `CREATE TABLE` / `CREATE INDEX` lines and **no `ERROR`**.

### 2.3 Verify

```powershell
psql -U postgres -d mira -c "\dt"
```

You should see **10 tables**, including `material_embedding_cache`, along with
`audit_logs`, `cnmc`, `cpses`, `feedback`, `mappings`, `match_suggestions`, `materials`,
`upload_batches`, and `users`.

> No `CREATE EXTENSION` step is needed. MIRA stores material vectors as JSONB for reuse
> during scoring. Milvus remains an optional source of extra nearest-neighbour candidates (§3).

---

## 3. Milvus (vector search)

MIRA runs two candidate sources: deterministic blocking **and** a semantic nearest-neighbour
search over embeddings stored in Milvus. Milvus is what lets the system find pairs that
share no keyword, so it is a normal part of the stack — start it before you upload anything.

### 3.1 Start the containers

Docker Desktop must be running (wait for the whale icon to stop animating). Then, **from the `backend` folder**:

```powershell
cd backend
docker compose -f docker-compose.milvus.yml up -d
```

First run downloads three images (a few hundred MB). Then check:

```powershell
docker ps --format "{{.Names}}  {{.Status}}"
```

All three must be listed as `Up` (Milvus itself may show `(healthy)` after ~30 seconds):

| Container | Purpose |
|---|---|
| `milvus-etcd` | metadata store |
| `milvus-minio` | object storage for vectors |
| `milvus-standalone` | the vector database (host port **19530**) |

If one is missing, wait 20–30 seconds and check again — Milvus takes a moment to become healthy.

### 3.2 Create the collection (once)

```powershell
python create_milvus_collection.py
```

Expected: `Created collection 'material_embeddings' with dynamic dim=1024.`

This is safe to re-run — it skips creation if the collection already exists. **Do not**
pass `--recreate` later: that drops the collection and discards any vectors you have already uploaded.

---

## 4. Python environment

### 4.1 Create and activate a virtual environment

From the **`backend`** folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

> If PowerShell refuses with *"running scripts is disabled on this system"*, run this once
> in the same window and try again:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

macOS/Linux alternative:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4.2 Install the packages

Install the CPU build of PyTorch **first** — the `+cpu` wheel lives on PyTorch's own server, not PyPI:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then everything else (5–15 minutes, one time only):

```powershell
pip install -r requirements.txt
```

Finally, the two packages the INT8 embedding model needs to load:

```powershell
pip install bitsandbytes accelerate
```

> If `pip install -r requirements.txt` stops with `No matching distribution found for torch==2.14.0+cpu`,
> it means torch was not installed in the previous step — install it, then re-run this command.

### 4.3 Verify the environment

```powershell
python -c "import fastapi, sqlalchemy, torch, sentence_transformers, pymilvus, bitsandbytes; print('dependencies OK')"
```

Must print `dependencies OK`.

---

## 5. Configuration

MIRA reads an optional `backend/.env` file. The only value you may need to change is the database URL.

**Default:** `postgresql://postgres:postgres@localhost:5432/mira`

- If your `postgres` password **is** `postgres` → nothing to do; skip to §6.
- If it is something else → create the file `backend\.env` with:

```
DATABASE_URL=postgresql://postgres:YOUR_ACTUAL_PASSWORD@localhost:5432/mira
```

**Every switch you can set (all optional):**

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/mira` | database connection |
| `MILVUS_ENABLED` | `true` | `false` disables vector search (the app still runs, but finds fewer candidate pairs) |
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | where Milvus runs |
| `MILVUS_TOP_K` | `50` | vector neighbours requested per material |
| `MODEL_AUTO_DOWNLOAD` | `true` | `false` = never download; fail loudly if the model is absent |
| `MODEL_HUB_ID` | `AshIndian/Mira.ai` | which Hugging Face checkpoint to fetch |
| `MIRA_MODELS_DIR` | *(unset)* | extra folder to look in / install into |
| `MIRA_MODEL_PATH`, `MIRA_QWEN_MODEL_PATH` | *(unset)* | explicit local model folder |
| `MIRA_EMBEDDING_MODEL` | *(unset)* | force a specific model path (harnesses only) |
| `MIRA_LLM_PROVIDER` | `none` | optional LLM extraction: `none`, `ollama`, `groq` |
| `MIRA_HIGH_CONFIDENCE_SCORE` / `MIRA_DIFFERENT_SCORE` | `0.85` / `0.45` | decision thresholds (calibrated — leave alone) |

Note the naming: application settings are plain (`DATABASE_URL`, `MILVUS_*`, `MODEL_*`);
the `MIRA_*` names above are read directly by specific modules. Never commit `.env` — it is
gitignored, and it holds a password.

---

## 6. Start the backend

From the `backend` folder, with `(.venv)` active:

```powershell
python -m uvicorn app.main:app --reload --reload-dir app --port 8000
```

**Expected output (abridged):**

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process ...
```

On the first start the seeding step also creates the baseline CPSEs and demo users. Leave
this window open; call it **Window A**.

**Verify:**

```powershell
# in a second window
curl.exe http://127.0.0.1:8000/health
```

Must return `{"status":"ok","service":"mira-backend"}`.

> Start commands from the `backend` folder. The app reads `.env` relative to the current
> directory and imports its own `app.*` package, so running from elsewhere will fail.

---

## 7. Start the frontend

Open a **new** window (**Window B**):

```powershell
cd "$env:USERPROFILE\Downloads\mira\frontend"
npm install
npm run dev
```

`npm install` takes a few minutes the first time. `npm run dev` prints:

```
  VITE v8.x.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/
```

The dev server proxies `/api` and `/health` to the backend on port 8000, so there is nothing
else to configure.

**Open http://localhost:5173 in a browser.**

### Demo accounts (seeded automatically at first backend start)

| Email | Password | Role |
|---|---|---|
| `admin@mira.gov.in` | `Admin@123` | Administrator (all permissions) |
| `steward@mira.gov.in` | `Steward@123` | Data Steward (CPCL) |
| `reviewer@mira.gov.in` | `Reviewer@123` | Reviewer (IOCL) |
| `auditor@mira.gov.in` | `Auditor@123` | Compliance Auditor (read-only) |

Log in as **admin** to walk through the whole flow.

---

## 8. First run — the walkthrough

The five seeded CPSEs are **IOCL, BPCL, CPCL, SAIL, NTPC**. Material rows must carry one of
those names (or another recognised CPSE) — unrecognised values are grouped under
`CPSE_GENERIC`, and since MIRA only compares *across* CPSEs, a file with placeholder names
produces **zero** candidates. That is deliberate: provenance is never guessed.

### 8.1 Save the demo file

Create `demo_materials.csv` in the repository root and paste this in:

```csv
cpse,material_code,description,category,unit,manufacturer,material_grade
NTPC,NTPC-V001,GATE VALVE 50MM CL150 SS316 FLANGED,Valve,NOS,Audco,SS316
NTPC,NTPC-V002,BALL VALVE 25MM CL300 CS BODY,Valve,NOS,Audco,CS
NTPC,NTPC-P001,MS PIPE 100MM DIA 6 MTR SCH40,Pipe,MTR,SAIL,MS
NTPC,NTPC-F001,HEX HEAD BOLT M16 X 60 MM SS304 GRADE A2-70,Fastener,NOS,TVS,A2-70
IOCL,IOCL-V001,GATE VALVE 50 MM CLASS 150 SS 316 FLGD,Valve,NOS,L&T,SS316
IOCL,IOCL-V002,BALL VALVE 25 MM CL 300 CARBON STEEL,Valve,NOS,L&T,CS
IOCL,IOCL-P001,M S PIPE 100 MM DIA 6 METER SCH 40,Pipe,MTR,SAIL,MS
IOCL,IOCL-F001,"HEXAGONAL HEAD BOLT, SS304, A2-70, M16x60",Fastener,NOS,TVS,A2-70
BPCL,BPCL-V001,GATE VALVE 50MM CL150 SS316 FLANGED,Valve,NOS,Audco,SS316
BPCL,BPCL-P001,SEAMLESS PIPE 100MM NB SCH40 CS,Pipe,MTR,Jindal,CS
BPCL,BPCL-F001,HEX HEAD BOLT M16 X 60 MM SS304 GRADE A2-70,Fastener,NOS,TVS,A2-70
CPCL,CPCL-P001,MS PIPE 100MM DIA 6 MTR SCH40,Pipe,MTR,SAIL,MS
```

**What this file is designed to show:** the same three items (a 50 mm SS316 gate valve, a
100 mm MS pipe, an M16 A2-70 bolt) written in CPSE-specific styles across four CPSEs — plus
two genuinely different items as distractors. Nothing in it is synthetic filler; the
descriptions follow the same TENDER-style conventions as the real records.

### 8.2 Upload

Sidebar → **Materials** → **Upload File** → choose `demo_materials.csv`.

Expected: toast **"Uploaded 12 records"**, and the table shows the rows with
**NTPC / IOCL / BPCL / CPCL** in the CPSE column.

> **The first upload waits for the model.** Watch Window A: you should see
> `MIRA embedding model not found locally -- downloading 'AshIndian/Mira.ai' … (~750 MB, one time)`
> followed by `Embedding model downloaded and verified: …\models\Mira.ai (dim=1024)`.
> The upload completes afterwards. See §10 if it fails.

### 8.3 Run matching

Click **Run Matching** on the same page.

**What to expect — measured, not guessed:**

| Quantity | Value |
|---|---|
| Materials | 12 |
| Cross-CPSE pairs that exist in total | 51 |
| Candidate pairs from blocking alone (cap 50) | **17** |
| Candidate pairs with Milvus running | **17 or more** |

The 17 is reproducible: it was computed with the repository's own blocking code on this exact
file. Blocking cuts 51 possible pairs down to 17 while keeping all 12 genuine same-item pairs,
which share 5–9 block keys each.

With Milvus running, the vector pass adds further same-category pairs that share no keyword,
so the total will be higher than 17.

> **Diagnostic:** if you see **exactly 17** while Milvus should be on, the vector pass
> contributed nothing — check `docker ps` and the `milvus_enabled` line in Window A.

### 8.4 Review the matches

Sidebar → **Match Review**.

- Every candidate is **PENDING**. Nothing is pre-approved — not even a 0.99 score.
  The engine only recommends; a human decides.
- The strongest pairs (the three replicated items across CPSEs) appear first with
  High Confidence recommendations.
- Open one, click **Approve**.

### 8.5 See the consequences

| Page | What you should find |
|---|---|
| **Match Review** | the approved pair now shows as approved |
| **Mappings** | one new Common Material Code mapping for the approved pair, with both source CPSE codes preserved |
| **Audit Trail** | a `MATCH_APPROVED` event with your user and a timestamp |
| **Analytics** | material counts by CPSE, decision mix, review backlog |

### 8.6 Confirm the model was fetched

```powershell
dir "$env:USERPROFILE\Downloads\mira\backend\models\Mira.ai"
type "$env:USERPROFILE\Downloads\mira\backend\models\Mira.ai\MIRA_MODEL_PROVENANCE.txt"
```

Must list `model.safetensors` (≈753 MB) and read
`Auto-downloaded from Hugging Face: AshIndian/Mira.ai` / `Verified load: dim=1024`.

Second and later uploads do not touch the network — the model is loaded from
`backend\models\Mira.ai` in a few seconds.

---

## 9. Optional — API-only route (no frontend)

Everything the UI does is available at **http://127.0.0.1:8000/docs**.

1. `POST /api/auth/login` with `{"email": "admin@mira.gov.in", "password": "Admin@123"}` → copy `access_token`
2. Click **Authorize** (top right) and paste `Bearer <token>`
3. `POST /api/materials/upload` → choose `demo_materials.csv`
4. `POST /api/matching/run-batch` → body `{"max_candidates_per_material": 50, "overwrite": false}`
5. `GET /api/matching/candidates` · `GET /api/review/queue` · `POST /api/review/queue/{id}/action`

---

## 10. The embedding model — what happens, and what to do if it fails

MIRA needs a 1024-dimension sentence-embedding model for semantic similarity. A fresh clone
does not contain it (`models/` is gitignored).

**Automatic path (normal case):** on the first operation that needs embeddings — i.e. the
first upload — the app downloads the fine-tuned checkpoint from
`https://huggingface.co/AshIndian/Mira.ai`, saves it to `backend\models\Mira.ai`, loads it
back to verify it, and checks the dimension is 1024. Only then does it use it. No token or
login is required.

**If that download fails**, the app stops with a `FileNotFoundError` listing every path it
searched and the reason the download failed. It never silently falls back to a weaker model.

Common causes and fixes:

| Symptom in Window A | Cause | Fix |
|---|---|---|
| `download/save failed: … 403` / connection error | no internet, or a proxy/firewall blocks `huggingface.co` | allow the domain, or use the manual path below |
| `huggingface_hub is not installed` | missing package | `pip install huggingface_hub` |
| error naming `bitsandbytes` or `accelerate` | INT8 support missing | `pip install bitsandbytes accelerate` |
| `could not be located` with a long path list | download failed *and* no local model | see the manual path below |

**Manual path — place the model yourself:**

1. Open https://huggingface.co/AshIndian/Mira.ai/files and download **all** files into
   `backend\models\Mira.ai` — at minimum `config.json`, `model.safetensors`,
   `modules.json`, `tokenizer.json`, `tokenizer_config.json`, `sentence_bert_config.json`,
   plus the `1_Pooling` and `2_Normalize` folders.
2. Restart the backend. It will find the folder and skip downloading.

**Strict mode** (useful when you want to prove no download happens):

```powershell
$env:MODEL_AUTO_DOWNLOAD="false"
python -m uvicorn app.main:app --reload --reload-dir app --port 8000
```

With the model absent and this flag set, startup-adjacent matching must fail loudly — that
is the intended behaviour, not a bug.

---

## 11. Verification checklist

| # | Check | Where | Passes when |
|---|---|---|---|
| 1 | Database ready | `psql -U postgres -d mira -c "\dt"` | 10 tables listed |
| 2 | Milvus up | `docker ps` | 3 containers `Up` |
| 3 | Collection exists | `cd backend; python create_milvus_collection.py` | `Created …` or `already exists` |
| 4 | Backend alive | http://127.0.0.1:8000/health | `{"status":"ok",...}` |
| 5 | Frontend alive | http://localhost:5173 | login screen |
| 6 | Seed data | login as admin → **User Directory** | 4 users, 5 CPSEs |
| 7 | Upload works | Materials → Upload File | "Uploaded 12 records" |
| 8 | Model provisioned | Window A + `MIRA_MODEL_PROVENANCE.txt` | `dim=1024` |
| 9 | Matching works | Run Matching | ≥ 17 candidates (17 = blocking only) |
| 10 | Governance holds | Match Review | **all PENDING**, none auto-approved |
| 11 | Approval works | approve one → Mappings + Audit Trail | mapping + `MATCH_APPROVED` event |
| 12 | Persistence | restart backend, reload browser | materials and decisions still there |

---

## 12. Troubleshooting

| Message / symptom | Cause | Fix |
|---|---|---|
| `connection refused` or `could not connect to server` | PostgreSQL not running | start the PostgreSQL service |
| `password authentication failed for user "postgres"` | wrong password | correct `DATABASE_URL` in `backend\.env` |
| `relation "materials" does not exist` | schema never applied | redo §2.2 |
| `No module named 'app'` | started from the wrong folder | `cd backend` first |
| `No module named 'fastapi'` | virtual environment not active | activate `.venv` (§4.1) |
| `Port 8000 already in use` | an old server is still running | close it, or use `--port 8001` (and update `frontend/vite.config.ts`) |
| `No matching distribution found for torch==2.14.0+cpu` | torch not installed from the CPU index | §4.2, first command |
| UI shows **0 candidates** | all rows landed in one CPSE (unrecognised names), or files were uploaded twice | check the CPSE column; unrecognised names become `CPSE_GENERIC` and same-CPSE pairs are never matched |
| Candidates noticeably fewer than expected | Milvus down → vector pass contributes nothing, silently | `docker ps`; restart with `docker compose -f docker-compose.milvus.yml up -d` |
| First upload appears frozen | the 753 MB model download | wait; watch Window A for the progress lines |
| `Upload failed` with a 500 on a re-upload of the same file | re-upload is not idempotent yet | clear the tables (§13) and upload once |
| `Upload failed` with a 500, log shows `new-line character seen in unquoted field` | the file's extension does not match its real format (an `.xlsx` named `.csv`, say) | rename it to its true extension and upload again |
| `Malformed XML syntax: invalid token` | a literal `&` in an XML value (`L&T`) | write it as `&amp;` (`L&amp;T`) |
| `running scripts is disabled on this system` | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

---

## 13. Starting over / cleaning up

**Clear the data (keeps the schema and users):**

```powershell
psql -U postgres -d mira -c "TRUNCATE mappings, cnmc, match_suggestions, materials, audit_logs RESTART IDENTITY CASCADE;"
```

Then restart the backend so its in-memory view is rebuilt, and reload the browser.

**Empty the vector store too** (after clearing the tables, so the two stay consistent):

```powershell
cd backend
python create_milvus_collection.py --recreate
```

> Run `--recreate` **only** as part of a full reset. Running it after an upload discards the
> vectors for rows that still exist in PostgreSQL, and the next matching run will then find
> no semantic neighbours.

**Stop the stack:**

```powershell
# Window B (frontend) and Window A (backend): Ctrl+C
cd backend
docker compose -f docker-compose.milvus.yml stop   # keeps containers
```

Use `stop`, not `down`, if you want the uploaded vectors to survive. `down` removes the
containers (the data in `backend/volumes/` is kept, but you will need `up -d` again).

---

## 14. What to read next

| Document | Contents |
|---|---|
| `Architecture.md` | How ingestion, blocking, Milvus, scoring, gates, review and mappings fit together |
| `README.md` | Short overview and the API surface |
| `data/sample/` | Sample material files (note: `sample_materials.csv` uses placeholder CPSE names, so it demonstrates ingestion but produces no cross-CPSE candidates) |
