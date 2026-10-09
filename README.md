# MIRA

**Material Identity & Resolution Architecture - SIH26099: AI-Driven Standardization and
Harmonization of Material Codes Across CPSEs.**

MIRA ingests heterogeneous CPSE material masters, finds duplicate / near-duplicate /
equivalent materials with a hybrid AI + rules matcher, routes uncertain pairs to human
reviewers, and produces traceable **Common National Material Codes (NMC)** with full audit
and analytics. Source CPSE codes are always preserved - MIRA is a harmonization layer,
never an ERP replacement.

> **Similarity finds the candidate. Specifications decide whether it is safe.
> Humans control the final mapping.**

## Quickstart (local run)

Full step-by-step guide: **[LOCAL_SETUP.md](LOCAL_SETUP.md)** (Windows-first, written for a
machine with nothing installed). Short version:

```powershell
# 1. code
git clone https://github.com/AshIndian-Coder/mira.git mira
cd mira

# 2. database (once)
psql -U postgres -c "CREATE DATABASE mira;"
psql -U postgres -d mira -f backend/mira_full_schema.sql

# 3. vector store (Docker Desktop must be running)
cd backend
docker compose -f docker-compose.milvus.yml up -d
python create_milvus_collection.py

# 4. backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install bitsandbytes accelerate
# optional: create backend/.env with DATABASE_URL if your postgres password is not "postgres"
python -m uvicorn app.main:app --reload --reload-dir app --port 8000
```

```powershell
# 5. frontend (second window)
cd frontend
npm install
npm run dev
```

Then open:

| URL | What it is |
|---|---|
| http://localhost:5173 | the application UI |
| http://127.0.0.1:8000/docs | interactive API |
| http://127.0.0.1:8000 | backend landing page |

macOS / Linux: use `python3 -m venv .venv` and `source .venv/bin/activate`; the rest is
the same.

For the Render + EC2 Qwen deployment layout, setup steps, and free-tier limitations, see
[`DEPLOYMENT.md`](DEPLOYMENT.md).

**Prerequisites:** Python 3.11-3.13, Node 20.19+ or 22.12+, PostgreSQL 14+, Docker Desktop,
Git. Install in `backend/` - the `requirements.txt` at the repository root is a stray
single-line pin (`pymilvus`) and is not the application's dependency list.

The current backend calls a separate authenticated embedding service at
`MIRA_MODEL_SERVER_URL`; it does not download or load Qwen in the API process. For the
Render + EC2 setup, follow [DEPLOYMENT.md](DEPLOYMENT.md). For local runs, configure the
backend `.env` with the model service URL and the same `MIRA_API_KEY` used by that service.
The older [LOCAL_SETUP.md](LOCAL_SETUP.md) contains model-download instructions from a
previous setup and is not accurate for this runtime.

On first backend start the configured admin and baseline CPSE records are seeded. Hosted
deployment disables demo-user seeding; set `SEED_ADMIN_EMAIL` and `SEED_ADMIN_PASSWORD`
to create the administrator.

## How matching works

```text
upload (CSV / TXT / XLSX / XLS / JSON / XML)
  -> normalize + parse specifications
  -> candidates: cross-CPSE blocking  +  Milvus vector search (additive)
  -> hybrid score -> critical gates -> engine decision
  -> human review -> mapping + Common National Material Code
  -> audit trail + analytics
```

- **Weights:** text 0.175, semantic 0.400, specification 0.250, grade 0.125, other 0.050.
  The five-component shape is frozen; only the numbers were tuned (spec >= 0.25,
  grade >= 0.10, sum = 1.0).
- **Decisions:** a critical conflict is DIFFERENT; score >= 0.85 with all applicable gates
  passing is HIGH_CONFIDENCE; an UNKNOWN critical value, or a score above 0.45, is REVIEW.
- **Nothing is auto-approved.** HIGH_CONFIDENCE is a recommendation for a human. The only
  candidates excluded from the approval queue are DIFFERENT ones, and `automation_rate`
  stays 0 by design.
- **Milvus only adds candidates.** With Milvus down the run still completes with the same
  blocking candidates, so a low candidate count is your signal to check `docker ps` and
  `MILVUS_ENABLED`.
- **Provenance is never guessed.** The CPSE is read from evidence, in this order: a `cpse`
  column, the Excel sheet name, the filename, then a code prefix such as `IOCL-V001`. If
  none of those match a known CPSE the row becomes `CPSE_GENERIC`, and since matching is
  cross-CPSE only, a file whose rows all land in one bucket yields zero candidates - that
  is expected, not a failure. `data/sample/sample_materials.csv` is such a file
  (placeholder CPSE names); use a file with real CPSE names.
- **Upload formats** (all six verified against the parser): `.csv`, `.txt`, `.xls`,
  `.xlsx`, `.json`, `.xml`. Extensions are case-insensitive. Excel workbooks may hold one
  CPSE per sheet - every sheet is parsed and its name counts as CPSE evidence. `.pdf` is
  not accepted by `/api/materials/upload`.

## API surface

38 endpoints under `/api`, plus `/` and `/health`. JWT auth required except for login,
health and the landing page. Full detail at `/docs`.

| Area | Endpoints |
|---|---|
| Materials | `POST /api/materials/upload` · `GET /api/materials` · `GET /api/materials/stats` · `GET /api/materials/{id}` |
| Matching | `POST /api/matching/compare` · `POST /api/matching/run-batch` · `GET /api/matching/candidates` · `GET /api/matching/candidates/{id}` · `GET /api/matching/stats` |
| CNMC matching | `POST /api/matching/cnmc/candidates` · `POST /api/matching/cnmc/run-batch` |
| Review | `GET /api/review/queue` · `GET /api/review/queue/{id}` · `POST /api/review/queue/{id}/action` · `GET /api/review/summary` |
| Mappings | `GET /api/mappings` · `GET /api/mappings/{id}` · `POST /api/mappings/generate` · `POST /api/mappings/{id}/approve` · `POST /api/mappings/{id}/attach` · `GET /api/mappings/export/flat` |
| Audit | `GET /api/audit` · `GET /api/audit/export` |
| Analytics | `GET /api/analytics/overview` · `/by-cpse` · `/categories` · `/scores` · `/data-quality` |
| Auth | `POST /api/auth/login` · `POST /api/auth/refresh` · `GET /api/auth/me` · `POST /api/auth/logout` |
| Users | `GET /api/users` · `POST /api/users` · `GET /api/users/{id}` · `PUT /api/users/{id}` · `DELETE /api/users/{id}` · `GET /api/users/cpses` |
| Meta | `GET /` (landing page) · `GET /health` |

## Persistence (PostgreSQL)

All runtime data is **Postgres-backed** - uploads, candidates, review decisions, mappings,
CNMC records and audit events survive server restarts.

- Schema: `backend/mira_full_schema.sql` (10 tables, including durable material
  embeddings). The app does not create tables: apply the schema before the first start.
- Each material's 1024D vector is generated during ingestion and stored as JSONB in
  `material_embedding_cache`. Matching reuses that vector for semantic scoring and sends
  the same vector to Milvus when it is available; the model is not rerun for saved rows.
- Engine: `backend/app/db_adapter.py` - `PersistentList` (drop-in list replacement) plus
  `DBRow` (a dict with write-through on in-place mutation, so
  `candidate["review_status"] = "APPROVED"` persists).
- Stores: `backend/app/store.py` (`MATERIALS`, `CANDIDATES`), `MAPPINGS` in
  `backend/app/api/v1/mappings.py`, `AUDIT_EVENTS` in `backend/app/api/v1/audit.py`.
- Route modules were not rewritten to use the ORM; SQLAlchemy models for materials, CPSEs,
  matches and users live in `backend/app/models/` with Pydantic schemas in
  `backend/app/schemas/`.
- Proof test: upload -> restart -> data is still there; approve -> restart -> still
  approved.

## Embedding model

- Model service: `backend/model_server/app.py` loads `AshIndian/Mira.ai` and provides
  authenticated `/embed` and public `/health` endpoints.
- Backend client: `backend/app/services/matching/embeddings.py` sends text batches to
  `MIRA_MODEL_SERVER_URL`, authenticates with `MIRA_API_KEY`, and validates 1024-dimension
  responses. The service selects CUDA when available and falls back to CPU.
- The API deployment should not download or load the model. See
  [DEPLOYMENT.md](DEPLOYMENT.md) for the separate EC2 model service setup.

## Repository layout

```text
mira/
├── backend/
│   ├── mira_full_schema.sql          # Postgres schema (10 tables)
│   ├── docker-compose.milvus.yml     # etcd + MinIO + Milvus
│   ├── create_milvus_collection.py   # one-time collection setup
│   ├── benchmark_matching.py · benchmark_pipeline.py
│   ├── requirements.txt · pytest.ini
│   ├── app/
│   │   ├── main.py                   # FastAPI app + landing page
│   │   ├── store.py                  # DB-backed shared stores
│   │   ├── db_adapter.py             # PersistentList / DBRow persistence engine
│   │   ├── core/                     # config, database engine, seed data
│   │   ├── api/v1/                   # auth, users, materials, matching, review,
│   │   │                             # mappings, audit, analytics
│   │   ├── models/ · schemas/        # SQLAlchemy models + Pydantic schemas
│   │   ├── services/                 # ingestion (+adapters, LLM fallback), normalization,
│   │   │                             # parsing, blocking, matching, cnmc, harmonization,
│   │   │                             # clustering, evaluation, training
│   │   └── ml_pipeline/              # dataset build, training, quantisation, evaluation
│   └── tests/                        # 35 modules / 278 test functions
├── frontend/                         # React 19 + Vite 8 + TypeScript (11 screens)
├── data/                             # dev + evaluation artifacts, sample files
├── notebooks/ · scripts/             # Colab fine-tune notebook, evaluation helpers
├── LOCAL_SETUP.md                    # local run guide (start here)
├── Architecture.md                   # system architecture (stack, AI design, API boundary)
├── Project_Requirements.md           # SIH requirements + implementation status
├── Phases.md                         # phased plan + current state
├── memory.md · rules.md              # project memory, repo rules
└── README.md                         # this file
```

## Documentation index

- **[LOCAL_SETUP.md](LOCAL_SETUP.md)** - run the whole stack on your machine, from nothing
- **[Architecture.md](Architecture.md)** - stack, AI design, matching pipeline, API boundary
- **[Project_Requirements.md](Project_Requirements.md)** - SIH requirements mapped to what is
  actually implemented, verified against the current commit
- **[Phases.md](Phases.md)** - phased plan + current state
- **[memory.md](memory.md)** - frozen decisions, status, latest updates
- **[rules.md](rules.md)** - repository rules

## Status & contributing

- Integration branch: `main`. Latest merge: PR #11 - Hugging Face auto-download now copies
  the INT8 checkpoint instead of re-saving it (commit `747d43a`).
- Tests: 35 modules / 278 test functions under `backend/tests/`; run `pytest` from
  `backend/`. The suite exercises the same database the app uses, so point it at a
  throwaway database.
- Governance rule that must not regress: **no automated approval path.** Only DIFFERENT
  candidates may be excluded from human review.
- Data hygiene: keep large or local-only datasets out of Git. `Final_Master_Material_Records.csv`
  is a **Git-LFS pointer** (a 239 MB payload that is not in the repository) - use
  `Original_company_records_no_synthetic.csv` or the files under `data/` instead.
- Never commit `.env`. Configure a unique admin password, generated `SECRET_KEY`, and
  exact frontend `CORS_ORIGINS` before deployment.
