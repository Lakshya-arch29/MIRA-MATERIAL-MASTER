# MIRA deployment guide

This repo is prepared for a small demo deployment with the React site and API on
Render, PostgreSQL on a managed provider, and the Qwen embedding model on an EC2
GPU instance. The model service stays separate because the free Render API
instance is too small to load the model.

## Cost and free-tier expectations

- Render can host the static frontend for free. Its free API service has 512 MB
  RAM and 0.1 CPU, sleeps after 15 minutes without traffic, and can take about a
  minute to wake. It is suitable for a demo, not dependable production service.
- Do not put real MIRA data in a free Render Postgres database: Render says those
  databases expire after 30 days, have a 1 GB limit, and have no backups. Use a
  managed Postgres provider with a retention policy that suits the data. Neon has
  a free option with usage limits; review its current plan before relying on it.
- EC2 GPU time is paid. An AWS G4dn instance uses an NVIDIA T4 and is intended for
  ML inference. Stop the instance when you do not need the model endpoint; while
  it is stopped, the Render backend cannot create new embeddings.
- The Qwen model and its cache live on the EC2 EBS volume. The model is downloaded
  at first boot, so allow time and disk space for that initial download.

## Services

| Component | Host | Purpose |
|---|---|---|
| Frontend | Render Static Site | React/Vite UI; serves static files |
| API | Render Free Web Service | FastAPI, auth, matching, PostgreSQL access |
| Database | Managed PostgreSQL | Materials, users, matches, reviews |
| Embedding model | EC2 GPU | Authenticated `/embed` API running `AshIndian/Mira.ai` |

## 1. Create the database

Create a PostgreSQL database with a provider you are comfortable using. Copy its
TLS-enabled connection string. Apply the schema once from a machine with `psql`:

```bash
psql "<DATABASE_URL>" -f backend/mira_full_schema.sql
```

For a database created with an earlier MIRA schema, apply the additive embedding
cache migration instead:

```bash
psql "<DATABASE_URL>" -f backend/002_create_material_embedding_cache.sql
```

Keep a backup of that connection string in a password manager. Do not put it in
the repo or in frontend variables.

## 2. Start the model API on EC2

Choose an EC2 GPU image with a working NVIDIA driver and CUDA-compatible PyTorch.
After connecting to the instance, verify `nvidia-smi` and check that Python can
see CUDA. Create the service account and writable application directory first:

```bash
sudo useradd --system --create-home --home-dir /opt/mira-model-server mira
sudo mkdir -p /opt/mira-model-server
sudo chown -R mira:mira /opt/mira-model-server
```

Copy `app.py`, `requirements.txt`, `mira-model-server.service`, and
`Caddyfile.example` from `backend/model_server/` in your local checkout into
`/opt/mira-model-server/`. Install a PyTorch build
matching that image using the official
[PyTorch install selector](https://pytorch.org/get-started/locally/), then install
the model service dependencies:

```bash
sudo -u mira -H python3 -m venv /opt/mira-model-server/.venv
sudo -u mira -H /opt/mira-model-server/.venv/bin/pip install --upgrade pip
# Run the CUDA-specific PyTorch install command from the selector, using this venv's pip.
sudo -u mira -H /opt/mira-model-server/.venv/bin/pip install -r /opt/mira-model-server/requirements.txt
```

When copying the selector command, replace its `pip` executable with
`/opt/mira-model-server/.venv/bin/pip` so PyTorch installs into this virtual
environment.

Create `/etc/mira-model-server.env` with a long random key and restrict its
permissions:

```ini
MIRA_API_KEY=<long-random-secret>
MIRA_MODEL_ID=AshIndian/Mira.ai
MIRA_EMBED_BATCH_SIZE=32
```

Install the checked-in unit file and start the service:

```bash
sudo cp /opt/mira-model-server/mira-model-server.service /etc/systemd/system/
sudo chmod 600 /etc/mira-model-server.env
sudo chown root:mira /etc/mira-model-server.env
sudo systemctl daemon-reload
sudo systemctl enable --now mira-model-server
sudo systemctl status mira-model-server
```

The service listens on `127.0.0.1:8001`; it is not exposed directly to the
internet. Give the hostname a stable public IP, such as an Elastic IP, so it
continues pointing to the instance after a stop/start. Put Caddy or another TLS
reverse proxy in front of it. For Caddy, copy
`/opt/mira-model-server/Caddyfile.example` to the Caddy config and replace
`model.example.com` with a hostname you control. Point that hostname to EC2 and
allow inbound 80/443. Restrict SSH to your own IP, and do not open port 8001.
Caddy should proxy to `127.0.0.1:8001` and obtain HTTPS for the hostname.

Check the public health route and authenticated embedding route:

```bash
curl https://model.example.com/health
curl -H "X-API-Key: <long-random-secret>" \
  -H "Content-Type: application/json" \
  -d '{"text":["industrial gate valve"]}' \
  https://model.example.com/embed
```

Health should report `status: ok`, a device (`cuda` on a working GPU), and 1024
dimensions. The model API key must match the Render backend's `MIRA_API_KEY`.

## 3. Configure Render

Connect this repo to Render and create a Blueprint from `render.yaml`. It defines
the API web service and the static frontend. Set the prompted variables:

**Backend service**

- `DATABASE_URL`: managed Postgres connection string.
- `CORS_ORIGINS`: the exact deployed frontend origin, for example
  `https://mira-frontend.onrender.com`.
- `MIRA_MODEL_SERVER_URL`: HTTPS model URL, for example
  `https://model.example.com`.
- `MIRA_API_KEY`: same secret used in `/etc/mira-model-server.env` on EC2.
- `SEED_ADMIN_EMAIL` and `SEED_ADMIN_PASSWORD`: your initial administrator login.
- `SECRET_KEY` is generated by Render in the Blueprint.

The deployment disables demo-user seeding; only the configured admin is seeded.
This prevents new demo accounts from being created; it does not remove demo users
already present in a database you reuse. Start with a fresh database or remove
those accounts before exposing a reused database publicly.
The model is not downloaded on Render. The backend sends requests to the EC2
model API instead.

**Frontend static site**

- `VITE_API_URL`: the deployed backend base URL, e.g.
  `https://mira-backend.onrender.com`.

Vite reads this variable at build time, so changing it requires a frontend
rebuild/deploy. The static-site rewrite keeps React Router routes working after a
refresh.

## 4. First-deploy checklist

1. Confirm the Postgres schema was applied and the API URL points to that DB.
2. Confirm `/health` succeeds on both the Render API and model hostname.
3. Confirm model health reports 1024 dimensions and CUDA (or the expected CPU
   fallback).
4. Confirm CORS lists only the frontend origin.
5. Log in using the configured admin account, upload a small CSV, and run a small
   matching batch before adding real data.
6. Keep a separate database backup. Free Render Postgres is not a durable backup.

## Local files added for deployment

- `render.yaml` declares the frontend and API services.
- `backend/model_server/` contains the authenticated EC2 embedding API and its
  systemd/Caddy examples.
- Ingestion stores each generated 1024D vector in PostgreSQL. Matching reuses that
  vector for scoring and copies the same vector to Milvus when Milvus is available.
- `backend/002_create_material_embedding_cache.sql` upgrades an existing database;
  the full schema includes this table for a fresh database.
- `backend/app/core/database.py` normalizes provider `postgres://` URLs to the
  installed psycopg v3 SQLAlchemy driver.
- `backend/app/main.py` uses configurable exact CORS origins.
- `backend/app/core/seed.py` lets hosted deployments skip hard-coded demo users.
