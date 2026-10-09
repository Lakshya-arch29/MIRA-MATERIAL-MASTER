from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.seed import seed_default_users_and_cpses


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Seed default CPSEs and demonstration users on startup
    seed_default_users_and_cpses()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="AI-driven cross-CPSE material harmonization backend",
    lifespan=lifespan,
)


def _cors_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ORIGINS", settings.cors_origins)
    return [origin.strip().rstrip("/") for origin in raw_origins.split(",") if origin.strip()]

# Enable CORS for React Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "mira-backend",
    }


@app.get("/", include_in_schema=False)
def landing_page():
    """Friendly landing page so the root URL shows something useful."""
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html>
        <head><title>MIRA Backend</title>
        <style>
          body { font-family: system-ui, sans-serif; max-width: 640px; margin: 48px auto; padding: 0 16px; }
          h1 { margin-bottom: 4px; }
          .ok { color: #15803d; font-weight: bold; }
          ul { line-height: 2; }
          code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
        </style>
        </head>
        <body>
          <h1>MIRA Backend</h1>
          <p class="ok">&#9679; Service is running</p>
          <ul>
            <li><a href="/docs">Interactive API docs (try every endpoint here)</a></li>
            <li><a href="/health"><code>GET /health</code></a> — health check</li>
            <li><a href="/api/materials"><code>GET /api/materials</code></a> — list materials</li>
            <li><a href="/api/matching/candidates"><code>GET /api/matching/candidates</code></a> — list match candidates</li>
            <li><a href="/api/review/queue"><code>GET /api/review/queue</code></a> — human review queue</li>
            <li><a href="/api/analytics/overview"><code>GET /api/analytics/overview</code></a> — dashboard KPIs</li>
          </ul>
          <p>Tip: use <a href="/docs"><code>/docs</code></a> to upload a CSV, run matching, and approve matches.</p>
        </body>
        </html>
        """
    )
