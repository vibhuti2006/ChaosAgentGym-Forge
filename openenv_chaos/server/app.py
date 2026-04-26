"""FastAPI entry point for the OpenEnv-compatible Chaos environment.

Run locally:
    uvicorn openenv_chaos.server.app:app --host 0.0.0.0 --port 8000

Run via Docker (HF Spaces does this automatically):
    docker build -t chaos-env -f openenv_chaos/server/Dockerfile .
    docker run -p 8000:8000 chaos-env

Mounts the static frontend (frontend/index.html) at /ui so the HF Space's
App tab shows an interactive console, not "Not Found". OpenEnv API endpoints
remain at their original paths (/health, /reset, /step, /tools/list, ...).
"""
import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

try:
    from .chaos_environment import ChaosEnvironment
except ImportError:
    from openenv_chaos.server.chaos_environment import ChaosEnvironment


max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    ChaosEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="chaos_env",
    max_concurrent_envs=max_concurrent,
)

# Mount the interactive frontend if the directory exists alongside the
# project. This is what HF Space visitors see at the App tab.
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def _root_to_ui():
        # Redirect bare `/` to the frontend so the App tab shows the UI.
        return RedirectResponse(url="/ui/")


def main():
    """Direct-run entry point: `python -m openenv_chaos.server.app`."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
