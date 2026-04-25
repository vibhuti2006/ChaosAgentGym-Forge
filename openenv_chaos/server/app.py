"""FastAPI entry point for the OpenEnv-compatible Chaos environment.

Run locally:
    uvicorn openenv_chaos.server.app:app --host 0.0.0.0 --port 8000

Run via Docker (HF Spaces does this automatically):
    docker build -t chaos-env -f openenv_chaos/server/Dockerfile .
    docker run -p 8000:8000 chaos-env
"""
import os

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


def main():
    """Direct-run entry point: `python -m openenv_chaos.server.app`."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
