"""Client for the OpenEnv-compatible ChaosAgentGym environment.

Subclasses MCPToolClient — gets reset(), list_tools(), call_tool(), step()
for free.

Example — synchronous (most common):
    from openenv_chaos import ChaosEnv
    with ChaosEnv(base_url="http://localhost:8000").sync() as env:
        env.reset(seed=7)
        task = env.call_tool("read_task")
        print("task:", task["description"])
        env.call_tool("put_user", patch={"email": "new@example.com"})
        env.call_tool("get_user")
        env.call_tool("put_user", patch={"email": "new@example.com"})
        env.call_tool("get_user")
        outcome = env.call_tool("verify_user", expect={"email": "new@example.com"})
        print("reward:", outcome["reward"], "done:", outcome["done"])

Example — async:
    async with ChaosEnv(base_url="http://localhost:8000") as env:
        await env.reset(seed=7)
        ...

Example — against the hosted HF Space:
    env = ChaosEnv.from_env("<your-hf-username>/chaos-env").sync()
"""
from openenv.core.mcp_client import MCPToolClient


class ChaosEnv(MCPToolClient):
    """Client for ChaosAgentGym."""

    pass
