# Examples

Runnable/reference examples for tai-skeleton.

- [`hello/`](hello/) — the smallest runnable app: one local tool, no agents, no
  external services. Walkthrough below.
- [`toolbox/`](toolbox/) — batteries: wires composition tool extensions (`chain`,
  `batch`) and generic tools from the [`tai-toolbox`](../../tai-toolbox) contrib
  package, all from the manifest. Needs the `toolbox` extra (`uv sync --extra
  toolbox` in the checkout).
- [`manifest.yml`](manifest.yml) — a fully-commented reference **manifest** showing
  the commonly used fields: tools, agents, external MCP servers, and plugin module hooks.

Environment variables are documented separately in [`.env.example`](../.env.example)
at the repo root.

## Hello world walkthrough

`hello/` contains two things:

```
hello/
├── manifest.yml        # tells the server what to load
└── myapp/
    ├── __init__.py
    └── tools.py        # defines the `greet` tool
```

**The tool** (`hello/myapp/tools.py`): a plain function registered with the
`@tai_app.tools.tool` decorator from `tai-contract`. The function's signature
becomes the tool's input schema and its docstring becomes the tool's description:

```python
from tai_contract.app import tai_app

@tai_app.tools.tool
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"
```

**The manifest** (`hello/manifest.yml`): each `tools:` entry names a Python module
to import; importing it runs the decorators, which registers the tools:

```yaml
tools:
  - title: hello
    module: myapp.tools
    include: [greet]
```

`module` is an import path, so the directory containing `myapp/` must be on
`PYTHONPATH` when the server starts.

**Run it** (from the repo root, after `uv sync`):

```bash
ACCESS_CONTROL_ENABLE=false \
PYTHONPATH=examples/hello \
uv run tai serve --manifest-path examples/hello/manifest.yml --port 8765
```

`ACCESS_CONTROL_ENABLE=false` turns off the request-auth gate; it is on by default
and checks every request against Redis, which this example does not use. The
manifest declares no connectors, so the connector engine skips its Postgres catalog
load and startup completes immediately, printing:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8765 (Press CTRL+C to quit)
```

**Call the tool** — the MCP endpoint is `http://127.0.0.1:8765/mcp`. From a second
terminal in the repo root:

```bash
uv run python - <<'EOF'
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8765/mcp") as c:
        print("tools:", [t.name for t in await c.list_tools()])
        result = await c.call_tool("greet", {"name": "World"})
        print("greet ->", result.content[0].text)

asyncio.run(main())
EOF
```

Expected output:

```
tools: ['greet']
greet -> Hello, World!
```

From here, grow the manifest: add more functions to `myapp/tools.py`, or see
[`manifest.yml`](manifest.yml) for agents, external MCP servers, and plugin modules.
