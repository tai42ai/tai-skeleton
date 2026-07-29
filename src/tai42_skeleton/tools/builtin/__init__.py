"""Runtime-native builtin tools shipped with the OS.

Only the two runtime-native tools live here — the ones with no HTTP route
equivalent by nature. Each module registers its tool through the ``tai42_app``
handle (``@tai42_app.tools.tool``) exactly as an external plugin would; there is
no default module list, so a deployment opts each module in by naming it in a
manifest ``tools[].module`` entry. The tools are ``file_loader`` (load a file
from a url or storage resource id) and ``interactions`` (the ``ask_user``
human-in-the-loop tool).

Management capabilities are not builtin tool modules: they live in the
operations layer and project onto the MCP tool surface directly from the
operations registry (gated by the manifest ``api_tools`` block), so there is no
hand-written management-tool module to opt in.
"""
