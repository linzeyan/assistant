from assistant.tools import build_registry
from assistant.tools.base import Tool, ToolResult


def test_builtin_tools_registered():
    reg = build_registry()
    names = {t.name for t in reg.all()}
    assert {"read_file", "write_file", "edit_file", "glob", "grep", "bash"} <= names


def test_mutating_tools_need_approval():
    reg = build_registry()
    assert reg.get("write_file").needs_approval is True
    assert reg.get("edit_file").needs_approval is True
    assert reg.get("bash").needs_approval is True
    # Read-only tools must not require approval.
    assert reg.get("read_file").needs_approval is False
    assert reg.get("grep").needs_approval is False


def test_schemas_are_openai_function_shaped():
    reg = build_registry()
    schemas = reg.schemas()
    sample = next(s for s in schemas if s["function"]["name"] == "read_file")
    assert sample["type"] == "function"
    assert sample["function"]["parameters"]["properties"]["path"]["type"] == "string"


async def _noop(args, ctx):
    return ToolResult(True, "ok")


def test_toolset_filtering():
    reg = build_registry()
    reg.register(Tool("aux", "aux", {"type": "object"}, _noop, toolset="extra"))
    core_names = {s["function"]["name"] for s in reg.schemas(toolsets={"core"})}
    assert "aux" not in core_names
    assert "read_file" in core_names
