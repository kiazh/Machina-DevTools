import ast
import json
import subprocess
import tempfile
import time
import tracemalloc
from pathlib import Path

from machina.core import task_descriptor as module

source = subprocess.check_output(
    ["git", "show", "a775fb09424fb89f323f3cf0b65c4beb6f716366:machina/core/task_descriptor.py"],
    text=True,
)
namespace = dict(vars(module))
names = {"_stack_from_package_json", "_recent_snippets"}
nodes = [
    node
    for node in ast.parse(source).body
    if isinstance(node, ast.FunctionDef) and node.name in names
]
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<baseline>", "exec"), namespace)


def measure(fn, arg):
    tracemalloc.start()
    started = time.perf_counter()
    result = fn(arg)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return {"seconds": round(elapsed, 6), "peak_bytes": peak}


with tempfile.TemporaryDirectory() as directory:
    manifest = Path(directory) / "package.json"
    manifest.write_bytes(b'{"dependencies":{"react":"1"}}' + b" " * (32 * 1024 * 1024))
    recent = [{"chunk_type": "user_message", "payload": {"content": "python"}}] * 1_000_000
    result = {}
    for name, arg in [("_stack_from_package_json", manifest), ("_recent_snippets", recent)]:
        result[name] = {
            "before": measure(namespace[name], arg),
            "after": measure(getattr(module, name), arg),
        }
    print(json.dumps(result, indent=2))
