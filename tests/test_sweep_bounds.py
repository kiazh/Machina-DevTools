"""Regression checks for bounded project facts and repository reads."""

import builtins
import io
import json

import pytest

from machina.core import task_descriptor
from machina.repomap import ContentHashCache, build_repomap, scan


def test_manifest_over_byte_budget_is_not_used(tmp_path):
    path = tmp_path / "package.json"
    path.write_bytes(b'{"dependencies":{"react":"1"}}' + b" " * 262144)
    assert task_descriptor._stack_from_package_json(path) == {}


def test_multibyte_manifest_respects_byte_budget(tmp_path):
    path = tmp_path / "package.json"
    data = {"dependencies": {"react": "1"}, "description": "\u00e9" * 150000}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert task_descriptor._stack_from_package_json(path) == {}


def test_normal_manifest_keeps_stack(tmp_path):
    path = tmp_path / "package.json"
    path.write_text('{"dependencies":{"react":"1"}}')
    assert task_descriptor._stack_from_package_json(path) == {"frameworks": ("react",)}


def test_recent_context_does_not_visit_unused_history():
    class Recent(list):
        def __iter__(self):
            yield from super().__iter__()
            raise AssertionError("visited history beyond the three-entry window")

    recent = Recent([{"chunk_type": "user_message", "payload": {"content": "python"}}] * 3)
    assert task_descriptor._recent_snippets(recent) == (["python"] * 3, [])


def test_scan_refuses_file_that_grows_after_size_check(tmp_path, monkeypatch):
    target = tmp_path / "large.py"
    target.write_bytes(b"#" * (scan.MAX_BYTES_PER_FILE + 1))
    original_size = scan.os.path.getsize
    monkeypatch.setattr(
        scan.os.path, "getsize", lambda p: 1 if str(p) == str(target) else original_size(p)
    )
    result = build_repomap(str(tmp_path))
    assert result.skipped_large == ["large.py"]
    assert result.files_scanned == 0
    assert result.loc_by_file == {}


def test_repomap_never_requests_an_unbounded_source_read(tmp_path, monkeypatch):
    target = tmp_path / "module.py"
    target.write_text("import os\n")
    real_open = builtins.open
    reads = []

    class BoundedReader(io.BufferedReader):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    def tracked_open(path, mode="r", *args, **kwargs):
        if str(path) == str(target) and mode == "rb":
            return BoundedReader(io.FileIO(path, mode))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracked_open)
    cache = ContentHashCache()
    first = build_repomap(str(tmp_path), cache)
    second = build_repomap(str(tmp_path), cache)
    assert first.import_edges == second.import_edges == {"module": ["os"]}
    assert second.files_scanned == 0
    assert reads and all(0 < size <= scan.MAX_BYTES_PER_FILE + 1 for size in reads)


@pytest.mark.parametrize("payload", ["[]", "null", "42", '"entry"'])
def test_malformed_cache_top_level_degrades_to_fresh_scan(tmp_path, payload):
    root = tmp_path / "root"
    root.mkdir()
    (root / "module.py").write_text("import os\n")
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(payload)

    cache = ContentHashCache(str(cache_path))
    assert cache._entries == {}
    result = build_repomap(str(root), cache)
    assert result.loc_by_file == {"module.py": 1}
    assert result.files_scanned == 1


def test_malformed_cache_entries_are_ignored(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    module = root / "module.py"
    module.write_text("import os\n")
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({str(module): ["not", "an", "entry"]}))

    cache = ContentHashCache(str(cache_path))
    assert cache._entries == {}
    result = build_repomap(str(root), cache)
    assert result.loc_by_file == {"module.py": 1}
    assert result.files_scanned == 1


def test_cache_prunes_deleted_paths_after_complete_scan(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    module = root / "gone.py"
    module.write_text("x = 1\n")
    cache = ContentHashCache()

    build_repomap(str(root), cache)
    assert str(module) in cache._entries
    module.unlink()
    result = build_repomap(str(root), cache)

    assert result.files == []
    assert cache._entries == {}


def test_cache_prune_preserves_entries_for_other_roots(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "module.py").write_text("import os\n")
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n")

    cache = ContentHashCache()
    assert cache.store(str(outside), (1, [], False))
    build_repomap(str(root), cache)

    assert str(outside) in cache._entries
