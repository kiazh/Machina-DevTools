"""A3 campaign: RepoMap cache insertion binds the bytes that were parsed (MCH-003)."""

from __future__ import annotations

from machina.repomap import ContentHashCache, build_repomap


def test_store_declines_when_the_file_changed_since_parsing(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("import os\n", encoding="utf-8")
    parsed_bytes = path.read_bytes()

    path.write_text("import re\n", encoding="utf-8")  # same size, new bytes

    cache = ContentHashCache()
    assert cache.store(str(path), (1, ["os"], False), source=parsed_bytes) is False
    assert cache.lookup(str(path)) is None


def test_scan_never_binds_an_old_parse_to_a_new_hash(tmp_path, monkeypatch):
    target = tmp_path / "m.py"
    target.write_text("import os\n", encoding="utf-8")

    from machina.repomap import scan

    real_parse = scan._parse
    swapped = False

    def parse_then_mutate(source):
        nonlocal swapped
        result = real_parse(source)
        if not swapped:
            swapped = True
            # Same-size rewrite between the scanner's read and cache.store.
            target.write_text("import re\n", encoding="utf-8")
        return result

    monkeypatch.setattr(scan, "_parse", parse_then_mutate)

    cache = ContentHashCache()
    first = build_repomap(str(tmp_path), cache=cache)
    second = build_repomap(str(tmp_path), cache=cache)

    # The first pass parsed the old bytes, so its own map is old — but the
    # cache must not serve that parse under the new file's digest.
    assert "os" in first.import_edges["m"]
    assert second.files_scanned == 1
    assert "re" in second.import_edges["m"]
    assert "os" not in second.import_edges["m"]
    assert cache.lookup(str(target))[1] == ["re"]
