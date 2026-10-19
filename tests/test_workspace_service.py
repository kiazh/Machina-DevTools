"""T5 workspace file service tests (plan §11, §24B/§24C relevant rows).

Service-level tests over temporary workspaces: list pagination/ordering/
exclusions, read bounds and refusals, save revision/conflict/atomicity,
document mirrors and the revision registry. Every refusal is asserted as a
structured reason, never as a crash or a guessed result.
"""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest

from machina.workspace import (
    DEFAULT_EXCLUSIONS,
    MAX_READ_BYTES,
    WorkspaceError,
    WorkspaceFiles,
    WorkspaceService,
    revision_from_bytes,
)
from machina.workspace.documents import DocumentMirrorStore
from machina.workspace.files import revision_registry


def _service(root: Path | str, trust: bool = True) -> tuple[WorkspaceService, object, list]:
    events: list[dict] = []
    service = WorkspaceService(send_event=events.append)
    identity = service.open(str(root), trust)
    return service, identity, events


def _root(tmp_path: Path, name: str = "ws") -> Path:
    root = tmp_path / name
    root.mkdir()
    return root


def _is_root_user() -> bool:
    geteuid = getattr(os, "geteuid", None)
    return geteuid is not None and geteuid() == 0


# ─── list: pagination, ordering, exclusions, symlinks, denials ───────────────
def test_list_pagination_cursor_stable(tmp_path):
    root = _root(tmp_path)
    for index in range(250):
        (root / f"f{index:03d}.txt").write_text("x")
    files = WorkspaceFiles(root)

    first = files.list_dir("", limit=200)
    assert len(first.entries) == 200
    assert first.truncated is True
    assert first.next_cursor == "200"
    assert [entry.name for entry in first.entries] == [f"f{i:03d}.txt" for i in range(200)]

    second = files.list_dir("", cursor=first.next_cursor, limit=200)
    assert [entry.name for entry in second.entries] == [f"f{i:03d}.txt" for i in range(200, 250)]
    assert second.truncated is False
    assert second.next_cursor is None

    # Exhausted cursor returns an empty page, not an error.
    exhausted = files.list_dir("", cursor="250", limit=200)
    assert exhausted.entries == ()
    assert exhausted.next_cursor is None


def test_list_dirs_first_case_aware_deterministic_ties(tmp_path):
    root = _root(tmp_path)
    (root / "Zebra").mkdir()
    (root / "alpha").mkdir()
    (root / "Banana.txt").write_text("b")
    (root / "apple.txt").write_text("a")
    (root / "Apple2.txt").write_text("a2")

    page = WorkspaceFiles(root).list_dir("")
    assert [entry.name for entry in page.entries] == [
        "alpha",
        "Zebra",
        "apple.txt",
        "Apple2.txt",
        "Banana.txt",
    ]
    assert [entry.kind for entry in page.entries] == [
        "directory",
        "directory",
        "file",
        "file",
        "file",
    ]


def test_list_exclusions_disclosed_and_hidden_toggle(tmp_path):
    root = _root(tmp_path)
    (root / "node_modules").mkdir()
    (root / ".git").mkdir()
    (root / "visible.txt").write_text("v")
    files = WorkspaceFiles(root)

    default = files.list_dir("")
    names = [entry.name for entry in default.entries]
    assert "visible.txt" in names
    assert "node_modules" in names  # declared exclusion policy, still reported
    assert ".git" not in names  # hidden files are off by default
    assert default.exclusions == tuple(sorted(DEFAULT_EXCLUSIONS))
    flagged = next(entry for entry in default.entries if entry.name == "node_modules")
    assert flagged.excluded is True
    assert flagged.kind == "directory"

    hidden = files.list_dir("", include_hidden=True)
    hidden_git = next(entry for entry in hidden.entries if entry.name == ".git")
    assert hidden_git.excluded is True


def test_list_symlink_escape_flagged_not_followed(tmp_path):
    root = _root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (root / "escape").symlink_to(outside)

    page = WorkspaceFiles(root).list_dir("")
    entry = next(item for item in page.entries if item.name == "escape")
    assert entry.symlink is True
    assert entry.size_bytes is None
    assert entry.kind == "file"  # outside targets are never probed

    with pytest.raises(WorkspaceError) as error:
        WorkspaceFiles(root).list_dir("escape")
    assert error.value.reason == "denied"
    assert "escapes" in error.value.message


def test_list_permission_denied_reported_distinctly(tmp_path):
    if _is_root_user():
        pytest.skip("permission-denied semantics do not apply to root")
    root = _root(tmp_path)
    (root / "empty").mkdir()
    locked = root / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        files = WorkspaceFiles(root)
        empty_page = files.list_dir("empty")
        assert empty_page.entries == ()
        with pytest.raises(WorkspaceError) as error:
            files.list_dir("locked")
        assert error.value.reason == "denied"
        assert "permission denied" in error.value.message
    finally:
        locked.chmod(0o700)


# ─── read: encoding, line endings, bounds, refusals ─────────────────────────
def test_read_utf8_revision_and_line_ending_detection(tmp_path):
    root = _root(tmp_path)
    cases = [
        (b"a\nb\n", "LF"),
        (b"a\r\nb\r\n", "CRLF"),
        (b"a\rb\r", "CR"),
        (b"a\r\nb\nc\r", "mixed"),
        (b"no newline", "LF"),
    ]
    files = WorkspaceFiles(root)
    for index, (data, expected) in enumerate(cases):
        name = f"case{index}.txt"
        (root / name).write_bytes(data)
        result = files.read_text(name)
        assert result.line_ending == expected
        assert result.encoding == "utf-8"
        assert result.size_bytes == len(data)

    payload = "héllo 😀\n".encode("utf-8")
    (root / "unicode.txt").write_bytes(payload)
    unicode_result = files.read_text("unicode.txt")
    assert unicode_result.text == payload.decode("utf-8")
    assert unicode_result.revision == revision_from_bytes(payload)
    assert unicode_result.language_id == "plaintext"
    assert unicode_result.read_only is False


def test_read_binary_and_unsupported_encoding_refusals(tmp_path):
    root = _root(tmp_path)
    files = WorkspaceFiles(root)

    (root / "nul.bin").write_bytes(b"abc\x00def")
    with pytest.raises(WorkspaceError) as nul_error:
        files.read_text("nul.bin")
    assert nul_error.value.reason == "binary"

    (root / "invalid.bin").write_bytes(b"\xc3\x28\xa0")
    with pytest.raises(WorkspaceError) as invalid_error:
        files.read_text("invalid.bin")
    assert invalid_error.value.reason == "binary"

    (root / "utf16.txt").write_bytes(b"\xff\xfeh\x00i\x00")
    with pytest.raises(WorkspaceError) as utf16_error:
        files.read_text("utf16.txt")
    assert utf16_error.value.reason == "unsupported-encoding"


def test_read_size_boundary_at_limit_and_over(tmp_path):
    root = _root(tmp_path)
    files = WorkspaceFiles(root)
    (root / "at_limit.txt").write_bytes(b"a" * MAX_READ_BYTES)
    at_limit = files.read_text("at_limit.txt")
    assert at_limit.size_bytes == MAX_READ_BYTES

    (root / "over.txt").write_bytes(b"a" * (MAX_READ_BYTES + 1))
    with pytest.raises(WorkspaceError) as error:
        files.read_text("over.txt")
    assert error.value.reason == "too-large"


def test_read_missing_file_and_directory_refusals(tmp_path):
    root = _root(tmp_path)
    files = WorkspaceFiles(root)
    with pytest.raises(WorkspaceError) as missing:
        files.read_text("nope.txt")
    assert missing.value.reason == "missing"

    (root / "dir").mkdir()
    with pytest.raises(WorkspaceError) as directory:
        files.read_text("dir")
    assert directory.value.reason == "not-regular"


def test_read_symlinked_file_and_parent_escape_refused(tmp_path):
    root = _root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    (root / "linkdir").symlink_to(outside)

    files = WorkspaceFiles(root)
    with pytest.raises(WorkspaceError) as file_error:
        files.read_text("link.txt")
    assert file_error.value.reason == "denied"
    with pytest.raises(WorkspaceError) as parent_error:
        files.read_text("linkdir/secret.txt")
    assert parent_error.value.reason == "denied"
    with pytest.raises(WorkspaceError) as list_error:
        files.list_dir("linkdir")
    assert list_error.value.reason == "denied"


def test_read_inside_root_symlink_is_followed(tmp_path):
    root = _root(tmp_path)
    (root / "real.txt").write_text("hello")
    (root / "link.txt").symlink_to(root / "real.txt")
    result = WorkspaceFiles(root).read_text("link.txt")
    assert result.text == "hello"


def test_read_permission_denied_file_reported_distinctly(tmp_path):
    if _is_root_user():
        pytest.skip("permission-denied semantics do not apply to root")
    root = _root(tmp_path)
    target = root / "secret.txt"
    target.write_text("x")
    target.chmod(0)
    try:
        with pytest.raises(WorkspaceError) as error:
            WorkspaceFiles(root).read_text("secret.txt")
        assert error.value.reason == "denied"
        assert "denied" in error.value.message
    finally:
        target.chmod(0o644)


def test_list_symlink_cycle_is_safe(tmp_path):
    root = _root(tmp_path)
    (root / "sub").mkdir()
    (root / "sub" / "loop").symlink_to(root)
    files = WorkspaceFiles(root)

    page = files.list_dir("sub")
    loop_entry = next(entry for entry in page.entries if entry.name == "loop")
    assert loop_entry.symlink is True
    assert loop_entry.kind == "directory"

    # One deliberate step through the cycle lists the root again; listing is
    # single-level and never recurses, so a cycle cannot hang the service.
    nested = files.list_dir("sub/loop")
    assert any(entry.name == "sub" for entry in nested.entries)


def test_traversal_and_absolute_paths_refused(tmp_path):
    root = _root(tmp_path)
    files = WorkspaceFiles(root)
    for rel in ("../outside.txt", "..", "/etc/passwd", "a/../../b", "..\\x", "a\\..\\b"):
        with pytest.raises(WorkspaceError) as error:
            files.read_text(rel)
        assert error.value.reason == "denied", rel


# ─── save: revisions, conflicts, atomicity, refusals ────────────────────────
async def test_save_success_returns_new_revision(tmp_path):
    root = _root(tmp_path)
    (root / "a.txt").write_bytes(b"old\n")
    files = WorkspaceFiles(root)
    base = files.read_text("a.txt").revision

    result = await files.save_text("a.txt", "new\n", base, ui_version=7)
    assert result.ok is True
    assert result.revision == revision_from_bytes(b"new\n")
    assert result.saved_ui_version == 7
    assert (root / "a.txt").read_bytes() == b"new\n"
    assert revision_registry.get(root / "a.txt") == result.revision


async def test_save_expected_revision_mismatch_conflicts_not_overwrites(tmp_path):
    root = _root(tmp_path)
    (root / "a.txt").write_bytes(b"one\n")
    files = WorkspaceFiles(root)
    base = files.read_text("a.txt").revision

    # Simulated external writer (TOCTOU between read and save).
    (root / "a.txt").write_bytes(b"externally changed\n")

    result = await files.save_text("a.txt", "mine\n", base)
    assert result.ok is False
    conflict = result.conflict
    assert conflict is not None
    assert conflict.disk_revision == revision_from_bytes(b"externally changed\n")
    assert conflict.base_revision == base
    assert conflict.reason == "revision-mismatch"
    assert (root / "a.txt").read_bytes() == b"externally changed\n"


async def test_concurrent_saves_of_same_path_serialize(tmp_path):
    root = _root(tmp_path)
    (root / "a.txt").write_bytes(b"one\n")
    files = WorkspaceFiles(root)
    base = files.read_text("a.txt").revision

    first, second = await asyncio.gather(
        files.save_text("a.txt", "A\n", base),
        files.save_text("a.txt", "B\n", base),
    )
    results = [first, second]
    winners = [result for result in results if result.ok]
    losers = [result for result in results if not result.ok]
    assert len(winners) == 1
    assert len(losers) == 1
    assert losers[0].conflict.disk_revision == winners[0].revision
    assert (root / "a.txt").read_bytes() in (b"A\n", b"B\n")


async def test_save_preserves_mode(tmp_path):
    root = _root(tmp_path)
    target = root / "a.txt"
    target.write_bytes(b"x")
    target.chmod(0o640)
    files = WorkspaceFiles(root)
    base = files.read_text("a.txt").revision

    result = await files.save_text("a.txt", "y\n", base)
    assert result.ok is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


async def test_noop_save_preserves_bytes_exactly(tmp_path):
    root = _root(tmp_path)
    original = b"line1\r\nline2\r\n   \r\n\t\r\n"
    (root / "a.txt").write_bytes(original)
    files = WorkspaceFiles(root)
    read = files.read_text("a.txt")
    assert read.text == original.decode("utf-8")
    assert read.line_ending == "CRLF"

    result = await files.save_text("a.txt", read.text, read.revision)
    assert result.ok is True
    assert (root / "a.txt").read_bytes() == original
    assert result.revision == read.revision


async def test_save_fallback_path_without_dir_fd(tmp_path, monkeypatch):
    """Platforms without dir_fd (e.g. Windows) keep the documented fallback."""
    from machina.workspace import files as files_module

    monkeypatch.setattr(files_module, "_dir_fd_supported", lambda: False)
    root = _root(tmp_path)
    (root / "a.txt").write_bytes(b"one\n")
    files = WorkspaceFiles(root)
    base = files.read_text("a.txt").revision

    saved = await files.save_text("a.txt", "two\n", base)
    assert saved.ok is True
    assert (root / "a.txt").read_bytes() == b"two\n"

    # The fallback still refuses a symlinked parent that escapes the root.
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linkdir").symlink_to(outside)
    with pytest.raises(WorkspaceError) as error:
        await files.save_text("linkdir/x.txt", "x", None)
    assert error.value.reason == "denied"
    assert not (outside / "x.txt").exists()


async def test_save_failure_leaves_original_and_no_temp(tmp_path, monkeypatch):
    """A mid-commit failure never destroys the original or leaves a temp file."""
    from machina.workspace import files as files_module

    root = _root(tmp_path)
    target = root / "a.txt"
    target.write_bytes(b"original\n")
    files = WorkspaceFiles(root)
    base = files.read_text("a.txt").revision

    def boom(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(files_module.os, "fsync", boom)
    with pytest.raises(WorkspaceError) as error:
        await files.save_text("a.txt", "replacement\n", base)
    assert error.value.reason == "denied"
    monkeypatch.undo()

    assert target.read_bytes() == b"original\n"
    assert [entry.name for entry in root.iterdir()] == ["a.txt"]


async def test_save_creates_new_file_and_refuses_unexpected_existing(tmp_path):
    root = _root(tmp_path)
    files = WorkspaceFiles(root)
    created = await files.save_text("new.txt", "hi\n", None)
    assert created.ok is True
    assert (root / "new.txt").read_bytes() == b"hi\n"
    # New files are owner-only (the agent file writer uses 0600 too); a
    # permissive umask must not widen workspace content.
    assert stat.S_IMODE((root / "new.txt").stat().st_mode) == 0o600

    blocked = await files.save_text("new.txt", "bye\n", None)
    assert blocked.ok is False
    assert blocked.conflict.reason == "file-exists"
    assert (root / "new.txt").read_bytes() == b"hi\n"

    # Saving a file that vanished after it was read conflicts, not recreates.
    base = files.read_text("new.txt").revision
    (root / "new.txt").unlink()
    vanished = await files.save_text("new.txt", "zombie\n", base)
    assert vanished.ok is False
    assert vanished.conflict.reason == "missing-on-disk"
    assert not (root / "new.txt").exists()


async def test_save_refuses_read_only_target(tmp_path):
    root = _root(tmp_path)
    target = root / "ro.txt"
    target.write_bytes(b"x")
    target.chmod(0o444)
    try:
        files = WorkspaceFiles(root)
        read = files.read_text("ro.txt")
        assert read.read_only is True
        with pytest.raises(WorkspaceError) as error:
            await files.save_text("ro.txt", "y\n", read.revision)
        assert error.value.reason == "denied"
        assert "read-only" in error.value.message
        assert target.read_bytes() == b"x"
    finally:
        target.chmod(0o644)


async def test_save_refuses_hardlinked_target(tmp_path):
    root = _root(tmp_path)
    (root / "a.txt").write_bytes(b"x")
    os.link(root / "a.txt", root / "b.txt")
    files = WorkspaceFiles(root)
    read = files.read_text("a.txt")
    assert read.read_only is True

    with pytest.raises(WorkspaceError) as error:
        await files.save_text("a.txt", "y\n", read.revision)
    assert error.value.reason == "denied"
    assert "hard links" in error.value.message
    assert (root / "a.txt").read_bytes() == b"x"
    assert (root / "b.txt").read_bytes() == b"x"


async def test_save_through_symlinked_parent_outside_refused(tmp_path):
    root = _root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linkdir").symlink_to(outside)
    files = WorkspaceFiles(root)
    with pytest.raises(WorkspaceError) as error:
        await files.save_text("linkdir/x.txt", "x", None)
    assert error.value.reason == "denied"
    assert not (outside / "x.txt").exists()


def test_workspace_root_is_realpath_resolved(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.txt").write_text("x")
    link = tmp_path / "link"
    link.symlink_to(real)

    files = WorkspaceFiles(link)
    assert files.root == Path(os.path.realpath(real))
    assert files.read_text("a.txt").text == "x"

    with pytest.raises(WorkspaceError) as error:
        WorkspaceFiles(tmp_path / "missing-root")
    assert error.value.reason == "missing"


async def test_revision_registry_same_path_agent_user_conflict_shape(tmp_path):
    """Two cooperative writers on one path get a structured conflict."""
    root = _root(tmp_path)
    target = root / "shared.py"
    target.write_bytes(b"v1\n")
    agent_files = WorkspaceFiles(root)
    user_files = WorkspaceFiles(root)

    base = agent_files.read_text("shared.py").revision
    assert user_files.read_text("shared.py").revision == base

    first, second = await asyncio.gather(
        agent_files.save_text("shared.py", "v2-agent\n", base),
        user_files.save_text("shared.py", "v2-user\n", base),
    )
    assert first.ok is True
    assert second.ok is False
    conflict = second.conflict.as_dict()
    assert set(conflict) == {"diskRevision", "baseRevision", "reason"}
    assert conflict["diskRevision"] == first.revision
    assert conflict["baseRevision"] == base
    assert conflict["reason"] == "revision-mismatch"
    assert target.read_bytes() == b"v2-agent\n"


# ─── document mirrors ────────────────────────────────────────────────────────
def test_documents_versions_strictly_increasing_and_resync():
    store = DocumentMirrorStore(epoch=1)
    uri = "file:///ws/a.py"
    opened = store.open_document(uri, "python", 1, "one\n", epoch=1)
    assert opened.accepted_version == 1
    assert opened.server_generation == 1
    assert opened.state == "ready"

    # A reopen at a stale version is rejected without replacing the buffer.
    stale = store.open_document(uri, "python", 1, "stale\n", epoch=1)
    assert stale.accepted_version == 1
    assert store.snapshot(uri)["text"] == "one\n"

    # A strictly higher version reopens (replaces) the mirror.
    reopened = store.open_document(uri, "python", 5, "reopened\n", epoch=1)
    assert reopened.accepted_version == 5
    assert store.snapshot(uri)["text"] == "reopened\n"

    changed = store.change_document(
        uri,
        5,
        6,
        [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 3},
                },
                "text": "REO",
            }
        ],
        epoch=1,
    )
    assert changed.accepted_version == 6
    assert changed.resync is False
    assert store.snapshot(uri)["text"] == "REOpened\n"

    # Version gap / stale base demands resync at the current version.
    gap = store.change_document(uri, 5, 7, [], epoch=1)
    assert gap.resync is True
    assert gap.accepted_version == 6
    non_increasing = store.change_document(uri, 6, 6, [], epoch=1)
    assert non_increasing.resync is True
    unknown = store.change_document("file:///ws/other.py", 1, 2, [], epoch=1)
    assert unknown.resync is True
    assert unknown.accepted_version == 0


def test_document_change_utf16_positions_and_multiline():
    store = DocumentMirrorStore(epoch=1)
    uri = "file:///ws/u.txt"
    store.open_document(uri, "plaintext", 1, "a😀b\nsecond\n", epoch=1)

    # Replace the surrogate-pair emoji (UTF-16 units 1..3) with "X".
    result = store.change_document(
        uri,
        1,
        2,
        [
            {
                "range": {
                    "start": {"line": 0, "character": 1},
                    "end": {"line": 0, "character": 3},
                },
                "text": "X",
            }
        ],
        epoch=1,
    )
    assert result.accepted_version == 2
    assert store.snapshot(uri)["text"] == "aXb\nsecond\n"

    # Multi-line replacement across CRLF text.
    store.open_document("file:///ws/crlf.txt", "plaintext", 1, "one\r\ntwo\r\n", epoch=1)
    replaced = store.change_document(
        "file:///ws/crlf.txt",
        1,
        2,
        [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 1, "character": 3},
                },
                "text": "MERGED",
            }
        ],
        epoch=1,
    )
    assert replaced.accepted_version == 2
    assert store.snapshot("file:///ws/crlf.txt")["text"] == "MERGED\r\n"


def test_document_budgets_per_file_and_total():
    store = DocumentMirrorStore(epoch=1, per_file_bytes=10, total_bytes=15)
    with pytest.raises(WorkspaceError) as per_file:
        store.open_document("file:///ws/a.txt", "plaintext", 1, "x" * 11, epoch=1)
    assert per_file.value.reason == "too-large"

    store.open_document("file:///ws/a.txt", "plaintext", 1, "x" * 10, epoch=1)
    store.open_document("file:///ws/b.txt", "plaintext", 1, "y" * 5, epoch=1)
    with pytest.raises(WorkspaceError) as total:
        store.open_document("file:///ws/c.txt", "plaintext", 1, "z", epoch=1)
    assert total.value.reason == "too-large"

    # Growth past the per-file cap during a change is refused too.
    with pytest.raises(WorkspaceError) as grown:
        store.change_document(
            "file:///ws/a.txt",
            1,
            2,
            [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0},
                    },
                    "text": "0123456789",
                }
            ],
            epoch=1,
        )
    assert grown.value.reason == "too-large"
    assert store.snapshot("file:///ws/a.txt")["text"] == "x" * 10

    # Replacing the same URI only counts the delta against the budget.
    replaced = store.open_document("file:///ws/b.txt", "plaintext", 2, "z" * 5, epoch=1)
    assert replaced.accepted_version == 2


def test_document_stale_epoch_rejected():
    store = DocumentMirrorStore(epoch=2)
    with pytest.raises(WorkspaceError) as error:
        store.open_document("file:///ws/a.txt", "python", 1, "", epoch=1)
    assert error.value.reason == "denied"
    assert "epoch" in error.value.message


# ─── service: identity, documents, intelligence ─────────────────────────────
def test_service_identity_realpath_binding_and_capabilities(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    service, identity, _ = _service(link)
    assert identity.resolved_root == os.path.realpath(real)
    assert identity.root_label == "real"
    assert identity.trusted is True
    assert identity.epoch == 1
    assert set(identity.capabilities) >= {
        "list",
        "read",
        "save",
        "documentOpen",
        "intelligence",
        "lspControl",
    }
    assert identity.workspace_id.startswith("ws_")
    assert identity.client_id.startswith("cl_")

    untrusted_service, untrusted_identity, _ = _service(real, trust=False)
    assert untrusted_identity.trusted is False
    assert "intelligence" not in untrusted_identity.capabilities
    assert "lspControl" not in untrusted_identity.capabilities
    assert "save" in untrusted_identity.capabilities
    assert untrusted_service.workspace_id == untrusted_identity.workspace_id


def test_service_document_epochs_reopen_and_close(tmp_path):
    root = _root(tmp_path)
    service, identity, _ = _service(root)
    uri = (root / "a.py").as_uri()

    opened = service.document_open(uri, "python", 1, "x = 1\n", epoch=service.epoch)
    assert opened.accepted_version == 1
    assert opened.server_generation == identity.epoch
    snapshot = service.document_snapshot(uri)
    assert snapshot is not None and snapshot["text"] == "x = 1\n"
    assert snapshot["epoch"] == identity.epoch

    with pytest.raises(WorkspaceError) as stale_epoch:
        service.document_open(uri, "python", 2, "y\n", epoch=99)
    assert stale_epoch.value.reason == "denied"

    reopened = service.open(str(root), True)
    assert reopened.epoch == identity.epoch + 1
    assert service.document_snapshot(uri) is None  # mirrors dropped on reopen
    with pytest.raises(WorkspaceError) as old_epoch:
        service.document_open(uri, "python", 1, "z\n", epoch=identity.epoch)
    assert old_epoch.value.reason == "denied"

    assert service.close() == {"closed": True}
    assert service.workspace_id is None
    with pytest.raises(WorkspaceError):
        service.list("")
    with pytest.raises(WorkspaceError):
        service.read(uri)


def test_service_rejects_relative_and_non_file_uris(tmp_path):
    service, _, _ = _service(_root(tmp_path))
    for uri in ("file:relative.txt", "https://example.com/x", "file://remotehost/x"):
        with pytest.raises(WorkspaceError) as error:
            service.read(uri)
        assert error.value.reason == "denied", uri


async def test_service_save_emits_file_changed_event(tmp_path):
    root = _root(tmp_path)
    target = root / "a.txt"
    target.write_text("x\n")
    service, identity, events = _service(root)
    uri = target.as_uri()

    read = service.read(uri)
    result = await service.save(uri, "y\n", read.revision, epoch=service.epoch)
    assert result.ok is True
    assert len(events) == 1
    event = events[0]
    assert event == {
        "type": "workspace.fileChanged",
        "workspaceId": identity.workspace_id,
        "epoch": identity.epoch,
        "uri": read.uri,
        "revision": result.revision,
        "origin": "save",
    }
    # A conflicting save emits nothing.
    await service.save(uri, "z\n", read.revision, epoch=service.epoch)
    assert len(events) == 1


async def test_intelligence_without_manager_is_truthful(tmp_path):
    root = _root(tmp_path)
    service, _, _ = _service(root)
    uri = (root / "a.py").as_uri()

    result = await service.intelligence(
        "completion", uri=uri, position={"line": 0, "character": 0}
    )
    payload = result.as_dict()
    assert payload["state"] == "missing"
    assert payload["freshness"] == "unavailable"
    assert payload["result"] is None
    assert payload["serverGeneration"] is None
    assert payload["source"] == "disk"
    assert payload["operation"] == "completion"

    with pytest.raises(WorkspaceError) as error:
        await service.intelligence("rename")
    assert error.value.reason == "denied"


async def test_intelligence_dispatches_to_manager_and_is_honest_on_failure(tmp_path):
    root = _root(tmp_path)
    service, _, _ = _service(root)
    uri = (root / "a.py").as_uri()
    calls: list[dict] = []

    class ReadyManager:
        async def intelligence(self, request):
            calls.append(request)
            return {
                "state": "ready",
                "freshness": "versioned",
                "truncated": False,
                "result": [{"message": "ok"}],
                "serverGeneration": 3,
            }

        async def status(self):
            return {
                "servers": [
                    {"profile": "python", "languageId": "python", "state": "ready", "generation": 3}
                ]
            }

        def state_for_language(self, language_id):
            return "ready"

    service.set_lsp_manager(ReadyManager())
    service.document_open(uri, "python", 1, "x = 1\n", epoch=service.epoch)
    dispatched = (await service.intelligence("diagnostics", uri=uri)).as_dict()
    assert dispatched["state"] == "ready"
    assert dispatched["freshness"] == "versioned"
    assert dispatched["serverGeneration"] == 3
    assert dispatched["source"] == "editor-buffer"
    assert dispatched["documentVersion"] == 1
    assert calls[0]["operation"] == "diagnostics"
    assert (await service.lsp_status())["servers"][0]["profile"] == "python"
    assert service.document_open(uri, "python", 2, "y\n", epoch=service.epoch).state == "ready"

    class BrokenManager:
        def intelligence(self, request):
            raise RuntimeError("server crashed")

        def status(self):
            raise RuntimeError("no status")

    service.set_lsp_manager(BrokenManager())
    failed = (await service.intelligence("hover", uri=uri)).as_dict()
    assert failed["state"] == "failed"
    assert failed["result"] is None
    assert (await service.lsp_status()) == {"servers": []}


async def test_lsp_control_requires_trust_and_reports_missing(tmp_path):
    root = _root(tmp_path)
    untrusted, _, _ = _service(root, trust=False)
    with pytest.raises(WorkspaceError) as denied:
        await untrusted.lsp_control("start", "python", epoch=untrusted.epoch)
    assert denied.value.reason == "denied"

    trusted, _, _ = _service(root, trust=True)
    missing = await trusted.lsp_control("start", "python", epoch=trusted.epoch)
    assert missing["state"] == "missing"
    assert missing["action"] == "start"
    assert missing["profile"] == "python"
    with pytest.raises(WorkspaceError):
        await trusted.lsp_control("install", "python", epoch=trusted.epoch)
    assert await trusted.lsp_status() == {"servers": []}


async def test_mutating_operations_require_epoch_and_client_binding(tmp_path):
    """F-19: mutations are fenced by a REQUIRED epoch and an optional client id.

    A stale caller that merely omits ``epoch`` must not be treated as current,
    while reads stay backward-compatible (an omitted epoch still means the
    live workspace).
    """
    root = _root(tmp_path)
    target = root / "a.py"
    target.write_text("x = 1\n")
    service, identity, _ = _service(root)
    uri = target.as_uri()
    read = service.read(uri)

    # Reads remain compatible with clients that never learned about epochs.
    assert service.list("").entries
    assert service.read(uri).text == "x = 1\n"
    quick = await service.quick_open("a.py")
    assert quick.workspace_epoch == identity.epoch  # reads still default to current
    assert (await service.intelligence("symbols", uri=uri)).state == "missing"

    # Missing epoch: every mutating operation denies instead of assuming current.
    with pytest.raises(WorkspaceError) as missing_save:
        await service.save(uri, "y\n", read.revision)
    assert missing_save.value.reason == "denied"
    assert "epoch is required" in missing_save.value.message

    with pytest.raises(WorkspaceError) as missing_open:
        service.document_open(uri, "python", 1, "x = 1\n")
    assert missing_open.value.reason == "denied"

    with pytest.raises(WorkspaceError) as missing_change:
        service.document_change(uri, 1, 2, [])
    assert missing_change.value.reason == "denied"

    with pytest.raises(WorkspaceError) as missing_close:
        service.document_close(uri)
    assert missing_close.value.reason == "denied"

    with pytest.raises(WorkspaceError) as missing_control:
        await service.lsp_control("start", "python")
    assert missing_control.value.reason == "denied"

    # Stale epoch values still deny.
    stale = identity.epoch + 1
    with pytest.raises(WorkspaceError) as stale_save:
        await service.save(uri, "y\n", read.revision, epoch=stale)
    assert stale_save.value.reason == "denied"
    assert "stale" in stale_save.value.message
    with pytest.raises(WorkspaceError) as stale_open:
        service.document_open(uri, "python", 1, "x = 1\n", epoch=stale)
    assert stale_open.value.reason == "denied"
    with pytest.raises(WorkspaceError) as stale_change:
        service.document_change(uri, 1, 2, [], epoch=stale)
    assert stale_change.value.reason == "denied"
    with pytest.raises(WorkspaceError) as stale_close:
        service.document_close(uri, epoch=stale)
    assert stale_close.value.reason == "denied"
    with pytest.raises(WorkspaceError) as stale_control:
        await service.lsp_control("start", "python", epoch=stale)
    assert stale_control.value.reason == "denied"

    # A supplied client id must match the identity minted by open().
    with pytest.raises(WorkspaceError) as wrong_client:
        await service.save(
            uri, "y\n", read.revision, epoch=identity.epoch, client_id="cl_not_this_one"
        )
    assert wrong_client.value.reason == "denied"
    assert "client id" in wrong_client.value.message
    with pytest.raises(WorkspaceError) as wrong_open_client:
        service.document_open(
            uri, "python", 1, "x = 1\n", epoch=identity.epoch, client_id="cl_not_this_one"
        )
    assert wrong_open_client.value.reason == "denied"
    with pytest.raises(WorkspaceError) as wrong_control_client:
        await service.lsp_control(
            "start", "python", epoch=identity.epoch, client_id="cl_not_this_one"
        )
    assert wrong_control_client.value.reason == "denied"

    # Current epoch works, with or without the client id (stdio may omit it).
    saved = await service.save(uri, "y\n", read.revision, epoch=identity.epoch)
    assert saved.ok is True
    opened = service.document_open(
        uri, "python", 1, "x = 1\n", epoch=identity.epoch, client_id=identity.client_id
    )
    assert opened.accepted_version == 1
    assert (
        service.document_change(
            uri, 1, 2, [], epoch=identity.epoch, client_id=identity.client_id
        ).accepted_version
        == 2
    )
    assert service.document_close(uri, epoch=identity.epoch)["closed"] is True
    control = await service.lsp_control("start", "python", epoch=identity.epoch)
    assert control["state"] == "missing"  # no manager attached: honest, not denied
