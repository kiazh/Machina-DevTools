"""Combined #5 / original #36: skill creation must not follow a directory symlink.

The pre-fix path called ``mkdir(parents=True, exist_ok=True)`` and then opened
``SKILL.md`` under the resulting path. A pre-existing ``<root>/<name>`` symlink
pointing outside the root made both steps operate on the outside directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from machina.skills.registry import SkillRegistry


def _symlink_dir(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:  # pragma: no cover - platform
        pytest.skip(f"directory symlinks unavailable: {error}")


def test_create_skill_refuses_a_preexisting_directory_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "skills"
    root.mkdir()
    _symlink_dir(root / "demo", outside)

    registry = SkillRegistry(root)
    with pytest.raises(ValueError, match="symlink|already exists|not accessible"):
        registry.create_skill("demo", "Deploy steps", "Steps to deploy safely")

    assert not (outside / "SKILL.md").exists(), "creation escaped the skill root"


def test_create_skill_still_creates_a_fresh_directory(tmp_path):
    root = tmp_path / "skills"
    registry = SkillRegistry(root)
    skill = registry.create_skill("demo", "Deploy steps", "Steps to deploy safely")

    assert Path(skill.file_path) == root / "demo" / "SKILL.md"
    assert Path(skill.file_path).is_file()
    assert os.stat(root / "demo").st_mode & 0o170000 == 0o040000  # real directory
