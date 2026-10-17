"""Combined #12: enabling/disabling a repo-installed skill refreshes its manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from machina.skills.registry import SkillRegistry


def test_set_enabled_refreshes_the_install_manifest(tmp_path):
    root = tmp_path / "skills"
    registry = SkillRegistry(root)
    skill = registry.create_skill("demo-toggle", "Deploy steps", "Steps to deploy safely.")
    skill_file = Path(skill.file_path)
    manifest_file = skill_file.parent / ".install-manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "files": {
                    f"{skill_file.parent.name}/{skill_file.name}": hashlib.sha256(
                        skill_file.read_bytes()
                    ).hexdigest()
                }
            }
        ),
        encoding="utf-8",
    )

    fresh = SkillRegistry(root)
    fresh.load_all()
    assert fresh.set_enabled("demo-toggle", False) is not None

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    recorded = manifest["files"][f"{skill_file.parent.name}/{skill_file.name}"]
    assert recorded == hashlib.sha256(skill_file.read_bytes()).hexdigest()
