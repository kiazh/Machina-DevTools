"""Combined #48: duplicate headings never bind ambiguously to the first match."""

from __future__ import annotations

from machina.skills.metadata import (
    CODE_SECTION_DUPLICATE_HEADING,
    SectionMetadata,
    SkillMetadata,
    bind_section_anchors,
)


def test_duplicate_headings_leave_sections_unbound():
    metadata = SkillMetadata(
        skill_id="demo",
        sections=(
            SectionMetadata(id="first", heading="Setup"),
            SectionMetadata(id="second", heading="Setup"),
        ),
    )
    content = "# Setup\n\nA\n\n# Setup\n\nB\n"
    bound, diagnostics = bind_section_anchors(metadata, content)

    assert all(section.anchor is None for section in bound.sections)
    assert any(d.code == CODE_SECTION_DUPLICATE_HEADING for d in diagnostics)
