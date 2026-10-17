"""Agent definitions cannot collide with human approval transport principals."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from machina.core.agent_def import AgentDef, AgentRegistry, agent_registry
from machina.gateway.errors import INVALID_PARAMS, RpcError
from machina.gateway.features.agents import agents_save

RESERVED = ("tui", "http", "headless", "mcp-management", "mcp-oauth-management")


@pytest.mark.parametrize("name", RESERVED)
async def test_gateway_rejects_reserved_name_before_saving(monkeypatch, name):
    save = AsyncMock()
    monkeypatch.setattr(agent_registry, "save", save)
    with pytest.raises(RpcError) as error:
        await agents_save(SimpleNamespace(require_db=lambda: None), {"name": name})
    assert error.value.code == INVALID_PARAMS
    save.assert_not_awaited()


@pytest.mark.parametrize("name", RESERVED)
async def test_registry_rejects_reserved_name_before_database_write(name):
    connection = SimpleNamespace(fetchrow=AsyncMock())
    with pytest.raises(ValueError, match="reserved"):
        await AgentRegistry().save_on_connection(AgentDef(name=name), connection)
    connection.fetchrow.assert_not_awaited()


@pytest.mark.parametrize("name", RESERVED)
@pytest.mark.parametrize("soulspec", [False, True])
def test_yaml_skips_reserved_names_but_keeps_ordinary_definitions(
    tmp_path, monkeypatch, name, soulspec
):
    monkeypatch.setenv("MACHINA_AGENTS_DIR", str(tmp_path))
    extra = "persona:\n  identity: helpful\n" if soulspec else ""
    (tmp_path / "reserved.yaml").write_text(f"name: {name}\n{extra}")
    (tmp_path / "ordinary.yaml").write_text("name: helper-http\nsystem_prompt: helpful\n")
    assert set(AgentRegistry()._file_definitions()) == {"helper-http"}
