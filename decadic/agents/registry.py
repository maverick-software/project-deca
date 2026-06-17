"""Registry of active agent runtimes."""

from __future__ import annotations

from pathlib import Path

from decadic.agents.runtime import AgentRuntime
from decadic.nn.faculties import CognitionFaculties
from decadic.nn.plastic import PlasticityFlags


class AgentRegistry:
    """Process-wide agent lookup."""

    def __init__(self, data_dir: Path) -> None:
        self._agents: dict[str, AgentRuntime] = {}
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # UI-set neuroplasticity defaults for newly created agents. None -> each
        # new agent reads the process env (the historical behaviour).
        self.new_agent_flags: PlasticityFlags | None = None
        # UI-set cognitive-faculty defaults for newly created agents. None -> each
        # new agent reads the process env (inherent-on defaults).
        self.new_agent_faculties: CognitionFaculties | None = None

    def create_agent(
        self,
        agent_id: str,
        flags: PlasticityFlags | None = None,
        faculties: CognitionFaculties | None = None,
    ) -> AgentRuntime:
        if agent_id in self._agents:
            raise KeyError(agent_id)
        db_path = self.data_dir / f"agent_{agent_id}_episodes.sqlite"
        graph_db_path = self.data_dir / f"agent_{agent_id}_graph.sqlite"
        agent = AgentRuntime(
            agent_id,
            episodic_db_path=db_path,
            graph_db_path=graph_db_path,
            flags=flags if flags is not None else self.new_agent_flags,
            faculties=faculties if faculties is not None else self.new_agent_faculties,
        )
        self._agents[agent_id] = agent
        agent.ensure_cycle_worker()
        return agent

    def get(self, agent_id: str) -> AgentRuntime | None:
        return self._agents.get(agent_id)

    def ids(self) -> list[str]:
        return list(self._agents.keys())

    def require(self, agent_id: str) -> AgentRuntime:
        agent = self.get(agent_id)
        if agent is None:
            raise KeyError(agent_id)
        return agent

    async def delete_agent(self, agent_id: str) -> None:
        agent = self._agents.pop(agent_id, None)
        if agent is not None:
            await agent.stop()
