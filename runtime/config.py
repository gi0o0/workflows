"""Lee un workflow exportado (`workflow.json`) y las variables de entorno que necesita.

El archivo lo genera Agents Platform al exportar (Workflows → Export). Lleva el grafo
completo, con los prompts ya armados; las API keys nunca van en él: se leen del entorno
(`.env` de la carpeta del workflow o variables del sistema).
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

SCHEMA_VERSION = 1

# Variable de entorno de la API key de cada proveedor.
PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


class ConfigError(Exception):
    """El workflow o el entorno no permiten ejecutar (mensaje para quien lo despliega)."""


@dataclass
class ModelConfig:
    provider: str
    model: str
    temperature: float = 0
    max_tokens: int | None = None


@dataclass
class ToolSelection:
    server: str
    names: list[str]


@dataclass
class KnowledgeDocument:
    name: str
    # Ruta del texto del documento, relativa a la carpeta del workflow.
    file: str


@dataclass
class KnowledgeTool:
    """Un bloque Knowledge: una tool de búsqueda del agente en uno o más documentos."""

    name: str
    tool: str
    description: str
    documents: list[KnowledgeDocument]
    embedding_provider: str
    embedding_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int


@dataclass
class AgentNode:
    key: str
    name: str
    description: str
    model: ModelConfig
    system_prompt: str
    recursion_limit: int
    tools: list[ToolSelection]
    next: str | None
    knowledge: list[KnowledgeTool] = field(default_factory=list)


@dataclass
class Route:
    id: str
    name: str
    target: str


@dataclass
class RouterNode:
    key: str
    name: str
    model: ModelConfig
    system_prompt: str
    routes: list[Route]


@dataclass
class Delegate:
    agent: str
    tool: str
    description: str


@dataclass
class SupervisorNode:
    key: str
    name: str
    model: ModelConfig
    system_prompt: str
    max_delegations: int
    delegates: list[Delegate]
    next: str | None


@dataclass
class RagNode:
    """Un RAG Agent: responde con los fragmentos de su documento más parecidos a la pregunta."""

    key: str
    name: str
    model: ModelConfig
    system_prompt: str
    document_name: str
    # Ruta del texto del documento, relativa a la carpeta del workflow.
    document_file: str
    embedding_provider: str
    embedding_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    description: str = ""
    # "request": busca con el mensaje del usuario; "previous": con la respuesta del paso anterior.
    query_source: str = "request"
    # Reescribe las preguntas de seguimiento con la conversación antes de buscar.
    rewrite_query: bool = True
    next: str | None = None


@dataclass
class McpServer:
    key: str
    name: str
    # Si la URL lleva parámetros (puede incluir una key), viene de la variable `url_env`.
    url: str | None
    url_env: str | None
    api_key_env: str | None

    def resolved_url(self) -> str:
        return self.url or os.environ[self.url_env]


@dataclass
class Workflow:
    name: str
    version: int
    start: str
    agents: dict[str, AgentNode] = field(default_factory=dict)
    routers: dict[str, RouterNode] = field(default_factory=dict)
    supervisors: dict[str, SupervisorNode] = field(default_factory=dict)
    rags: dict[str, RagNode] = field(default_factory=dict)
    mcp_servers: dict[str, McpServer] = field(default_factory=dict)
    # Carpeta del workflow (ahí están sus documentos y el índice `.index/` de los RAG Agents).
    folder: Path = field(default_factory=Path)

    def name_of(self, key: str) -> str:
        node = self.agents.get(key) or self.routers.get(key) or self.supervisors.get(key) or self.rags.get(key)
        return node.name if node else key


def _model(raw: dict) -> ModelConfig:
    return ModelConfig(
        provider=raw["provider"],
        model=raw["model"],
        temperature=raw.get("temperature") or 0,
        max_tokens=raw.get("maxTokens"),
    )


def _knowledge(folder: Path, agent_name: str, raw: dict) -> KnowledgeTool:
    documents = [KnowledgeDocument(name=d["name"], file=d["file"]) for d in raw["documents"]]
    for document in documents:
        if not (folder / document.file).is_file():
            raise ConfigError(f"Falta el documento {document.file} de \"{raw['name']}\" (agente \"{agent_name}\").")
    return KnowledgeTool(
        name=raw["name"],
        tool=raw["tool"],
        description=raw.get("description") or "",
        documents=documents,
        embedding_provider=raw["embedding"]["provider"],
        embedding_model=raw["embedding"]["model"],
        top_k=raw.get("topK") or 4,
        chunk_size=raw.get("chunkSize") or 1000,
        chunk_overlap=raw.get("chunkOverlap") or 0,
    )


def load_workflow(folder: str | Path) -> Workflow:
    """Carga `<folder>/workflow.json` y `<folder>/.env` (si existe)."""
    folder = Path(folder)
    path = folder / "workflow.json"
    if not path.is_file():
        raise ConfigError(f"No se encontró {path}.")
    load_dotenv(folder / ".env")

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schemaVersion") != SCHEMA_VERSION:
        raise ConfigError(
            f"workflow.json tiene la versión de formato {data.get('schemaVersion')}; "
            f"este runtime entiende la {SCHEMA_VERSION}. Exporta de nuevo el workflow."
        )

    workflow = Workflow(
        name=data["workflow"]["name"], version=data["workflow"]["version"], start=data["start"], folder=folder
    )
    for key, node in data["nodes"].items():
        if node["type"] == "agent":
            knowledge = [_knowledge(folder, node["name"], raw) for raw in node.get("knowledge") or []]
            workflow.agents[key] = AgentNode(
                key=key,
                name=node["name"],
                description=node.get("description") or "",
                model=_model(node),
                system_prompt=node["systemPrompt"],
                recursion_limit=node.get("recursionLimit") or 25,
                tools=[ToolSelection(server=t["server"], names=t["names"]) for t in node.get("tools", [])],
                next=node.get("next"),
                knowledge=knowledge,
            )
        elif node["type"] == "router":
            workflow.routers[key] = RouterNode(
                key=key,
                name=node["name"],
                model=_model(node),
                system_prompt=node["systemPrompt"],
                routes=[Route(id=r["id"], name=r["name"], target=r["target"]) for r in node["routes"]],
            )
        elif node["type"] == "supervisor":
            workflow.supervisors[key] = SupervisorNode(
                key=key,
                name=node["name"],
                model=_model(node),
                system_prompt=node["systemPrompt"],
                max_delegations=node.get("maxDelegations") or 6,
                delegates=[
                    Delegate(agent=d["agent"], tool=d["tool"], description=d["description"])
                    for d in node["delegates"]
                ],
                next=node.get("next"),
            )
        elif node["type"] == "rag":
            document = node["document"]
            if not (folder / document["file"]).is_file():
                raise ConfigError(f"Falta el documento {document['file']} del RAG Agent \"{node['name']}\".")
            workflow.rags[key] = RagNode(
                key=key,
                name=node["name"],
                model=_model(node),
                system_prompt=node["systemPrompt"],
                document_name=document["name"],
                document_file=document["file"],
                embedding_provider=node["embedding"]["provider"],
                embedding_model=node["embedding"]["model"],
                top_k=node.get("topK") or 4,
                chunk_size=node.get("chunkSize") or 1000,
                chunk_overlap=node.get("chunkOverlap") or 0,
                description=node.get("description") or "",
                query_source=node.get("querySource") or "request",
                rewrite_query=node.get("rewriteQuery", True),
                next=node.get("next"),
            )
    for key, server in data.get("mcpServers", {}).items():
        workflow.mcp_servers[key] = McpServer(
            key=key,
            name=server["name"],
            url=server.get("url"),
            url_env=server.get("urlEnv"),
            api_key_env=server.get("apiKeyEnv"),
        )

    check_environment(workflow)
    return workflow


def check_environment(workflow: Workflow) -> None:
    """Falla al arrancar (no a mitad de una petición) si falta alguna API key."""
    providers = {
        node.model.provider
        for node in [
            *workflow.agents.values(),
            *workflow.routers.values(),
            *workflow.supervisors.values(),
            *workflow.rags.values(),
        ]
    }
    providers |= {node.embedding_provider for node in workflow.rags.values()}
    providers |= {k.embedding_provider for node in workflow.agents.values() for k in node.knowledge}
    missing = [PROVIDER_ENV.get(p, f"{p.upper()}_API_KEY") for p in sorted(providers)]
    missing = [name for name in missing if not os.getenv(name)]
    missing += [
        name
        for server in workflow.mcp_servers.values()
        for name in (server.url_env, server.api_key_env)
        if name and not os.getenv(name)
    ]
    if missing:
        raise ConfigError("Faltan variables de entorno: " + ", ".join(missing) + ". Revisa el .env del workflow.")


def provider_key(provider: str) -> str:
    return os.environ[PROVIDER_ENV.get(provider, f"{provider.upper()}_API_KEY")]
