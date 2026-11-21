"""Ejecuta un workflow exportado con LangGraph, igual que Agents Platform.

- Agente: un `create_agent` con su prompt y las tools elegidas de sus servidores MCP.
- Secuencia: el primer bloque recibe la conversación; los siguientes, la petición
  original junto con la respuesta del agente anterior.
- Router: un LLM con salida estructurada `{route, reason}` elige una ruta (o Default si
  la respuesta no se entiende) y el flujo sigue por esa rama, sin volver.
- Supervisor: un `create_agent` cuyas tools son `delegate_to_<agente>(task)`; cada una
  ejecuta ese agente aislado con solo la tarea y devuelve su respuesta. Pasado
  `maxDelegations`, las tools responden que ya no se puede delegar.
- RAG Agent: busca en su documento los fragmentos más parecidos a la pregunta
  (runtime/rag.py, Chroma en `.index/`) y responde con ellos como contexto. Sin tools. Va en
  el flujo como un agente (o delegado de un supervisor, que le pasa la tarea como pregunta).
  Las preguntas de seguimiento se reescriben con la conversación antes de buscar.
- Knowledge: una tool `search_<nombre>(query)` del agente al que está conectado, que busca en
  uno o más documentos; el agente decide cuándo usarla, como las tools de sus MCP.
"""

import operator
import os
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, create_model

from runtime.config import (
    AgentNode,
    KnowledgeTool,
    ModelConfig,
    RagNode,
    RouterNode,
    SupervisorNode,
    Workflow,
    provider_key,
)
from runtime.rag import context_prompt, search_knowledge
from runtime.rag import search as search_document

DEFAULT_ROUTE = "default"
LIMIT_REACHED = (
    "Límite de delegaciones alcanzado en esta ejecución: no delegues más. "
    "Responde al usuario con lo que ya tienes."
)


def chat_model(config: ModelConfig):
    api_key = provider_key(config.provider)
    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.model, api_key=api_key, temperature=config.temperature, max_tokens=config.max_tokens)
    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model, api_key=api_key, temperature=config.temperature, max_tokens=config.max_tokens)
    if config.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model, google_api_key=api_key, temperature=config.temperature, max_output_tokens=config.max_tokens
        )
    raise ValueError(f"Proveedor no soportado: {config.provider}")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


async def run_agent(node: AgentNode, messages: list[dict], tools: list) -> str:
    agent = create_agent(model=chat_model(node.model), tools=tools, system_prompt=node.system_prompt)
    result = await agent.ainvoke({"messages": messages}, config={"recursion_limit": node.recursion_limit})
    return _text(result["messages"][-1].content)


REWRITE_PROMPT = (
    "Reescribe la última pregunta del usuario como una pregunta autocontenida para buscar en un "
    "documento, usando la conversación para completar lo que falte (de qué se habla, a quién se "
    "refiere). Si ya es autocontenida, devuélvela igual. Responde solo con la pregunta, en su idioma."
)


async def rewrite_question(node: RagNode, history: list[dict], question: str) -> str:
    try:
        llm = chat_model(ModelConfig(provider=node.model.provider, model=node.model.model, temperature=0))
        response = await llm.ainvoke(
            [{"role": "system", "content": REWRITE_PROMPT}, *history[-6:], {"role": "user", "content": question}]
        )
        return _text(response.content).strip() or question
    except Exception:
        return question


async def answer_rag(folder, node: RagNode, question: str, history: list[dict], messages: list[dict]):
    """Busca en el documento y responde. Devuelve (respuesta, fragmentos usados)."""
    query = await rewrite_question(node, history, question) if node.rewrite_query and history else question
    chunks = await search_document(folder, node, query)
    agent = create_agent(model=chat_model(node.model), tools=[], system_prompt=context_prompt(node, chunks))
    result = await agent.ainvoke({"messages": messages})
    return _text(result["messages"][-1].content), chunks


@dataclass
class Step:
    """Lo que hizo cada bloque en una ejecución (se devuelve junto con la respuesta)."""

    node: str
    name: str
    type: str
    output: str
    route: str | None = None
    reason: str | None = None
    parent: str | None = None
    task: str | None = None
    # RAG Agent: los fragmentos del documento que usó.
    sources: list[dict] | None = None


class State(TypedDict):
    request: str
    history: list[dict]
    last_output: str
    last_agent: str
    route: str
    outputs: Annotated[dict[str, str], operator.or_]


def compose_input(state: State) -> list[dict]:
    if not state["last_agent"]:
        return [*state["history"], {"role": "user", "content": state["request"]}]
    content = (
        f"Petición original del usuario:\n{state['request']}\n\n"
        f"Resultado del paso anterior ({state['last_agent']}):\n{state['last_output']}"
    )
    return [*state["history"], {"role": "user", "content": content}]


class DelegateInput(BaseModel):
    task: str = Field(description="La tarea completa y autocontenida para el agente (no ve la conversación).")


class KnowledgeQuery(BaseModel):
    query: str = Field(
        description="Qué buscar en los documentos: una pregunta o frase autocontenida, con los términos clave."
    )


def knowledge_tool(folder, knowledge: KnowledgeTool) -> StructuredTool:
    async def run(query: str) -> str:
        try:
            return await search_knowledge(folder, knowledge, query)
        except Exception as error:  # el agente ve el error y decide si sigue sin la búsqueda
            return f"Error al buscar en {knowledge.name}: {error}"

    return StructuredTool.from_function(
        coroutine=run, name=knowledge.tool, description=knowledge.description, args_schema=KnowledgeQuery
    )


class WorkflowRunner:
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self._tools: dict[str, list] | None = None

    async def _load_tools(self) -> dict[str, list]:
        """Conecta una vez con cada servidor MCP (y reutiliza sus tools entre ejecuciones)."""
        if self._tools is not None:
            return self._tools
        if not self.workflow.mcp_servers:
            self._tools = {}
            return self._tools
        config = {}
        for server in self.workflow.mcp_servers.values():
            api_key = os.getenv(server.api_key_env) if server.api_key_env else None
            config[server.key] = {
                "url": server.resolved_url().rstrip("/"),
                "transport": "streamable_http",
                "headers": {"Authorization": f"Bearer {api_key}"} if api_key else {},
            }
        client = MultiServerMCPClient(config)
        self._tools = {key: await client.get_tools(server_name=key) for key in self.workflow.mcp_servers}
        return self._tools

    def _agent_tools(self, node: AgentNode, tools_by_server: dict[str, list]) -> list:
        mcp_tools = [
            tool
            for selection in node.tools
            for tool in tools_by_server.get(selection.server, [])
            if tool.name in selection.names
        ]
        return [*mcp_tools, *(knowledge_tool(self.workflow.folder, k) for k in node.knowledge)]

    def _agent(self, node: AgentNode, tools: list, steps: list[Step]):
        async def run(state: State) -> dict:
            output = await run_agent(node, compose_input(state), tools)
            steps.append(Step(node=node.key, name=node.name, type="agent", output=output))
            return {"last_output": output, "last_agent": node.name, "outputs": {node.key: output}}

        return run

    def _rag(self, node: RagNode, steps: list[Step]):
        async def run(state: State) -> dict:
            from_previous = node.query_source == "previous" and bool(state["last_agent"])
            output, chunks = await answer_rag(
                self.workflow.folder,
                node,
                state["last_output"] if from_previous else state["request"],
                [] if from_previous else state["history"],
                compose_input(state),
            )
            steps.append(
                Step(
                    node=node.key,
                    name=node.name,
                    type="rag",
                    output=output,
                    sources=[{"fragment": c.index + 1, "score": c.score, "text": c.text} for c in chunks],
                )
            )
            return {"last_output": output, "last_agent": node.name, "outputs": {node.key: output}}

        return run

    def _router(self, node: RouterNode, steps: list[Step]):
        ids = tuple(route.id for route in node.routes)
        schema = create_model(
            "RouteDecision",
            route=(Literal[ids], Field(description="Id del camino elegido.")),
            reason=(str, Field(description="Por qué se eligió ese camino, en una frase.")),
        )

        async def run(state: State) -> dict:
            llm = chat_model(ModelConfig(provider=node.model.provider, model=node.model.model, temperature=0))
            response = await llm.with_structured_output(schema, include_raw=True).ainvoke(
                [SystemMessage(content=node.system_prompt), *compose_input(state)]
            )
            parsed = response.get("parsed")
            route = getattr(parsed, "route", None)
            reason = (getattr(parsed, "reason", None) or "").strip()
            if route not in ids:
                route, reason = DEFAULT_ROUTE, "El modelo no devolvió una ruta válida; se tomó Default."
            name = next(r.name for r in node.routes if r.id == route)
            steps.append(Step(node=node.key, name=node.name, type="router", output=name, route=name, reason=reason))
            return {"route": route}

        return run

    def _supervisor(self, node: SupervisorNode, tools_by_server: dict[str, list], steps: list[Step]):
        async def run(state: State) -> dict:
            count = 0

            def delegate_tool(delegate) -> StructuredTool:
                # Un delegado puede ser un agente o un RAG Agent (que busca con la tarea).
                agent = self.workflow.agents.get(delegate.agent) or self.workflow.rags[delegate.agent]

                async def call(task: str) -> str:
                    nonlocal count
                    if count >= node.max_delegations:
                        return LIMIT_REACHED
                    count += 1
                    messages = [{"role": "user", "content": task}]
                    try:
                        if isinstance(agent, RagNode):
                            output, _ = await answer_rag(self.workflow.folder, agent, task, [], messages)
                        else:
                            output = await run_agent(agent, messages, self._agent_tools(agent, tools_by_server))
                    except Exception as error:  # el error vuelve al supervisor, que decide
                        output = f"Error de {agent.name}: {error}"
                    kind = "rag" if isinstance(agent, RagNode) else "agent"
                    steps.append(
                        Step(node=agent.key, name=agent.name, type=kind, output=output, parent=node.key, task=task)
                    )
                    return output or "(El agente terminó sin respuesta.)"

                return StructuredTool.from_function(
                    coroutine=call, name=delegate.tool, description=delegate.description, args_schema=DelegateInput
                )

            supervisor = create_agent(
                model=chat_model(ModelConfig(provider=node.model.provider, model=node.model.model, temperature=0)),
                tools=[delegate_tool(d) for d in node.delegates],
                system_prompt=node.system_prompt,
            )
            result = await supervisor.ainvoke(
                {"messages": compose_input(state)},
                config={"recursion_limit": max(25, node.max_delegations * 2 + 6)},
            )
            output = _text(result["messages"][-1].content)
            steps.append(Step(node=node.key, name=node.name, type="supervisor", output=output))
            return {"last_output": output, "last_agent": node.name, "outputs": {node.key: output}}

        return run

    async def _compile(self, steps: list[Step]):
        tools_by_server = await self._load_tools()
        workflow = self.workflow
        graph = StateGraph(State)
        delegated = {d.agent for s in workflow.supervisors.values() for d in s.delegates}

        for key, node in workflow.agents.items():
            if key in delegated:
                continue
            graph.add_node(key, self._agent(node, self._agent_tools(node, tools_by_server), steps))
            graph.add_edge(key, node.next or END)
        for key, node in workflow.supervisors.items():
            graph.add_node(key, self._supervisor(node, tools_by_server, steps))
            graph.add_edge(key, node.next or END)
        for key, node in workflow.rags.items():
            if key in delegated:
                continue
            graph.add_node(key, self._rag(node, steps))
            graph.add_edge(key, node.next or END)
        for key, node in workflow.routers.items():
            graph.add_node(key, self._router(node, steps))
            graph.add_conditional_edges(key, lambda state: state["route"], {r.id: r.target for r in node.routes})

        graph.add_edge(START, workflow.start)
        return graph.compile()

    async def invoke(self, message: str, history: list[dict] | None = None) -> dict:
        """Ejecuta el flujo completo para un mensaje. Devuelve la respuesta final y los pasos."""
        steps: list[Step] = []
        compiled = await self._compile(steps)
        blocks = (
            len(self.workflow.agents) + len(self.workflow.routers) + len(self.workflow.supervisors) + len(self.workflow.rags)
        )
        state = await compiled.ainvoke(
            {
                "request": message,
                "history": history or [],
                "last_output": "",
                "last_agent": "",
                "route": "",
                "outputs": {},
            },
            config={"recursion_limit": blocks * 2 + 10},
        )
        return {"output": state["last_output"], "steps": [step.__dict__ for step in steps]}
