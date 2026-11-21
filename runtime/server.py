"""API HTTP del workflow: `POST /invoke` con el mensaje, devuelve la respuesta y los pasos."""

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from runtime.config import load_workflow
from runtime.engine import WorkflowRunner


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class InvokeRequest(BaseModel):
    message: str = Field(..., min_length=1)
    # Turnos anteriores de la conversación (la API no guarda conversaciones).
    history: list[HistoryMessage] = Field(default_factory=list)


def create_app(folder: str | None = None) -> FastAPI:
    workflow = load_workflow(folder or os.environ.get("WORKFLOW_DIR", "."))
    runner = WorkflowRunner(workflow)
    app = FastAPI(title=workflow.name, version=str(workflow.version))

    @app.get("/health")
    def health():
        return {"status": "ok", "workflow": workflow.name, "version": workflow.version}

    @app.post("/invoke")
    async def invoke(request: InvokeRequest):
        try:
            return await runner.invoke(request.message, [m.model_dump() for m in request.history])
        except Exception as error:
            raise HTTPException(status_code=502, detail=f"El workflow falló: {error}")

    return app
