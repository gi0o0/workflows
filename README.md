# Workflows exportados de Agents Platform

Este repositorio lo mantiene **Agents Platform** (Workflows → Export). Cada workflow exportado
vive en su carpeta, `workflows/<nombre>/`, con:

- `workflow.json`: el grafo completo (agentes y sus prompts, routers, supervisores, RAG Agents,
  bloques Knowledge, servidores MCP y las tools elegidas). Lo regenera cada exportación: no lo
  edites a mano.
- `documents/`: el texto de los documentos que usan sus RAG Agents y bloques Knowledge. Se
  indexan la primera vez que se usan, en `.index/` (un caché que se puede borrar).
- `.env.example`: las variables que necesita (API keys de los proveedores y de los MCP).
- `README.md`: cómo ejecutarlo.

`runtime/` es el programa que ejecuta cualquiera de esos workflows con LangGraph, igual que la
plataforma. También se actualiza en cada exportación.

## Ejecutar un workflow

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
cp workflows/<nombre>/.env.example workflows/<nombre>/.env   # y completa las keys

# Una pregunta en la terminal
.venv/bin/python -m runtime run workflows/<nombre> "Hola" --steps

# API HTTP en http://localhost:8000
.venv/bin/python -m runtime serve workflows/<nombre>
```

`POST /invoke` recibe `{"message": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}`
y devuelve `{"output": "...", "steps": [...]}` (qué hizo cada bloque: la ruta que eligió un
router, las tareas que delegó un supervisor…). `GET /health` dice qué workflow y versión corre.

## Docker

```bash
docker build -t mi-workflow --build-arg WORKFLOW=workflows/<nombre> .
docker run -p 8000:8000 --env-file workflows/<nombre>/.env mi-workflow
```
