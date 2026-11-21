# AGENT-EXPT T workflow

Exportado desde Agents Platform · versión 2.

## Bloques

- **AGENT-EXPT T** · Agente · `openai/gpt-4.1-mini`

## Ejecutar

Desde la raíz del repositorio (ver el README principal para instalar):

```bash
cp workflows/agent-expt-t-workflow/.env.example workflows/agent-expt-t-workflow/.env   # y completa las variables
python -m runtime run workflows/agent-expt-t-workflow "Hola" --steps
python -m runtime serve workflows/agent-expt-t-workflow       # API en http://localhost:8000
```
