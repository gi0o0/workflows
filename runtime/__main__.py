"""Línea de comandos.

    python -m runtime serve workflows/<nombre>            # API en http://localhost:8000
    python -m runtime run workflows/<nombre> "Hola"       # una sola pregunta, en la terminal
"""

import argparse
import asyncio
import json
import os
import sys

from runtime.config import ConfigError, load_workflow
from runtime.engine import WorkflowRunner


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m runtime", description="Ejecuta un workflow exportado.")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Levanta la API HTTP del workflow.")
    serve.add_argument("folder", help="Carpeta del workflow (la que tiene workflow.json).")
    serve.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))

    run = commands.add_parser("run", help="Ejecuta el workflow con un mensaje y muestra la respuesta.")
    run.add_argument("folder", help="Carpeta del workflow (la que tiene workflow.json).")
    run.add_argument("message", help="Mensaje del usuario.")
    run.add_argument("--steps", action="store_true", help="Muestra también los pasos.")

    args = parser.parse_args()
    try:
        if args.command == "serve":
            import uvicorn

            from runtime.server import create_app

            uvicorn.run(create_app(args.folder), host=args.host, port=args.port)
        else:
            result = asyncio.run(WorkflowRunner(load_workflow(args.folder)).invoke(args.message))
            if args.steps:
                print(json.dumps(result["steps"], indent=2, ensure_ascii=False))
            print(result["output"])
    except ConfigError as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
