"""
Hornea el flujo del Evaluador para el HF Space, SIN secretos dentro.

Toma un export del flujo (el que produce GET /api/v1/flows/{id} o el botón
Export de Langflow) y genera `flows/evaluador-tesis.json`, listo para cargarse
en el arranque del contenedor vía LANGFLOW_LOAD_FLOWS_PATH.

Transformaciones:
  1. Los nodos GroqModel dejan de llevar la GROQ_API_KEY en texto plano: el campo
     api_key pasa a load_from_db=True con value="GROQ_API_KEY", de modo que
     Langflow la resuelve desde una VARIABLE GLOBAL homónima. Esa variable se
     crea desde el entorno del Space mediante
     LANGFLOW_VARIABLES_TO_GET_FROM_ENVIRONMENT=GROQ_API_KEY (ver Dockerfile).
  2. Se fija endpoint_name="evaluador-tesis" para invocar el flujo por un nombre
     estable (POST /api/v1/run/evaluador-tesis), independiente del UUID.

Uso:
    python hf_space/bake_flow.py <export.json>
Si no se pasa ruta, reutiliza flows/evaluador-tesis.json como fuente y destino.
"""
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DEST = HERE / "flows" / "evaluador-tesis.json"
ENDPOINT_NAME = "evaluador-tesis"
GROQ_VAR = "GROQ_API_KEY"  # nombre de la variable global en Langflow

src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEST
f = json.load(io.open(src, encoding="utf-8"))

# Normalizar a la forma de export que entiende LOAD_FLOWS_PATH.
flow = {
    "id": f["id"],
    "name": f.get("name", "Evaluador Tesis 6 Agentes"),
    "description": f.get("description", ""),
    "data": f["data"],
    "is_component": False,
    "endpoint_name": ENDPOINT_NAME,
    "tags": [],
}

stripped = 0
for n in flow["data"]["nodes"]:
    if n["data"].get("type") == "GroqModel":
        ak = n["data"]["node"]["template"]["api_key"]
        ak["load_from_db"] = True       # resolver desde variable global…
        ak["value"] = GROQ_VAR          # …llamada GROQ_API_KEY
        stripped += 1

DEST.parent.mkdir(parents=True, exist_ok=True)
with io.open(DEST, "w", encoding="utf-8") as out:
    json.dump(flow, out, ensure_ascii=False)

print(f"OK -> {DEST}")
print(f"  endpoint_name : {ENDPOINT_NAME}")
print(f"  nodos Groq sin clave literal (usan var global): {stripped}")
print(f"  nodes/edges   : {len(flow['data']['nodes'])}/{len(flow['data']['edges'])}")
