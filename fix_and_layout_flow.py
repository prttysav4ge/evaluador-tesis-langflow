"""
Parchea EN SITIO el flujo de Langflow (mismo flow_id) sin tocar la lógica:

  1. REPARA la conexión final PromptAssembler -> ChatOutput. build_flow.py la creó
     con targetHandle type="str", pero el campo input_value de ChatOutput es de
     tipo "other" (input_types Data/JSON/DataFrame/Table/Message). Por ese
     desajuste Langflow la marcaba "incompatible" y la borraba al abrir el editor,
     dejando la salida sin conectar ("Input data cannot be None"). La recreamos
     con el tipo correcto para que sobreviva a futuras aperturas del editor.

  2. REORDENA el canvas en 3 carriles (sin cambiar nodos ni aristas):
        Fila 1 (y=0):   Prompt x6  ......  Assembler  ChatOutput
        Fila 2 (y=220): Groq x6
        Fila 3 (y=440): Clean x6

Se hace vía PATCH /api/v1/flows/{id}, preservando el flow_id.
"""
import io
import json
import urllib.request
import urllib.error

BASE = "http://localhost:7860"
FLOW_ID = open("flow_id.txt").read().strip()
OE = "œ"  # 'œ' — Langflow codifica las comillas de los handles con este carácter

TOKEN = json.load(open("auto.json", encoding="utf-8"))["access_token"]

AGENT_COL = {"Supervisor": 0, "Investigador": 1, "Auditor": 2,
             "Metodologo": 3, "Redactor": 4, "Sintesis": 5}
X_STEP = 380
ASM_X = 6 * X_STEP          # Assembler tras la última columna de agentes
OUT_X = ASM_X + X_STEP      # ChatOutput al final
ROW_Y = {"prompt": 0, "groq": 220, "clean": 440}


def new_position(node_id: str):
    if node_id == "PromptAssembler":
        return {"x": ASM_X, "y": ROW_Y["prompt"]}
    if node_id == "ChatOutputMain":
        return {"x": OUT_X, "y": ROW_Y["prompt"]}
    if node_id.startswith("Prompt"):
        return {"x": AGENT_COL[node_id[len("Prompt"):]] * X_STEP, "y": ROW_Y["prompt"]}
    if node_id.startswith("Groq"):
        return {"x": AGENT_COL[node_id[len("Groq"):]] * X_STEP, "y": ROW_Y["groq"]}
    if node_id.startswith("Clean"):
        return {"x": AGENT_COL[node_id[len("Clean"):]] * X_STEP, "y": ROW_Y["clean"]}
    return None  # no debería ocurrir


def enc(obj, spaces):
    sep = (", ", ": ") if spaces else (",", ":")
    return json.dumps(obj, ensure_ascii=False, separators=sep).replace('"', OE)


def build_output_edge():
    # Mismo patrón que build_flow.py.edge(), pero con type="other" (el real del
    # campo input_value de ChatOutput) en vez de "str".
    sh = {"dataType": "Prompt Template", "id": "PromptAssembler",
          "name": "prompt", "output_types": ["Message"]}
    th = {"fieldName": "input_value", "id": "ChatOutputMain",
          "inputTypes": ["Data", "JSON", "DataFrame", "Table", "Message"],
          "type": "other"}
    eid = f"xy-edge__PromptAssembler{enc(sh, False)}-ChatOutputMain{enc(th, False)}"
    return {
        "animated": False, "className": "",
        "data": {"sourceHandle": sh, "targetHandle": th},
        "id": eid, "selected": False,
        "source": "PromptAssembler", "sourceHandle": enc(sh, True),
        "target": "ChatOutputMain", "targetHandle": enc(th, True),
    }


# ── Cargar el flujo actual ───────────────────────────────────────────────
flow = json.load(io.open("__pwtest/flow_now.json", encoding="utf-8"))
data = flow["data"]
nodes, edges = data["nodes"], data["edges"]

# 1. Reposicionar
for n in nodes:
    pos = new_position(n["id"])
    if pos:
        n["position"] = pos
        n["positionAbsolute"] = pos

# 2. Reparar la arista de salida si falta
has_output_edge = any(
    e["source"] == "PromptAssembler" and e["target"] == "ChatOutputMain"
    for e in edges
)
if not has_output_edge:
    edges.append(build_output_edge())
    print("Arista de salida RE-AÑADIDA (PromptAssembler -> ChatOutputMain, type=other)")
else:
    print("La arista de salida ya existía; no se duplica")

data["viewport"] = {"x": 50, "y": 80, "zoom": 0.45}
print(f"NODES: {len(nodes)}  EDGES: {len(edges)}")

# ── PATCH en sitio (mismo flow_id) ───────────────────────────────────────
payload = {"name": flow["name"], "description": flow.get("description", ""), "data": data}
req = urllib.request.Request(
    f"{BASE}/api/v1/flows/{FLOW_ID}",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
    method="PATCH",
)
try:
    with urllib.request.urlopen(req) as resp:
        out = json.load(resp)
    print("PATCH OK — flow_id:", out["id"], "| edges:", len(out["data"]["edges"]))
except urllib.error.HTTPError as e:
    print("HTTP", e.code)
    print(e.read().decode()[:2000])
