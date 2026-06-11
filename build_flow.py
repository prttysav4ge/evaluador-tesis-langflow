"""
Constructor del flujo de Langflow (6 agentes LLM en canvas) vía la API.

Genera los prompts con PARIDAD EXACTA llamando a los builders reales de
prompts/agent_prompts.py con centinelas, y reemplazándolos por variables
{{...}} de Langflow (use_double_brackets=true, así las llaves literales del
esquema JSON quedan intactas).

Topología:
  PromptX -> LlmX  (cadena LLM, OpenAIModel gpt-4o-mini)
  LlmX.text_output -> PromptY.<memvar>  (memoria acumulada)
  Llm*  -> PromptAssembler.<var> -> ChatOutput  (estado completo)

Los campos de datos (question, context, reference_context, previous_iteration)
NO se cablean: el cliente los inyecta por `tweaks` en tiempo de ejecución.
"""
import json
import copy
import urllib.request

BASE = "http://localhost:7860"
OE = "œ"  # carácter que Langflow usa para codificar handles

# ---- credenciales ----
with open("auto.json", encoding="utf-8") as f:
    TOKEN = json.load(f)["access_token"]

with open("lf_all.json", encoding="utf-8") as f:
    ALL = json.load(f)

OPENAI_MODEL = "gpt-4o-mini"
with open(".env", encoding="utf-8") as f:
    for line in f:
        if line.startswith("OPENAI_API_KEY="):
            OPENAI_KEY = line.split("=", 1)[1].strip()

SYS_MSG = "Eres un evaluador académico experto. Responde ÚNICAMENTE en JSON válido."

# ====================================================================== #
#  1. Generar los prompts con paridad exacta                              #
# ====================================================================== #
import sys
sys.path.insert(0, ".")
from prompts.agent_prompts import (
    build_mentor_intake_prompt, build_investigador_prompt, build_auditor_prompt,
    build_metodologico_prompt, build_redactor_prompt, build_mentor_final_prompt,
)

Q, C, R, P = "__VAR_QUESTION__", "__VAR_CONTEXT__", "__VAR_REFCTX__", "__VAR_PREVITER__"
MEM = {
    "mentor_intake": "__VAR_INTAKE__",
    "investigador":  "__VAR_RESEARCH__",
    "auditor":       "__VAR_AUDIT__",
    "metodologico":  "__VAR_METHOD__",
    "redactor":      "__VAR_WRITING__",
}

def to_langflow(text):
    """Reemplaza centinelas por variables {{...}} de Langflow."""
    # memoria: aparecen como cadenas json -> "__VAR_X__" -> {{x}} (sin comillas)
    text = text.replace('"__VAR_INTAKE__"',   "{{intake}}")
    text = text.replace('"__VAR_RESEARCH__"',  "{{research}}")
    text = text.replace('"__VAR_AUDIT__"',     "{{audit}}")
    text = text.replace('"__VAR_METHOD__"',    "{{method}}")
    text = text.replace('"__VAR_WRITING__"',   "{{writing}}")
    # datos: aparecen sin comillas (f-string directo)
    text = text.replace(Q, "{{question}}")
    text = text.replace(C, "{{context}}")
    text = text.replace(R, "{{reference_context}}")
    text = text.replace(P, "{{previous_iteration}}")
    return text

prompts = {
    "Supervisor":   (to_langflow(build_mentor_intake_prompt(Q, C)), ["question", "context"]),
    "Investigador": (to_langflow(build_investigador_prompt(Q, C, MEM, reference_context=R)),
                     ["question", "context", "reference_context", "intake"]),
    "Auditor":      (to_langflow(build_auditor_prompt(Q, C, MEM)),
                     ["question", "context", "intake", "research"]),
    "Metodologo":   (to_langflow(build_metodologico_prompt(Q, C, MEM, reference_context=R)),
                     ["question", "context", "reference_context", "intake", "research", "audit"]),
    "Redactor":     (to_langflow(build_redactor_prompt(Q, C, MEM)),
                     ["question", "context", "intake", "audit", "method"]),
    "Sintesis":     (to_langflow(build_mentor_final_prompt(Q, MEM, previous_iteration=P)),
                     ["question", "previous_iteration", "intake", "research", "audit", "method", "writing"]),
}
ASSEMBLER_TMPL = ('{"intake_result":{{intake}},"research_findings":{{research}},'
                  '"audit_result":{{audit}},"method_result":{{method}},'
                  '"writing_result":{{writing}},"mentor_result":{{mentor}}}')
ASSEMBLER_VARS = ["intake", "research", "audit", "method", "writing", "mentor"]

# ====================================================================== #
#  2. Constructores de nodos                                              #
# ====================================================================== #

def comp(cat, name):
    return copy.deepcopy(ALL[cat][name])

def prompt_var_field(varname):
    return {
        "advanced": False, "display_name": varname, "dynamic": False,
        "field_type": "str", "fileTypes": [], "file_path": "", "info": "",
        "input_types": ["Message", "Text"], "list": False, "load_from_db": False,
        "multiline": True, "name": varname, "placeholder": "", "required": False,
        "show": True, "title_case": False, "type": "str", "value": "",
    }

def make_node(node_id, ctype, node_def, pos):
    return {
        "id": node_id, "type": "genericNode", "position": pos,
        "data": {"id": node_id, "type": ctype, "node": node_def},
        "selected": False, "dragging": False,
    }

def make_prompt(node_id, template_text, varnames, pos):
    nd = comp("models_and_agents", "Prompt Template")
    nd["template"]["template"]["value"] = template_text
    nd["template"]["use_double_brackets"]["value"] = True
    nd["custom_fields"] = {"template": list(varnames)}
    for v in varnames:
        nd["template"][v] = prompt_var_field(v)
    nd["field_order"] = nd.get("field_order", []) + list(varnames)
    return make_node(node_id, "Prompt Template", nd, pos)

def make_openai(node_id, pos):
    nd = comp("openai", "OpenAIModel")
    nd["template"]["model_name"]["value"] = OPENAI_MODEL
    nd["template"]["api_key"]["value"] = OPENAI_KEY
    # load_from_db=False: usar el valor LITERAL, no resolverlo como nombre de
    # variable global de Langflow (que provocaría "OPENAI_API_KEY no seteada").
    nd["template"]["api_key"]["load_from_db"] = False
    nd["template"]["temperature"]["value"] = 0.3
    nd["template"]["max_tokens"]["value"] = 800
    nd["template"]["system_message"]["value"] = SYS_MSG
    return make_node(node_id, "OpenAIModel", nd, pos)

def make_io(node_id, name, pos):
    nd = comp("input_output", name)
    return make_node(node_id, name, nd, pos)

def make_cleaner(node_id, pos):
    nd = comp("processing", "JSONCleaner")
    return make_node(node_id, "JSONCleaner", nd, pos)

# ---- handles / edges ----

def enc(obj, spaces):
    sep = (", ", ": ") if spaces else (",", ":")
    return json.dumps(obj, ensure_ascii=False, separators=sep).replace('"', OE)

def edge(src, src_out, src_types, tgt, field, in_types, ftype):
    sh = {"dataType": NODE_CTYPE[src], "id": src, "name": src_out, "output_types": src_types}
    th = {"fieldName": field, "id": tgt, "inputTypes": in_types, "type": ftype}
    sh_c, th_c = enc(sh, False), enc(th, False)
    eid = f"xy-edge__{src}{sh_c}-{tgt}{th_c}"
    return {
        "animated": False, "className": "",
        "data": {"sourceHandle": sh, "targetHandle": th},
        "id": eid, "selected": False,
        "source": src, "sourceHandle": enc(sh, True),
        "target": tgt, "targetHandle": enc(th, True),
    }

# ====================================================================== #
#  3. Ensamblar nodos                                                     #
# ====================================================================== #
NODE_CTYPE = {}
nodes = []
agents = ["Supervisor", "Investigador", "Auditor", "Metodologo", "Redactor", "Sintesis"]

# Layout en 3 carriles para legibilidad: cada agente ocupa una columna
# (x = i*X_STEP) y sus 3 nodos se apilan en filas fijas — Prompt arriba (y=0),
# LLM en medio (y=220), Clean abajo (y=440). Así la cadena de cada agente se
# lee recta hacia abajo y el avance entre agentes de izquierda a derecha.
X_STEP = 380
for i, a in enumerate(agents):
    pid, gid, cid = f"Prompt{a}", f"Llm{a}", f"Clean{a}"
    tmpl, vrs = prompts[a]
    nodes.append(make_prompt(pid, tmpl, vrs, {"x": i * X_STEP, "y": 0}))
    nodes.append(make_openai(gid, {"x": i * X_STEP, "y": 220}))
    nodes.append(make_cleaner(cid, {"x": i * X_STEP, "y": 440}))
    NODE_CTYPE[pid] = "Prompt Template"
    NODE_CTYPE[gid] = "OpenAIModel"
    NODE_CTYPE[cid] = "JSONCleaner"

nodes.append(make_prompt("PromptAssembler", ASSEMBLER_TMPL, ASSEMBLER_VARS, {"x": 6 * X_STEP, "y": 0}))
NODE_CTYPE["PromptAssembler"] = "Prompt Template"
nodes.append(make_io("ChatOutputMain", "ChatOutput", {"x": 7 * X_STEP, "y": 0}))
NODE_CTYPE["ChatOutputMain"] = "ChatOutput"

# ====================================================================== #
#  4. Edges                                                               #
# ====================================================================== #
MSG = ["Message"]
edges = []

def link_prompt_to_llm(a):
    edges.append(edge(f"Prompt{a}", "prompt", MSG, f"Llm{a}", "input_value", MSG, "str"))

def link_llm_to_clean(a):
    edges.append(edge(f"Llm{a}", "text_output", MSG, f"Clean{a}", "json_str", MSG, "str"))

def link_mem(src_agent, tgt_prompt, var):
    # consume la salida YA LIMPIA (sin fences markdown) del agente fuente
    edges.append(edge(f"Clean{src_agent}", "output", MSG, tgt_prompt, var, ["Message", "Text"], "str"))

for a in agents:
    link_prompt_to_llm(a)
    link_llm_to_clean(a)

# memoria acumulada (réplica del orden de services/agent_service.py)
link_mem("Supervisor",  "PromptInvestigador", "intake")
link_mem("Supervisor",  "PromptAuditor",      "intake")
link_mem("Investigador","PromptAuditor",      "research")
link_mem("Supervisor",  "PromptMetodologo",   "intake")
link_mem("Investigador","PromptMetodologo",   "research")
link_mem("Auditor",     "PromptMetodologo",   "audit")
link_mem("Supervisor",  "PromptRedactor",     "intake")
link_mem("Auditor",     "PromptRedactor",     "audit")
link_mem("Metodologo",  "PromptRedactor",     "method")
link_mem("Supervisor",  "PromptSintesis",     "intake")
link_mem("Investigador","PromptSintesis",     "research")
link_mem("Auditor",     "PromptSintesis",     "audit")
link_mem("Metodologo",  "PromptSintesis",     "method")
link_mem("Redactor",    "PromptSintesis",     "writing")

# ensamblador: estado completo
link_mem("Supervisor",  "PromptAssembler", "intake")
link_mem("Investigador","PromptAssembler", "research")
link_mem("Auditor",     "PromptAssembler", "audit")
link_mem("Metodologo",  "PromptAssembler", "method")
link_mem("Redactor",    "PromptAssembler", "writing")
link_mem("Sintesis",    "PromptAssembler", "mentor")

# IMPORTANTE: el campo input_value de ChatOutput es de tipo "other" (no "str").
# Si se crea con "str", Langflow considera la arista incompatible y la BORRA al
# abrir el editor → la salida queda desconectada ("Input data cannot be None").
edges.append(edge("PromptAssembler", "prompt", MSG, "ChatOutputMain", "input_value",
                  ["Data", "JSON", "DataFrame", "Table", "Message"], "other"))

# ====================================================================== #
#  5. Crear el flujo vía API                                             #
# ====================================================================== #
flow = {
    "name": "Evaluador Tesis 6 Agentes",
    "description": "Panel multiagente (Supervisor→Investigador→Auditor→Metodólogo→Redactor→Síntesis) con paridad de prompts.",
    "data": {"nodes": nodes, "edges": edges, "viewport": {"x": 0, "y": 0, "zoom": 0.5}},
    "is_component": False,
    "endpoint_name": None,
}

req = urllib.request.Request(
    f"{BASE}/api/v1/flows/",
    data=json.dumps(flow).encode("utf-8"),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
    method="POST",
)
try:
    with urllib.request.urlopen(req) as resp:
        out = json.load(resp)
    print("FLOW_ID:", out["id"])
    with open("flow_id.txt", "w") as f:
        f.write(out["id"])
except urllib.error.HTTPError as e:
    print("HTTP", e.code)
    print(e.read().decode()[:2000])
