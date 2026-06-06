import json, urllib.request, urllib.error
BASE="http://localhost:7860"
KEY=json.load(open("apikey.json"))["api_key"]
FID=open("flow_id.txt").read().strip()
Q="evalúa la formulación del problema de investigación"
C="El presente estudio aborda el bajo rendimiento académico en estudiantes de secundaria. Se plantea como problema la falta de hábitos de estudio."
R="[Biblioteca] Hernández Sampieri: el planteamiento del problema debe incluir objetivos, preguntas y justificación."
P=""
tweaks={
 "PromptSupervisor":{"question":Q,"context":C},
 "PromptInvestigador":{"question":Q,"context":C,"reference_context":R},
 "PromptAuditor":{"question":Q,"context":C},
 "PromptMetodologo":{"question":Q,"context":C,"reference_context":R},
 "PromptRedactor":{"question":Q,"context":C[:800]},
 "PromptSintesis":{"question":Q,"previous_iteration":P},
}
payload={"input_value":Q,"output_type":"chat","input_type":"chat","tweaks":tweaks}
req=urllib.request.Request(f"{BASE}/api/v1/run/{FID}",
  data=json.dumps(payload).encode(),
  headers={"Content-Type":"application/json","x-api-key":KEY},method="POST")
try:
    with urllib.request.urlopen(req,timeout=180) as r:
        d=json.load(r)
    text=d["outputs"][0]["outputs"][0]["results"]["message"]["text"]
    print("RAW TEXT (first 600):",text[:600])
    parsed=json.loads(text)
    print("\nPARSED KEYS:",list(parsed.keys()))
    for k,v in parsed.items():
        print(" ",k,"->",type(v).__name__, (json.dumps(v,ensure_ascii=False)[:80] if not isinstance(v,str) else v[:80]))
except urllib.error.HTTPError as e:
    print("HTTP",e.code); print(e.read().decode()[:3000])
except Exception as e:
    print("ERR",type(e).__name__,str(e)[:500])
