import asyncio, json
from langflow.client import langflow_client

async def main():
    q="evalúa la formulación del problema de investigación"
    c="El presente estudio aborda el bajo rendimiento académico en estudiantes de secundaria. Se plantea como problema la falta de hábitos de estudio."
    r="[Biblioteca] Hernández Sampieri: el planteamiento del problema debe incluir objetivos, preguntas y justificación."
    res = await langflow_client.call_chatflow(question=q, context=c, reference_context=r, previous_iteration=None)
    print("TYPE:", type(res).__name__)
    print("KEYS:", list(res.keys()))
    for k in res:
        v=res[k]
        print(" ", k, "->", type(v).__name__, "fields:", list(v.keys())[:4] if isinstance(v,dict) else v[:40])

asyncio.run(main())
