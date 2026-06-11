---
title: Evaluador Tesis Langflow
emoji: 🎓
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Evaluador de Tesis — motor Langflow

Backend de flujos (Langflow) del Evaluador de Proyectos de Tesis. Aloja el panel
multiagente de 6 agentes (Supervisor → Investigador → Auditor → Metodólogo →
Redactor → Síntesis) y expone su API para que el frontend Streamlit lo consuma.

El flujo viene **horneado** en la imagen (`flows/evaluador-tesis.json`) y se carga
al arrancar con `LANGFLOW_LOAD_FLOWS_PATH`, así no depende del almacenamiento
efímero del Space.

## Secrets requeridos (Settings → Secrets del Space)

| Secret | Para qué |
|--------|----------|
| `OPENAI_API_KEY` | Clave de OpenAI; se inyecta como variable global del flujo |
| `LANGFLOW_SUPERUSER` | Usuario admin (auth API; `AUTO_LOGIN=false`) |
| `LANGFLOW_SUPERUSER_PASSWORD` | Contraseña admin |
| `LANGFLOW_SECRET_KEY` | (Recomendado) clave fija para cifrar variables |

## Cómo lo llama el frontend

`POST https://<este-space>.hf.space/api/v1/run/evaluador-tesis`
(autenticado con un token obtenido por login del superusuario).
