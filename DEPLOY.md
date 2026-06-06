# Despliegue — Evaluador de Tesis (Streamlit Cloud + Langflow en HF Space)

Arquitectura de producción:

```
Navegador ─► Streamlit Community Cloud (streamlit_app.py + FastAPI embebido)
                       │  POST /api/v1/run/evaluador-tesis  (HTTPS + x-api-key)
                       ▼
             Hugging Face Space (Docker)  ──►  Langflow 1.9.5 + flujo de 6 agentes
                                                 (GROQ_API_KEY como variable global)
```

- Los **6 agentes** corren en **Langflow**, alojado en un **HF Space (Docker SDK)**.
- El **flujo va horneado** en la imagen (`hf_space/flows/evaluador-tesis.json`) y se
  carga al arrancar con `LANGFLOW_LOAD_FLOWS_PATH` → no depende del disco efímero.
- **Nada hardcodeado**: la clave de Groq y las credenciales viven en *Secrets*.
- Auth: `AUTO_LOGIN=false` + superusuario. El cliente Streamlit hace login y
  **crea/renueva la API key en runtime** (sobrevive a los reinicios del Space).

> Validado en local: la imagen del Space corriendo como uid 1000 (igual que HF)
> auto-carga el flujo, exige auth en `/run`, resuelve la clave Groq desde el
> entorno y devuelve las 6 claves del panel. El cliente adaptado funciona contra
> ese contenedor y también en modo local sin auth.

---

## PARTE 1 — Hugging Face Space (motor Langflow)

1. Crea cuenta en <https://huggingface.co> → **New Space**.
   - **SDK: Docker** · nombre p.ej. `evaluador-tesis-langflow` · visibilidad a tu gusto
     (la imagen NO contiene secretos, pero privado es más conservador).
2. Sube el contenido de la carpeta **`hf_space/`** al repo del Space (web *Upload files*
   o `git push`). Deben quedar en la raíz del Space:
   - `Dockerfile`
   - `README.md`  (su frontmatter ya declara `sdk: docker` y `app_port: 7860`)
   - `flows/evaluador-tesis.json`
3. **Settings → Variables and secrets** → añade como **Secrets**:

   | Secret | Valor |
   |--------|-------|
   | `GROQ_API_KEY` | tu clave de Groq (`gsk_…`) |
   | `LANGFLOW_SUPERUSER` | p.ej. `admin` |
   | `LANGFLOW_SUPERUSER_PASSWORD` | una contraseña fuerte |
   | `LANGFLOW_SECRET_KEY` | una cadena aleatoria larga (cifra variables) |

4. Espera a que el Space pase a **Running** (primer build ~ varios min).
5. Verifica (sustituye la URL real del Space, *Settings → Embed/Direct URL*):
   ```
   curl https://TU-USUARIO-evaluador-tesis-langflow.hf.space/health      → 200
   ```
   NO pruebes `/run` sin auth (responderá 403, es lo correcto).

> Nota: en el plan gratuito el Space **se duerme** tras inactividad. Al despertar,
> rehornea el flujo desde la imagen y el cliente vuelve a derivar la API key sola;
> la primera petición tras dormir puede tardar (~30 s) o caer al fallback una vez.

---

## PARTE 2 — Subir el código a GitHub (para Streamlit Cloud)

El repo ya está `git init`-ado y el `.gitignore` excluye **todos** los secretos
(`.env`, `.streamlit/secrets.toml`, `auto.json`, `apikey.json`, …) — verificado.

```powershell
cd c:\LANGFLOW
git add -A
git commit -m "Despliegue: Langflow en HF Space + cliente con auth de producción"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/TU-REPO.git
git push -u origin main
```

Tras el push, confirma en GitHub que **no** aparecen `.env` ni `secrets.toml`.

---

## PARTE 3 — Streamlit Community Cloud (frontend)

1. <https://share.streamlit.io> → **New app** → conecta tu repo de GitHub.
   - Branch: `main` · Main file path: `streamlit_app.py`
2. **Advanced settings → Secrets**: pega el contenido de
   [`.streamlit/secrets.cloud.example.toml`](.streamlit/secrets.cloud.example.toml)
   con los valores reales. Lo esencial:
   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "gsk_…"
   USE_LANGFLOW = "true"
   LANGFLOW_URL = "https://TU-USUARIO-evaluador-tesis-langflow.hf.space"
   LANGFLOW_FLOW_ID = "evaluador-tesis"
   LANGFLOW_API_KEY = ""                       # vacío → se usa el superusuario
   LANGFLOW_SUPERUSER = "admin"                # mismas credenciales que el Space
   LANGFLOW_SUPERUSER_PASSWORD = "…"
   ```
3. **Deploy**. El primer arranque es lento: descarga el modelo de embeddings
   (~470 MB) e indexa la Biblioteca Metodológica (en segundo plano, varios min).

---

## Verificación final (¿de verdad usa Langflow?)

En la app, tras una evaluación, el badge debe decir **`Modo: langflow`** y la barra
lateral **"Langflow: 🟢 activo"**. Si dijera `python_agents_fallback`, Langflow no
respondió (revisa que el Space esté Running y los Secrets de Langflow en Streamlit).

---

## Riesgos conocidos

- **Memoria de Streamlit Cloud**: el modelo de embeddings + `torch` + ChromaDB +
  la biblioteca en memoria van justos en el tier gratuito. Si hay OOM, considera
  reducir la biblioteca o un embedding más liviano.
- **El fallback enmascara fallos**: si el Space está dormido/caído, la app igual
  responde por Python. Por eso hay que mirar el badge `langflow`.
- **`/run` exige API key**, no JWT, y la DB del Space es efímera → por eso el
  cliente usa credenciales de superusuario (no una key fija) y la renueva sola.

## Re-hornear el flujo (si cambias los agentes)

```powershell
# con Langflow local corriendo y el flujo reconstruido (build_flow.py):
python hf_space\bake_flow.py        # regenera hf_space/flows/evaluador-tesis.json sin secretos
# luego sube el JSON actualizado al repo del Space
```
