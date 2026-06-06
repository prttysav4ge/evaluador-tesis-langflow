"""
Ruta: POST /api/v1/query

Orquesta el pipeline RAG + agentes:
  1. Recupera chunks relevantes desde ChromaDB (RAG)
  2. Formatea el contexto
  3a. Si USE_LANGFLOW=true  → llama al flujo de Langflow
  3b. Si USE_LANGFLOW=false → ejecuta los 6 agentes en Python directamente
  4. Retorna la respuesta final al frontend
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from vectorstore.chroma_store import chroma_store
from vectorstore.refs_store import refs_store

logger = logging.getLogger(__name__)
router = APIRouter()

# ====================================================================== #
#  Topes de tiempo (segundos)                                             #
# ====================================================================== #
# Tope por una sola corrida del pipeline Python de agentes (fallback). El
# cliente Langflow ya trae su propio timeout de 120 s; aquí acotamos la pasada
# Python equivalente para que un rate-limit de Groq no cuelgue la UI.
_PYTHON_PIPELINE_TIMEOUT = 240
# Tope GLOBAL de la evaluación completa (suma de TODAS las iteraciones). Con un
# máximo de 3 iteraciones encadenadas y llamadas LLM lentas, 600 s da margen
# suficiente sin dejar la petición colgada indefinidamente en Streamlit Cloud.
_EVAL_GLOBAL_TIMEOUT = 600


# ====================================================================== #
#  Modelos de entrada / salida                                            #
# ====================================================================== #

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Pregunta o instrucción de evaluación sobre la tesis.",
        example="evalúa la formulación del problema de investigación",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de fragmentos relevantes a recuperar de ChromaDB.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="ID de sesión para mantener conversación en Flowise (opcional).",
    )
    iterations: int = Field(
        default=1,
        ge=1,
        le=3,
        description=(
            "Cantidad de iteraciones del panel multiagente. En cada iteración, "
            "el agente de Síntesis recibe la síntesis previa y la refina."
        ),
    )
    seccion: Optional[str] = Field(
        default=None,
        description=(
            "Nombre completo de la sección del TOC a evaluar (ej. '1.2 Objetivos'). "
            "Si se envía, el retrieval se acota a esa sección y sus subsecciones "
            "(metadata `seccion`), igual que langgraph. None = búsqueda semántica global."
        ),
    )
    page_start: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "[Compat] Página inicial de la sección. Solo se usa si `seccion` no se "
            "envía; acota el retrieval a ese rango de páginas."
        ),
    )
    page_end: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "[Compat] Página final (inclusive). Solo se usa si `seccion` no se envía."
        ),
    )


class QueryResponse(BaseModel):
    question: str
    mode: str  # "langflow" | "python_agents"
    chunks_retrieved: int
    elapsed_seconds: float
    context_preview: str
    reference_chunks_retrieved: int = 0
    reference_context_preview: str  = ""
    iterations_count: int           = 1
    result: Dict[str, Any]


# ====================================================================== #
#  Endpoint principal                                                     #
# ====================================================================== #

@router.post(
    "/query",
    summary="Consultar la tesis y ejecutar evaluación multiagente",
    response_model=QueryResponse,
)
async def query_thesis(body: QueryRequest) -> QueryResponse:
    """
    Pipeline RAG + agentes secuenciales.

    **Flujo:**
    - Recupera los chunks más relevantes de ChromaDB para la pregunta.
    - Si `USE_LANGFLOW=true`: envía contexto + pregunta a Langflow.
    - Si `USE_LANGFLOW=false`: ejecuta los 6 agentes secuenciales en Python.

    **Ejemplo de pregunta:**
    - `"evalúa la formulación del problema de investigación"`
    - `"¿es adecuado el marco metodológico?"`
    - `"¿qué debilidades tiene el marco teórico?"`
    """
    start = time.time()

    # ------------------------------------------------------------------ #
    #  1. Verificar que hay datos en ChromaDB                             #
    # ------------------------------------------------------------------ #
    info = chroma_store.get_info()
    if info["total_chunks"] == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                "No hay ninguna tesis cargada. "
                "Sube primero un PDF en POST /api/v1/upload-pdf."
            ),
        )

    # ------------------------------------------------------------------ #
    #  2. Retrieval desde ChromaDB                                        #
    # ------------------------------------------------------------------ #
    # Prioridad de acotamiento (anti-cross-topic):
    #   1) `seccion`  → recuperación por sección + subsecciones (metadata, como
    #                   langgraph). Es el camino principal del nuevo frontend.
    #   2) page_start/page_end → filtro por rango de páginas (compat).
    #   3) ninguno    → búsqueda semántica global (Vista general).
    try:
        if body.seccion:
            logger.info(
                f"🔍 Retrieval acotado a sección '{body.seccion}' "
                f"para: '{body.question[:60]}…'"
            )
            raw_results = chroma_store.query_by_section(
                body.seccion, fallback_question=body.question
            )
        else:
            page_where = _build_page_where(body.page_start, body.page_end)
            if page_where is not None:
                logger.info(
                    f"🔍 Retrieval acotado a páginas [{body.page_start}–{body.page_end or 'fin'}] "
                    f"para: '{body.question[:60]}…'"
                )
            else:
                logger.info(f"🔍 Buscando chunks relevantes para: '{body.question[:80]}…'")

            raw_results = chroma_store.query(body.question, top_k=body.top_k, where=page_where)
            # Degradación graciosa: si el filtro por rango de páginas no devuelve
            # nada, reintentamos con búsqueda global para no dejar al usuario sin
            # evaluación.
            if not raw_results and page_where is not None:
                logger.warning(
                    "⚠️  El filtro por rango de páginas no devolvió fragmentos; "
                    "reintentando con búsqueda semántica global."
                )
                raw_results = chroma_store.query(body.question, top_k=body.top_k)
    except Exception as exc:
        logger.exception("Error en retrieval ChromaDB")
        raise HTTPException(status_code=500, detail=f"Error en ChromaDB: {str(exc)}")

    if not raw_results:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron fragmentos relevantes. Revisa la pregunta o sube más contenido.",
        )

    retrieved_context = chroma_store.format_context(raw_results)
    context_preview = retrieved_context[:300] + "…" if len(retrieved_context) > 300 else retrieved_context

    logger.info(f"📚 Fragmentos recuperados (tesis): {len(raw_results)}")

    # ------------------------------------------------------------------ #
    #  2b. Retrieval cruzado contra Biblioteca Metodológica               #
    # ------------------------------------------------------------------ #
    # Recuperamos fragmentos de los libros metodológicos usando la misma
    # pregunta. Si la biblioteca está vacía o falla, el pipeline sigue
    # funcionando con solo el contexto de la tesis (degradación graciosa).
    refs_raw: list[Dict[str, Any]] = []
    reference_context: str = ""
    try:
        if refs_store.collection.count() > 0:
            refs_raw = refs_store.query(body.question, top_k=_REFS_TOP_K)
            reference_context = _format_refs_context(refs_raw)
            logger.info(f"📖 Fragmentos recuperados (biblioteca): {len(refs_raw)}")
    except Exception as exc:
        logger.warning(f"⚠️  Retrieval de biblioteca falló (continuando sin refs): {exc}")
        refs_raw = []
        reference_context = ""

    reference_context_preview = (
        reference_context[:300] + "…" if len(reference_context) > 300 else reference_context
    )

    # ------------------------------------------------------------------ #
    #  3. Agentes — loop de iteraciones                                   #
    # ------------------------------------------------------------------ #
    # En cada iteración corremos el panel completo (6 agentes). A partir
    # de la iteración 2, la síntesis previa se pasa como contexto extra
    # al agente Síntesis para que refine en vez de empezar de cero.
    async def _run_all_iterations() -> tuple[list, Dict[str, Any], str]:
        """Corre todas las iteraciones. Se envuelve en un wait_for para acotar
        el TOTAL de la evaluación (no cada iteración por separado)."""
        history: list[Dict[str, Any]] = []
        prev_text: Optional[str]      = None
        last_result: Dict[str, Any]   = {}
        run_mode: str                 = "python_agents"

        for iter_num in range(1, body.iterations + 1):
            logger.info(f"🔁 Iteración {iter_num}/{body.iterations}")

            if settings.USE_LANGFLOW:
                iter_result, iter_mode = await _call_langflow_with_fallback(
                    body.question, retrieved_context, reference_context,
                    body.session_id, previous_iteration=prev_text,
                )
            else:
                iter_mode = "python_agents"
                iter_result = await _call_python_agents(
                    body.question, retrieved_context, reference_context,
                    previous_iteration=prev_text,
                )

            # Extraemos el output de la síntesis de esta iteración para alimentar
            # la siguiente. JSON string compacto para minimizar tokens.
            iter_synthesis = _extract_synthesis_json(iter_result)
            prev_text = iter_synthesis if iter_synthesis else None

            history.append({
                "iteration": iter_num,
                "mode": iter_mode,
                "result": iter_result,
            })
            last_result = iter_result
            run_mode    = iter_mode

        return history, last_result, run_mode

    # Tope GLOBAL de la evaluación completa: a diferencia del tope por-iteración
    # de _call_python_agents, este acota la suma de TODAS las iteraciones para
    # que en Streamlit Cloud (TestClient sin timeout de socket) la UI no se
    # cuelgue aunque haya varias iteraciones encadenadas.
    try:
        iterations_history, final_result, mode = await asyncio.wait_for(
            _run_all_iterations(), timeout=_EVAL_GLOBAL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"⏰ La evaluación completa superó el tope global de "
            f"{_EVAL_GLOBAL_TIMEOUT}s. Abortando para no colgar la UI."
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"La evaluación superó el tope global de {_EVAL_GLOBAL_TIMEOUT}s. "
                "Baja las iteraciones a 1 y el Top-K, verifica que el nodo End de "
                "Flowise devuelva 'Last Output', o sube a Groq Dev Tier."
            ),
        )

    # El frontend espera el último iter como top-level (backward compat con
    # las 4 pestañas) y la historia completa en 'iterations_history' para
    # poder renderizar P2 con sesiones múltiples.
    result = final_result
    result["iterations_history"] = [
        {
            "iteration": h["iteration"],
            "mode": h["mode"],
            "memory": h["result"].get("memory"),
            "langflow_response": h["result"].get("langflow_response"),
            "_langflow_fallback": h["result"].get("_langflow_fallback"),
        }
        for h in iterations_history
    ]

    # Adjuntamos contexto cruzado al resultado para que el frontend lo
    # use en la sub-pestaña 'De libros de referencia' / 'Contexto cruzado'.
    result["reference_context"] = reference_context
    result["reference_chunks"]  = [
        {
            "text":   r.get("text", ""),
            "source": r.get("metadata", {}).get("source", "?"),
            "page":   r.get("metadata", {}).get("page", "?"),
            "score":  r.get("score"),
        }
        for r in refs_raw
    ]

    # ------------------------------------------------------------------ #
    #  4. Redactor con rúbrica (post-pipeline, ambos modos)              #
    #     Selecciona secciones, califica la ENTRADA y decide umbral:     #
    #       ≥ umbral → solo recomendar pulido (NO reescribe).            #
    #       < umbral → produce el TEXTO DE SALIDA mejorado.              #
    #     `rubrica_entrada` y `secciones` quedan en el result para que   #
    #     las métricas on-demand (Gain/G-Eval) las reusen sin recalcular.#
    #     Se acota en tiempo para no superar el timeout del cliente.     #
    # ------------------------------------------------------------------ #
    _REDACTOR_TIMEOUT = 90  # segundos máximos (incluye 2 llamadas de juez + reescritura)
    try:
        from services.agent_service import run_redactor_rubrica
        evaluation_data       = _extract_evaluation_data(result)
        investigador_findings = _extract_investigador_findings(result)
        redactor = await asyncio.wait_for(
            run_redactor_rubrica(
                question=body.question,
                original_context=retrieved_context,
                final_evaluation=evaluation_data,
                investigador_findings=investigador_findings,
            ),
            timeout=_REDACTOR_TIMEOUT,
        )
        result["redactor_rubrica"] = redactor
        # texto_sugerido = el TEXTO DE SALIDA (None si la entrada ya era buena).
        result["texto_sugerido"]   = redactor.get("texto_salida")
        result["original_context"] = retrieved_context   # para comparación en UI
    except asyncio.TimeoutError:
        logger.warning(
            f"⚠️  run_redactor_rubrica excedió {_REDACTOR_TIMEOUT}s — se omite."
        )
        result["redactor_rubrica"] = None
        result["texto_sugerido"]   = None
        result["original_context"] = retrieved_context
    except Exception as exc:
        logger.warning(f"⚠️  No se pudo ejecutar el Redactor con rúbrica: {exc}")
        result["redactor_rubrica"] = None
        result["texto_sugerido"]   = None
        result["original_context"] = retrieved_context

    elapsed = round(time.time() - start, 2)
    logger.info(f"✅ Query completada en {elapsed}s (modo: {mode})")

    return QueryResponse(
        question=body.question,
        mode=mode,
        chunks_retrieved=len(raw_results),
        elapsed_seconds=elapsed,
        context_preview=context_preview,
        reference_chunks_retrieved=len(refs_raw),
        reference_context_preview=reference_context_preview,
        iterations_count=body.iterations,
        result=result,
    )


# ====================================================================== #
#  Helpers privados                                                       #
# ====================================================================== #

# Cantidad de fragmentos de la Biblioteca Metodológica a recuperar por consulta.
# Más bajo que el TOP_K de la tesis para no inflar el prompt de los agentes.
_REFS_TOP_K = 3


def _build_page_where(
    page_start: Optional[int], page_end: Optional[int]
) -> Optional[Dict[str, Any]]:
    """
    Construye el filtro `where` de ChromaDB para acotar el retrieval al rango
    de páginas de la sección seleccionada en el frontend.

    - Ambos definidos → rango cerrado [start, end]. ChromaDB exige `$and` para
      combinar dos comparadores sobre el mismo campo (`page`).
    - Solo start (última sección del documento) → desde start hasta el final.
    - Ninguno (Vista general) → None: sin filtro, búsqueda semántica global.
    """
    if page_start is None:
        return None
    conds: list[Dict[str, Any]] = [{"page": {"$gte": page_start}}]
    if page_end is not None:
        conds.append({"page": {"$lte": page_end}})
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _format_refs_context(refs_results: list) -> str:
    """
    Formatea los chunks recuperados de la Biblioteca Metodológica con
    atribución (libro + página) en lugar de 'sección detectada'.
    """
    if not refs_results:
        return ""
    parts: list = []
    for i, r in enumerate(refs_results, 1):
        meta = r.get("metadata", {}) or {}
        source = meta.get("source", "?")
        page   = meta.get("page", "?")
        parts.append(
            f"[Biblioteca | Fragmento {i} | Libro: {source} | p.{page}]\n"
            f"{r.get('text', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_synthesis_json(result: Dict[str, Any]) -> str:
    """
    Extrae el JSON de la síntesis final (Mentor Final / Síntesis y Consenso)
    como string compacto para pasarlo a la siguiente iteración.

    Funciona en ambos modos:
      - Langflow: result['langflow_response']['text'] suele ser el JSON.
      - Python:  result['memory']['mentor_final'] es el dict.
    """
    # Modo Python
    if "memory" in result:
        synth = result["memory"].get("mentor_final")
        if synth:
            try:
                return json.dumps(synth, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                return ""

    # Modo Langflow
    if "langflow_response" in result:
        flow_resp = result["langflow_response"]
        if isinstance(flow_resp, dict):
            text = flow_resp.get("text") or flow_resp.get("output") or ""
            if isinstance(text, str) and text.strip():
                return text.strip()

    return ""




async def _call_langflow_with_fallback(
    question: str,
    context: str,
    reference_context: str,
    session_id: Optional[str],
    previous_iteration: Optional[str] = None,
) -> tuple[Dict[str, Any], str]:
    """
    Intenta llamar a Langflow. Si el cliente lanza RuntimeError (timeout, HTTP
    error, JSON inválido o estructura inesperada) o cae la conexión, hace
    fallback automático a los agentes Python. Retorna (result_dict, mode_str).

    El JSON final ya parseado que devuelve Langflow se envuelve en el MISMO
    sobre que usa Langflow: {"langflow_response": {"text": <json string>}}. Así el
    frontend Streamlit (_extract_agent_outputs) y los extractores de este módulo
    (_extract_synthesis_json / _extract_evaluation_data / _extract_investigador_findings)
    funcionan sin ningún cambio. Si el flujo de Langflow devuelve el estado
    completo (intake_result, research_findings, …) el frontend poblará los 6
    agentes; si solo devuelve la síntesis final, poblará el Mentor Final.
    """
    from langflow.client import langflow_client

    try:
        parsed = await langflow_client.call_chatflow(
            question=question,
            context=context,
            reference_context=reference_context,
            session_id=session_id,
            previous_iteration=previous_iteration,
        )
        langflow_envelope = {
            "text": json.dumps(parsed, ensure_ascii=False)
            if isinstance(parsed, dict) else str(parsed)
        }
        return {"langflow_response": langflow_envelope}, "langflow"

    except (RuntimeError, httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
        exc_type = type(exc).__name__
        logger.warning(
            f"⚠️  Langflow falló ({exc_type}: {exc}) — usando fallback Python."
        )
        result = await _call_python_agents(
            question, context, reference_context, previous_iteration=previous_iteration
        )
        result["_langflow_fallback"] = (
            f"Langflow no respondió ({exc_type}). "
            "Se usaron los agentes Python como fallback."
        )
        return result, "python_agents_fallback"


async def _call_python_agents(
    question: str,
    context: str,
    reference_context: str = "",
    previous_iteration: Optional[str] = None,
) -> Dict[str, Any]:
    from services.agent_service import run_sequential_pipeline

    try:
        return await asyncio.wait_for(
            run_sequential_pipeline(
                question, context, reference_context, previous_iteration=previous_iteration
            ),
            timeout=_PYTHON_PIPELINE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"⏰ Pipeline Python superó {_PYTHON_PIPELINE_TIMEOUT}s "
            "(probable rate-limit de Groq en tier gratuito). Abortando para no "
            "colgar la UI."
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"La evaluación con agentes Python superó {_PYTHON_PIPELINE_TIMEOUT}s, "
                "probablemente por rate-limiting de Groq (tier gratuito). "
                "Reintenta con menos iteraciones / Top-K más bajo, o sube a Groq Dev Tier."
            ),
        )
    except Exception as exc:
        logger.exception("Error en pipeline de agentes Python")
        raise HTTPException(
            status_code=500,
            detail=f"Error en agentes: {str(exc)}",
        )


def _extract_evaluation_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae el dict de evaluación final del resultado del pipeline.

    - Modo Flowise:  parsea result["langflow_response"]["text"] como JSON.
    - Modo Python:   retorna result["memory"]["mentor_final"].
    """
    # Modo Langflow / Langflow
    if "langflow_response" in result:
        flowise_resp = result["langflow_response"]
        if isinstance(flowise_resp, dict):
            text = flowise_resp.get("text", "")
            if text:
                try:
                    parsed = json.loads(text.strip())
                except Exception:
                    return {}
                if isinstance(parsed, dict):
                    # Formato Langflow (estado completo): la evaluación final vive
                    # en 'mentor_result' (puede ser dict o JSON string).
                    if "mentor_result" in parsed:
                        mentor = parsed["mentor_result"]
                        if isinstance(mentor, str):
                            try:
                                mentor = json.loads(mentor)
                            except Exception:
                                pass
                        return mentor if isinstance(mentor, dict) else {}
                    # Formato Flowise legacy: el text ES la síntesis final.
                    return parsed
        return {}

    # Modo Python
    if "memory" in result:
        return result["memory"].get("mentor_final", {})

    return {}


def _extract_investigador_findings(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae los hallazgos del agente Investigador del resultado.

    - Modo Python:   result["memory"]["investigador"]  (directo).
    - Modo Flowise:  busca el nodo 'Investigador' en agentFlowExecutedData
                     y parsea su output.content como JSON.
    """
    # Modo Python
    if "memory" in result:
        return result["memory"].get("investigador", {})

    # Modo Langflow (estado completo): research_findings en el text serializado.
    if "langflow_response" in result:
        flow_data = result["langflow_response"]
        if isinstance(flow_data, dict):
            text = flow_data.get("text", "")
            if text:
                try:
                    parsed = json.loads(text.strip())
                except Exception:
                    parsed = None
                if isinstance(parsed, dict) and "research_findings" in parsed:
                    rf = parsed["research_findings"]
                    if isinstance(rf, str):
                        try:
                            rf = json.loads(rf)
                        except Exception:
                            pass
                    if isinstance(rf, dict):
                        return rf

    # Modo Langflow legacy: buscar en el árbol de ejecución
    if "langflow_response" in result:
        flow_data = result["langflow_response"]
        if isinstance(flow_data, dict):
            exec_data = flow_data.get("agentFlowExecutedData", [])
            for node in exec_data:
                if not isinstance(node, dict):
                    continue
                label = node.get("nodeLabel", "").lower()
                if "investigador" in label or "investigat" in label:
                    content = (
                        node.get("data", {})
                            .get("output", {})
                            .get("content", "")
                    )
                    if content:
                        try:
                            return json.loads(content)
                        except Exception:
                            pass

    return {}
