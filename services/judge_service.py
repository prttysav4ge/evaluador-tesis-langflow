"""
JUEZ LLM (LLM-as-judge) — el componente que califica contra la rúbrica.

Un ÚNICO juez (modelo `GEVAL_JUDGE_MODEL`, distinto del modelo generador para
evitar sesgo de autoevaluación) cumple tres funciones:

  1. select_rubric_sections()  → razona qué secciones de la rúbrica aplican a la
                                  PARTE evaluada (selección dinámica).
  2. score_against_rubric()    → califica un texto ítem por ítem contra las
                                  secciones seleccionadas (regla pts_max/50%/0).
                                  Se usa para la ENTRADA (umbral del Redactor +
                                  'pre' del Gain) y para la SALIDA ('post').
  3. geval_quality()           → G-Eval 1-5 de calidad del TEXTO DE SALIDA.

Reutiliza el retry con backoff y el parseo tolerante de JSON de agent_service
para no duplicar la gestión de rate-limit de Groq.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from prompts.agent_prompts import (
    build_geval_prompt,
    build_score_rubrica_prompt,
    build_seleccion_secciones_prompt,
)
from services import rubric_service
from services.agent_service import _ainvoke_with_retry, _parse_json

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = SystemMessage(
    content=(
        "Eres un evaluador académico riguroso y objetivo que actúa como JUEZ de "
        "una rúbrica metodológica. Respondes ÚNICAMENTE en JSON válido."
    )
)


def _get_judge_llm():
    """
    Construye el LLM JUEZ usando `GEVAL_JUDGE_MODEL` (distinto del generador).
    Sigue el mismo orden de proveedor que agent_service (groq → openai → ollama),
    pero forzando el modelo juez. temperature=0 para calificaciones reproducibles.
    """
    from app.config import settings
    from langchain_openai import ChatOpenAI

    provider = settings.LLM_PROVIDER.lower()
    use_groq = (provider == "groq") or (provider == "auto" and settings.GROQ_API_KEY)
    if use_groq:
        if not settings.GROQ_API_KEY:
            raise ValueError("Juez: GROQ_API_KEY no está configurado en .env.")
        return ChatOpenAI(
            api_key=settings.GROQ_API_KEY,
            model=settings.GEVAL_JUDGE_MODEL,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.0,
            max_tokens=900,
            max_retries=0,
        )

    use_openai = (provider == "openai") or (provider == "auto" and settings.OPENAI_API_KEY)
    if use_openai:
        if not settings.OPENAI_API_KEY:
            raise ValueError("Juez: OPENAI_API_KEY no está configurado en .env.")
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=0.0,
            max_tokens=900,
            max_retries=0,
        )

    # Ollama
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=0.0,
        num_predict=900,
    )


async def _ask_judge(prompt: str, llm=None) -> Dict[str, Any]:
    """Envía un prompt al juez y devuelve el JSON parseado (tolerante)."""
    llm = llm or _get_judge_llm()
    response = await _ainvoke_with_retry(llm, [_JUDGE_SYSTEM, HumanMessage(content=prompt)])
    return _parse_json(response.content)


# ---------------------------------------------------------------------- #
#  1. Selección dinámica de secciones                                     #
# ---------------------------------------------------------------------- #

async def select_rubric_sections(
    question: str, context: str, llm=None
) -> Dict[str, Any]:
    """
    El juez razona qué secciones de la rúbrica aplican a la parte evaluada.

    Returns:
        {"secciones": [int], "razon": str, "modo": "llm"|"fallback_todas"}
    Si el juez falla o no devuelve secciones válidas, cae a TODAS las secciones
    (degradación graciosa: nunca deja la evaluación sin rúbrica).
    """
    indice = "\n".join(
        f"  {s['numero']}. {s['nombre']} ({s['puntaje_maximo']} pts)"
        for s in rubric_service.seccion_index()
    )
    prompt = build_seleccion_secciones_prompt(question, context, indice)
    try:
        data = await _ask_judge(prompt, llm)
        seleccion = rubric_service.normalizar_numeros(
            data.get("secciones_aplicables", []) or []
        )
        if seleccion:
            return {
                "secciones": seleccion,
                "razon": str(data.get("razon", "")).strip(),
                "modo": "llm",
            }
        logger.warning("Juez no devolvió secciones válidas; usando todas.")
    except Exception as exc:
        logger.warning("select_rubric_sections falló (%s); usando todas.", exc)

    todas = rubric_service.normalizar_numeros(
        [s["numero"] for s in rubric_service.seccion_index()]
    )
    return {"secciones": todas, "razon": "Fallback: se evalúan todas las secciones.",
            "modo": "fallback_todas"}


# ---------------------------------------------------------------------- #
#  2. Calificación contra la rúbrica (entrada y salida)                   #
# ---------------------------------------------------------------------- #

async def score_against_rubric(
    texto: str,
    secciones: List[int],
    etiqueta_texto: str = "TEXTO A CALIFICAR",
    llm=None,
) -> Dict[str, Any]:
    """
    Califica `texto` contra las `secciones` seleccionadas, ítem por ítem.

    Returns el dict de compute_part_grade enriquecido con:
        {... , "justificaciones": {item_id: str}, "ok": bool}
    Los puntajes se acotan a [0, pts_max] en rubric_service.compute_part_grade.
    Si el juez falla, devuelve ok=False con obtenido=0 (no rompe el pipeline).
    """
    bloque = rubric_service.format_sections_block(secciones)
    prompt = build_score_rubrica_prompt(texto, bloque, etiqueta_texto)
    try:
        data = await _ask_judge(prompt, llm)
        item_scores = data.get("items", {}) or {}
        # Normaliza claves a str y valores a float (tolerante a "3.1": "1.5").
        limpio: Dict[str, float] = {}
        for k, v in item_scores.items():
            try:
                limpio[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        grade = rubric_service.compute_part_grade(secciones, limpio)
        grade["justificaciones"] = data.get("justificaciones", {}) or {}
        grade["ok"] = True
        return grade
    except Exception as exc:
        logger.warning("score_against_rubric falló (%s).", exc)
        grade = rubric_service.compute_part_grade(secciones, {})
        grade["justificaciones"] = {}
        grade["ok"] = False
        return grade


# ---------------------------------------------------------------------- #
#  3. G-Eval 1-5 de calidad del texto de SALIDA                           #
# ---------------------------------------------------------------------- #

async def geval_quality(
    texto_salida: str,
    secciones: List[int],
    llm=None,
) -> Dict[str, Any]:
    """
    G-Eval: calidad del TEXTO DE SALIDA en escala 1-5 usando la rúbrica como
    criterio. Métrica PRIMARIA de calidad (no se mezcla con la escala de puntos).

    Returns:
        {"score": float|None (1-5), "razonamiento": str, "ok": bool}
    """
    bloque = rubric_service.format_sections_block(secciones)
    prompt = build_geval_prompt(texto_salida, bloque)
    try:
        data = await _ask_judge(prompt, llm)
        raw = data.get("score")
        score = max(1.0, min(5.0, float(raw))) if raw is not None else None
        return {
            "score": score,
            "razonamiento": str(data.get("razonamiento", "")).strip(),
            "ok": score is not None,
        }
    except Exception as exc:
        logger.warning("geval_quality falló (%s).", exc)
        return {"score": None, "razonamiento": "", "ok": False}
