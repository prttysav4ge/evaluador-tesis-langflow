"""
Métricas de evaluación del ciclo de mentoría.

Stack EXACTO (las 4 que se calculan siempre):

  1. G-Eval (LLM-as-judge, estilo G-Eval) — métrica PRIMARIA de calidad. Evalúa
     el TEXTO DE SALIDA (reescrito por el Redactor) en escala 1-5 contra la
     rúbrica. La produce el JUEZ (services.judge_service), un modelo DISTINTO
     del generador para evitar sesgo. → geval_quality().

  2. Gain Score (ganancia de Hake) — métrica de PROCESO. g = (post-pre)/(máx-pre).
     `pre` y `post` son la nota de rúbrica (en PUNTOS) del texto de ENTRADA y de
     SALIDA, ambas calificadas por el MISMO juez con la MISMA rúbrica. Nunca un
     auto-score de agente. → compute_gain_score().

  3. Cosine Similarity (embeddings multilingüe e5) — GUARDRAIL semántico, NO mide
     calidad. Cálculo determinista (sin LLM). Compara ENTRADA vs SALIDA: muy alto
     = no reescribió; muy bajo = se desvió del sentido. → compute_cosine_similarity().

  4. Context Precision (componente RAG, variante SIN referencia, equivalente a
     `llm_context_precision_without_reference` de Ragas) — mide si los fragmentos
     recuperados de los LIBROS indexados son relevantes. Average Precision sobre
     veredictos de relevancia del juez. → compute_context_precision().

Métrica CONDICIONAL (5) — Iterative Consistency: solo válida con ≥2 iteraciones
equivalentes. Queda IMPLEMENTADA pero DESACTIVADA (ITERATIVE_CONSISTENCY_ENABLED).

Nota sobre Ragas: el paquete `ragas` no es instalable en este entorno
(Python 3.14 + langchain v1: ragas 0.3/0.4 requieren scikit-network sin wheel
para 3.14, y ragas 0.2 importa módulos que langchain v1 ya removió). Por eso la
métrica 4 reimplementa FIELMENTE el algoritmo de
`llm_context_precision_without_reference` con el mismo juez LLM: mismos veredictos
de relevancia por chunk y la misma fórmula de Average Precision. El resultado es
equivalente al de Ragas, sin la dependencia pesada.

Métricas ELIMINADAS en este refactor (ya NO se calculan): ROUGE, BLEU, Toulmin
automático, LDA Coherence, Answer Correctness, Cohen's Kappa y Quadratic
Weighted Kappa, y cualquier auto-score de agente usado como nota.

Estrategia de fallback: cada métrica es safe-fail (None ante error), para que la
UI muestre "—" en vez de romper el cálculo completo.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# La métrica condicional (Iterative Consistency) queda desactivada hasta que el
# flujo garantice ≥2 iteraciones EQUIVALENTES del panel. Mientras esté en False,
# compute_iteration_consistency() devuelve None y la UI no la muestra.
ITERATIVE_CONSISTENCY_ENABLED = False


# ---------------------------------------------------------------------- #
#  2/3. Cálculos DETERMINISTAS (sin LLM)                                  #
# ---------------------------------------------------------------------- #

def compute_cosine_similarity(text1: str, text2: str) -> float:
    """
    GUARDRAIL semántico: similitud coseno [0,1] sobre embeddings multilingual-e5
    (ya normalizados). Reutiliza el embedder singleton del backend. NO mide
    calidad: solo alerta si el texto no se reescribió (≈1) o se desvió (≈0).
    """
    import numpy as np

    from embeddings.embedder import embedder

    vectors = embedder.embed_documents([text1, text2])
    v1, v2 = np.array(vectors[0]), np.array(vectors[1])
    # embed_documents normaliza los vectores → cos sim = producto punto directo.
    return round(float(np.dot(v1, v2)), 4)


def compute_gain_score(
    score_before: float,
    score_after: float,
    max_score: float,
) -> Optional[float]:
    """
    Ganancia normalizada de Hake: g = (post - pre) / (máx - pre).

    `score_before`/`score_after` son la nota de rúbrica EN PUNTOS de la ENTRADA y
    la SALIDA (mismo juez, misma rúbrica). `max_score` es el máximo de las
    secciones evaluadas. Rango [-1, 1]: positivo = mejora; negativo = retroceso.

    Devuelve None si el denominador no es positivo (la entrada ya estaba en el
    máximo, o datos faltantes): el Gain no está definido ahí.
    """
    if score_before is None or score_after is None or max_score is None:
        return None
    denom = max_score - score_before
    if denom <= 0:
        return None
    return round((score_after - score_before) / denom, 4)


# ---------------------------------------------------------------------- #
#  4. Context Precision (RAG) — fiel a llm_context_precision_without_reference
# ---------------------------------------------------------------------- #

def _average_precision(verdicts: List[int]) -> Optional[float]:
    """
    Average Precision sobre una lista ordenada de veredictos 0/1, idéntica a la
    de Ragas: AP = Σ_k (P@k · v_k) / Σ_k v_k, con P@k = (Σ_{j≤k} v_j) / k.
    Devuelve 0.0 si no hay ningún relevante; None si la lista está vacía.
    """
    if not verdicts:
        return None
    total_rel = sum(verdicts)
    if total_rel == 0:
        return 0.0
    acumulado = 0
    suma_precisions = 0.0
    for k, v in enumerate(verdicts, start=1):
        if v:
            acumulado += 1
            suma_precisions += acumulado / k
    return round(suma_precisions / total_rel, 4)


async def compute_context_precision(
    question: str,
    retrieved_contexts: List[str],
    response: str = "",
    llm=None,
) -> Dict[str, Any]:
    """
    Context Precision SIN referencia sobre los fragmentos de los LIBROS.

    El juez emite un veredicto de relevancia 0/1 por fragmento (en una sola
    llamada por eficiencia) y luego se calcula el Average Precision de forma
    determinista. Equivalente a Ragas llm_context_precision_without_reference.

    Returns:
        {"score": float|None, "veredictos": [int], "detalle": [{idx,relevante,razon}], "ok": bool}
    """
    from prompts.agent_prompts import build_context_precision_prompt
    from services.judge_service import _ask_judge

    contextos = [c for c in (retrieved_contexts or []) if c and c.strip()]
    if not contextos:
        return {"score": None, "veredictos": [], "detalle": [], "ok": False}

    fragmentos = "\n\n".join(
        f"[Fragmento {i}]\n{c}" for i, c in enumerate(contextos, start=1)
    )
    prompt = build_context_precision_prompt(question, response, fragmentos)
    try:
        data = await _ask_judge(prompt, llm)
        raw = data.get("veredictos", []) or []
        # Mapea por idx para respetar el orden aunque el juez los reordene.
        por_idx: Dict[int, Dict[str, Any]] = {}
        for v in raw:
            try:
                idx = int(v.get("idx"))
            except (TypeError, ValueError, AttributeError):
                continue
            por_idx[idx] = v
        veredictos: List[int] = []
        detalle: List[Dict[str, Any]] = []
        for i in range(1, len(contextos) + 1):
            v = por_idx.get(i, {})
            rel = 1 if int(v.get("relevante", 0) or 0) == 1 else 0
            veredictos.append(rel)
            detalle.append({"idx": i, "relevante": rel, "razon": str(v.get("razon", ""))})
        score = _average_precision(veredictos)
        return {"score": score, "veredictos": veredictos, "detalle": detalle, "ok": True}
    except Exception as exc:
        logger.warning("compute_context_precision falló (%s).", exc)
        return {"score": None, "veredictos": [], "detalle": [], "ok": False}


# ---------------------------------------------------------------------- #
#  5. Iterative Consistency (CONDICIONAL — DESACTIVADA)                    #
# ---------------------------------------------------------------------- #

def compute_iteration_consistency(scores: List[float]) -> Optional[float]:
    """
    [DESACTIVADA] Consistencia entre iteraciones: proporción de puntajes dentro
    de ±1.0 del promedio. Solo es válida si el flujo corre ≥2 iteraciones
    EQUIVALENTES del panel. Mientras ITERATIVE_CONSISTENCY_ENABLED sea False,
    devuelve None y no se muestra en la UI.
    """
    if not ITERATIVE_CONSISTENCY_ENABLED:
        return None
    if not scores or len(scores) < 2:
        return None
    avg = sum(scores) / len(scores)
    within = sum(1 for s in scores if abs(s - avg) <= 1.0) / len(scores)
    return round(within, 4)


# ---------------------------------------------------------------------- #
#  Orquestador on-demand                                                   #
# ---------------------------------------------------------------------- #

async def compute_all_metrics(
    *,
    input_text: str,
    output_text: str,
    question: str,
    secciones: Optional[List[int]] = None,
    reference_chunks: Optional[List[str]] = None,
    pre_grade: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calcula las 4 métricas comparando el texto de ENTRADA (input_text) con el de
    SALIDA (output_text). Pensado para invocarse on-demand desde la Pestaña 4.

    Reusa trabajo del pipeline cuando está disponible:
      - `secciones`: secciones de rúbrica seleccionadas (las elige el Redactor).
        Si no se pasan, el juez las selecciona aquí.
      - `pre_grade`: nota de rúbrica de la ENTRADA ya calculada por el Redactor
        (evita recalificar la entrada). Si no se pasa, se califica aquí.

    Cada métrica es safe-fail (None ante error). Devuelve un dict plano + detalles.
    """
    from services import judge_service, rubric_service

    metrics: Dict[str, Any] = {
        "geval": None,
        "gain_score": None,
        "gain_detail": None,
        "cosine_similarity": None,
        "context_precision": None,
        "context_precision_detail": None,
        "iteration_consistency": compute_iteration_consistency([]),  # None (desactivada)
        "secciones": [],
        "rubrica_entrada": pre_grade,
        "rubrica_salida": None,
    }

    # ── Selección de secciones (reusa la del Redactor si vino) ────────────
    if not secciones:
        try:
            sel = await judge_service.select_rubric_sections(question, input_text)
            secciones = sel.get("secciones", [])
        except Exception as exc:
            logger.warning("metrics: selección de secciones falló (%s).", exc)
            secciones = rubric_service.normalizar_numeros(
                [s["numero"] for s in rubric_service.seccion_index()]
            )
    metrics["secciones"] = secciones

    # ── Cosine (determinista, guardrail) ──────────────────────────────────
    if input_text and output_text:
        try:
            metrics["cosine_similarity"] = compute_cosine_similarity(input_text, output_text)
        except Exception as exc:
            logger.warning("metrics: cosine falló (%s).", exc)

    # ── Calificación de rúbrica ENTRADA (pre) y SALIDA (post) ─────────────
    if pre_grade is None and input_text:
        pre_grade = await judge_service.score_against_rubric(input_text, secciones, "ENTRADA")
        metrics["rubrica_entrada"] = pre_grade

    post_grade = None
    if output_text:
        post_grade = await judge_service.score_against_rubric(output_text, secciones, "SALIDA")
        metrics["rubrica_salida"] = post_grade

    # ── Gain Score (pre/post del MISMO juez, en puntos) ───────────────────
    if pre_grade and post_grade:
        gain = compute_gain_score(
            pre_grade.get("obtenido"), post_grade.get("obtenido"), pre_grade.get("maximo")
        )
        metrics["gain_score"] = gain
        metrics["gain_detail"] = {
            "pre": pre_grade.get("obtenido"),
            "post": post_grade.get("obtenido"),
            "max": pre_grade.get("maximo"),
        }

    # ── G-Eval 1-5 de la SALIDA (métrica PRIMARIA de calidad) ─────────────
    if output_text:
        metrics["geval"] = await judge_service.geval_quality(output_text, secciones)

    # ── Context Precision sobre los LIBROS recuperados ────────────────────
    if reference_chunks:
        metrics["context_precision_detail"] = await compute_context_precision(
            question, reference_chunks, response=output_text or ""
        )
        metrics["context_precision"] = metrics["context_precision_detail"].get("score")

    return metrics
