"""
Prompts de los 6 agentes especializados + generador de texto sugerido.

Criterios de diseño:
  - Salida SIEMPRE en JSON válido (sin texto extra), salvo build_texto_sugerido_prompt
    que pide texto académico plano listo para pegar en la tesis.
  - Bajo consumo de tokens: reciben sólo lo necesario de la memoria acumulada
  - Cada agente tiene un ROL único y claro
  - El agente Síntesis y Consenso sintetiza todo para el estudiante
"""
from __future__ import annotations
import json
from typing import Any, Dict


# ====================================================================== #
#  AGENTE 1 — Supervisor (triage inicial del panel)                      #
# ====================================================================== #

def build_mentor_intake_prompt(question: str, context: str) -> str:
    return f"""Eres el SUPERVISOR del panel multiagente de evaluación de tesis universitarias.

ROL: Hacer el triage inicial del contexto recuperado y la pregunta del evaluador, y delimitar qué van a evaluar los demás agentes del panel (Investigador, Auditor, Metodólogo, Redactor, Síntesis).

=== CONTEXTO RECUPERADO DE LA TESIS ===
{context}

=== PREGUNTA DEL EVALUADOR ===
{question}

=== INSTRUCCIONES ===
1. Lee el contexto recuperado de la tesis con atención.
2. Identifica el tema central y la sección académica presente.
3. Evalúa si el contexto recuperado es suficiente para responder la pregunta.
4. Identifica los 3 aspectos clave que deben evaluarse.
5. Señala limitaciones del contexto recuperado (si las hay).

RESPONDE ÚNICAMENTE en formato JSON válido, sin texto adicional antes ni después:
{{
  "tema_identificado": "tema central de la tesis en 1 oración",
  "seccion_relevante": "nombre de la sección académica identificada",
  "pertinencia": "alta|media|baja",
  "contexto_suficiente": true,
  "aspectos_clave": ["aspecto1", "aspecto2", "aspecto3"],
  "evaluacion_inicial": "evaluación concisa del fragmento en 2-3 oraciones",
  "limitaciones_contexto": ["limitacion1"],
  "flags": []
}}"""


# ====================================================================== #
#  AGENTE 2 — Investigador (análisis de calidad investigativa)           #
# ====================================================================== #

def build_investigador_prompt(
    question: str,
    context: str,
    memory: Dict[str, Any],
    reference_context: str = "",
) -> str:
    mentor_summary = json.dumps(memory.get("mentor_intake", {}), ensure_ascii=False)
    refs_block = (
        f"\n=== BIBLIOTECA METODOLÓGICA (libros de referencia) ===\n{reference_context}\n"
        if reference_context else ""
    )
    refs_instr = (
        "6. Si la Biblioteca aporta principios relevantes, citalos al respaldar tus "
        "observaciones (ej. 'según Hernández Sampieri...'). Privilegia la coincidencia "
        "entre lo que dice la tesis y lo que recomienda la literatura metodológica.\n"
        if reference_context else ""
    )
    return f"""Eres el AGENTE INVESTIGADOR especializado en análisis de investigación académica.

ROL: Analizar la calidad investigativa del fragmento de tesis.

=== PREGUNTA ===
{question}

=== CONTEXTO DE LA TESIS ===
{context}
{refs_block}
=== EVALUACIÓN PREVIA (Supervisor) ===
{mentor_summary}

=== INSTRUCCIONES ===
1. Analiza la solidez de la argumentación e investigación.
2. Evalúa si hay respaldo teórico y bibliográfico.
3. Identifica fortalezas y debilidades investigativas concretas.
4. Sugiere 2-3 mejoras específicas y realizables.
5. Asigna una puntuación de 0 a 10.
{refs_instr}
RESPONDE ÚNICAMENTE en formato JSON válido:
{{
  "fortalezas": ["fortaleza1", "fortaleza2"],
  "debilidades": ["debilidad1", "debilidad2"],
  "respaldo_teorico": "adecuado|parcial|insuficiente",
  "relevancia_cientifica": "alta|media|baja",
  "sugerencias": ["sugerencia1", "sugerencia2"],
  "puntuacion": 7.5,
  "comentario": "análisis investigativo en 2-3 oraciones",
  "biblioteca_aplicada": ["principio/cita del libro X usado", "..."]
}}"""


# ====================================================================== #
#  AGENTE 3 — Auditor (rigor académico)                                  #
# ====================================================================== #

def build_auditor_prompt(
    question: str, context: str, memory: Dict[str, Any]
) -> str:
    prev_summary = json.dumps(
        {k: memory[k] for k in ["mentor_intake", "investigador"] if k in memory},
        ensure_ascii=False,
    )
    return f"""Eres el AGENTE AUDITOR de rigor académico y calidad científica.

ROL: Auditar la coherencia, consistencia y rigor del fragmento de tesis.

=== PREGUNTA ===
{question}

=== CONTEXTO DE LA TESIS ===
{context}

=== EVALUACIONES PREVIAS ===
{prev_summary}

=== INSTRUCCIONES ===
1. Verifica la coherencia interna del argumento.
2. Detecta inconsistencias, contradicciones o afirmaciones sin soporte.
3. Evalúa el uso correcto de terminología académica.
4. Identifica brechas o vacíos en el desarrollo.
5. Señala los problemas críticos que deben corregirse.

RESPONDE ÚNICAMENTE en formato JSON válido:
{{
  "nivel_rigor": "alto|medio|bajo",
  "coherencia_interna": "alta|media|baja",
  "inconsistencias": ["inconsistencia1"],
  "terminologia": "correcta|parcialmente_correcta|incorrecta",
  "brechas_detectadas": ["brecha1"],
  "problemas_criticos": ["problema1"],
  "puntuacion_rigor": 7.0,
  "recomendaciones": ["recomendacion1", "recomendacion2"]
}}"""


# ====================================================================== #
#  AGENTE 4 — Metodológico (análisis del marco metodológico)             #
# ====================================================================== #

def build_metodologico_prompt(
    question: str,
    context: str,
    memory: Dict[str, Any],
    reference_context: str = "",
) -> str:
    prev_summary = json.dumps(
        {k: memory[k] for k in ["mentor_intake", "investigador", "auditor"] if k in memory},
        ensure_ascii=False,
    )
    refs_block = (
        f"\n=== BIBLIOTECA METODOLÓGICA (libros de referencia) ===\n{reference_context}\n"
        if reference_context else ""
    )
    refs_instr = (
        "6. CRÍTICO: contrasta lo que hace la tesis con las recomendaciones de los libros "
        "de la Biblioteca. Si difieren, indícalo explícitamente. Si coinciden, refuerza la "
        "evaluación citando la fuente. La Biblioteca es tu fuente de verdad metodológica.\n"
        if reference_context else ""
    )
    return f"""Eres el METODÓLOGO del panel multiagente, especializado en marcos y diseños de investigación científica.

ROL: Evaluar el enfoque y diseño metodológico presente en el fragmento de tesis.

=== PREGUNTA ===
{question}

=== CONTEXTO DE LA TESIS ===
{context}
{refs_block}
=== EVALUACIONES PREVIAS ===
{prev_summary}

=== INSTRUCCIONES ===
1. Identifica el enfoque metodológico (cualitativo/cuantitativo/mixto).
2. Evalúa si el diseño de investigación es adecuado al problema.
3. Analiza instrumentos o técnicas de recolección mencionados.
4. Identifica limitaciones metodológicas explícitas o implícitas.
5. Sugiere ajustes metodológicos concretos.
{refs_instr}
RESPONDE ÚNICAMENTE en formato JSON válido:
{{
  "enfoque": "cualitativo|cuantitativo|mixto|no_especificado",
  "tipo_investigacion": "descriptiva|explicativa|correlacional|experimental|exploratoria|mixta",
  "diseno": "descripción del diseño identificado",
  "adecuacion_metodologica": "alta|media|baja",
  "instrumentos_identificados": ["instrumento1"],
  "limitaciones_metodologicas": ["limitacion1"],
  "sugerencias_metodologicas": ["sugerencia1"],
  "puntuacion_metodologia": 7.0,
  "comentario": "análisis metodológico en 2-3 oraciones",
  "alineacion_con_biblioteca": "alta|media|baja|no_aplica",
  "citas_biblioteca": ["principio metodológico usado del libro X", "..."]
}}"""


# ====================================================================== #
#  AGENTE 5 — Redactor (mejora de escritura académica)                   #
# ====================================================================== #

def build_redactor_prompt(
    question: str, context: str, memory: Dict[str, Any]
) -> str:
    prev_summary = json.dumps(
        {
            k: memory[k]
            for k in ["mentor_intake", "auditor", "metodologico"]
            if k in memory
        },
        ensure_ascii=False,
    )
    return f"""Eres el AGENTE REDACTOR especializado en escritura académica en español.

ROL: Mejorar la calidad de redacción y presentación del fragmento de tesis más relevante.

=== PREGUNTA ===
{question}

=== CONTEXTO DE LA TESIS (fragmento a mejorar) ===
{context[:800]}

=== EVALUACIONES PREVIAS ===
{prev_summary}

=== INSTRUCCIONES ===
1. Selecciona el fragmento más relevante del contexto para mejorar.
2. Reescribe el fragmento con mayor claridad, precisión y estilo académico.
3. Mantén EXACTAMENTE el significado original; solo mejora la forma.
4. Lista los cambios específicos realizados.
5. Provee sugerencias generales de escritura para la tesis completa.

RESPONDE ÚNICAMENTE en formato JSON válido:
{{
  "fragmento_original": "fragmento seleccionado del contexto",
  "fragmento_mejorado": "versión mejorada y clara del fragmento",
  "cambios_realizados": ["cambio1", "cambio2", "cambio3"],
  "nivel_escritura_original": "alto|medio|bajo",
  "sugerencias_generales": ["sugerencia1", "sugerencia2"],
  "comentario": "comentario sobre la calidad de escritura en 2-3 oraciones"
}}"""


# ====================================================================== #
#  AGENTE 6 — Síntesis y Consenso (cierre del panel multiagente)         #
# ====================================================================== #

def build_mentor_final_prompt(
    question: str,
    memory: Dict[str, Any],
    previous_iteration: str | None = None,
) -> str:
    # Serialización compacta (sin indent) para reducir el tamaño del prompt
    # y evitar que el agente 6 reciba >2000 tokens de contexto de agentes previos.
    full_memory = json.dumps(memory, ensure_ascii=False, separators=(",", ":"))

    # Bloque opcional con la síntesis de la iteración anterior. Si está
    # presente, el agente debe refinarla, no repetirla. Si está vacío, el
    # prompt funciona idéntico al original (primera iteración).
    iter_block = (
        f"\n=== SÍNTESIS DE LA ITERACIÓN ANTERIOR ===\n{previous_iteration}\n"
        if previous_iteration else ""
    )
    iter_extra_instr = (
        "10. CRÍTICO: recibiste la SÍNTESIS DE LA ITERACIÓN ANTERIOR. Tu tarea NO es "
        "repetirla — es refinarla. Conserva lo que sigue siendo válido, agudiza lo que "
        "quedó genérico, ajusta puntuación si el panel reveló matices nuevos, y revisita "
        "el debate/consenso/disenso para incorporar precisiones. Cada iteración del panel "
        "debe agregar valor.\n"
        if previous_iteration else ""
    )

    return f"""Eres SÍNTESIS Y CONSENSO, el agente final del panel multiagente de evaluación de tesis.

ROL: Integrar las evaluaciones del Supervisor, Investigador, Auditor, Metodólogo y Redactor en (a) un feedback pedagógico final y (b) la transcripción del DEBATE entre las 3 perspectivas centrales del panel.

Las 3 perspectivas del debate son:
  • Perspectiva FORMAL       → la voz del Auditor (rigor, coherencia, citas).
  • Perspectiva METODOLÓGICA → la voz del Metodólogo (diseño, instrumentos, validez).
  • Perspectiva CONTEXTUAL   → la voz del Investigador (literatura, antecedentes, evidencia).

=== PREGUNTA ORIGINAL ===
{question}

=== EVALUACIONES COMPLETAS DEL PANEL ===
{full_memory}
{iter_block}
=== INSTRUCCIONES ===
1. Sintetiza los hallazgos más importantes de TODOS los agentes previos.
2. Identifica los 3 puntos fuertes principales de la tesis.
3. Lista las 3 áreas de mejora más urgentes.
4. Genera recomendaciones concretas y priorizadas (máximo 5).
5. Calcula la puntuación general (promedio ponderado de las puntuaciones previas).
6. Redacta un mensaje constructivo, motivador y pedagógico para el estudiante.
7. Indica el SIGUIENTE PASO concreto más importante.
8. Reconstruye el DEBATE: resume en 2-3 oraciones lo que dijo cada una de las 3 perspectivas (formal/metodológica/contextual) y produce una síntesis breve.
9. Lista 2-4 puntos de CONSENSO (donde las 3 perspectivas coinciden) y 2-4 puntos de DISENSO (donde 2 perspectivas chocan o se contradicen). Sé honesto: si no hay disenso real, devolvé [].
{iter_extra_instr}

RESPONDE ÚNICAMENTE en formato JSON válido:
{{
  "resumen_ejecutivo": "resumen claro en 3-5 oraciones para el estudiante",
  "puntos_fuertes": ["punto1", "punto2", "punto3"],
  "areas_mejora": ["area1", "area2", "area3"],
  "recomendaciones_priorizadas": [
    {{"prioridad": 1, "recomendacion": "...", "justificacion": "..."}},
    {{"prioridad": 2, "recomendacion": "...", "justificacion": "..."}},
    {{"prioridad": 3, "recomendacion": "...", "justificacion": "..."}}
  ],
  "puntuacion_general": 7.2,
  "nivel_tesis": "excelente|buena|aceptable|necesita_mejoras|insuficiente",
  "mensaje_pedagogico": "mensaje motivador y constructivo para el estudiante",
  "siguiente_paso": "acción concreta más importante que debe realizar ahora",
  "debate": {{
    "perspectiva_formal":       "resumen 2-3 oraciones de lo que dijo el Auditor",
    "perspectiva_metodologica": "resumen 2-3 oraciones de lo que dijo el Metodólogo",
    "perspectiva_contextual":   "resumen 2-3 oraciones de lo que dijo el Investigador",
    "sintesis":                 "síntesis breve de cómo las 3 perspectivas se integran o se tensionan"
  }},
  "consenso": [
    "punto donde las 3 perspectivas coinciden 1",
    "punto donde las 3 perspectivas coinciden 2"
  ],
  "disenso": [
    "punto donde 2 perspectivas chocan o se contradicen 1",
    "(si no hay disenso real, devolver array vacío)"
  ]
}}"""


# ====================================================================== #
#  TEXTO SUGERIDO — Reescritura mejorada de la sección analizada         #
# ====================================================================== #

def build_texto_sugerido_prompt(
    original_context: str,
    question: str,
    final_evaluation: Dict[str, Any],
    investigador_findings: Dict[str, Any],
) -> str:
    """
    Construye el prompt para generar un texto académico mejorado que
    reemplace la sección analizada.  El Investigador es el agente clave:
    sus debilidades y sugerencias guían qué debe cambiar en el contenido.
    La evaluación final aporta las recomendaciones priorizadas y áreas de mejora.
    """
    # ── Datos de la Síntesis (o evaluación Flowise) ────────────────────────
    areas_mejora      = final_evaluation.get("areas_mejora", [])
    puntos_fuertes    = final_evaluation.get("puntos_fuertes", [])
    recomendaciones   = final_evaluation.get("recomendaciones_priorizadas", [])

    # ── Datos del Investigador ────────────────────────────────────────────
    debilidades_inv   = investigador_findings.get("debilidades", [])
    sugerencias_inv   = investigador_findings.get("sugerencias", [])
    respaldo          = investigador_findings.get("respaldo_teorico", "")

    # ── Formateo de secciones opcionales ─────────────────────────────────
    def bullet(items):
        return "\n".join(f"  • {i}" for i in items) if items else "  • (no especificado)"

    recs_text = "\n".join(
        f"  [{r.get('prioridad', i + 1)}] {r.get('recomendacion', str(r))}"
        + (f"\n      → {r['justificacion']}" if r.get("justificacion") else "")
        for i, r in enumerate(recomendaciones[:4])
    ) if recomendaciones else "  [1] Mejorar profundidad argumentativa y respaldo teórico"

    investigador_block = ""
    if debilidades_inv or sugerencias_inv:
        investigador_block = f"""
=== HALLAZGOS DEL AGENTE INVESTIGADOR ===
Debilidades del contenido que DEBES corregir:
{bullet(debilidades_inv)}

Sugerencias de investigación para enriquecer el texto:
{bullet(sugerencias_inv)}
{f"Nivel de respaldo teórico actual: {respaldo}" if respaldo else ""}
"""

    return f"""Eres un EXPERTO EN REDACCIÓN ACADÉMICA universitaria en español.

=== TEXTO ORIGINAL A MEJORAR ===
{original_context}

=== EVALUACIÓN QUE SE USÓ COMO BASE ===
Pregunta evaluada: {question}

Puntos fuertes a CONSERVAR:
{bullet(puntos_fuertes)}

Áreas de mejora a CORREGIR:
{bullet(areas_mejora)}

Recomendaciones priorizadas a IMPLEMENTAR:
{recs_text}
{investigador_block}
=== TU TAREA ===
Reescribe el texto original produciendo una versión mejorada directamente
usable en el documento de tesis.

REGLAS ESTRICTAS:
1. Conserva la misma sección académica y tema del original.
2. Corrige CADA área de mejora listada.
3. Implementa las recomendaciones, en especial las de mayor prioridad.
4. Integra las sugerencias del Investigador para fortalecer el argumento.
5. Mantén y refuerza los puntos fuertes identificados.
6. Usa lenguaje académico formal en español; mejora la cohesión y precisión.
7. La extensión debe ser igual o mayor a la del original.
8. NO inventes datos, estadísticas ni citas que no figuren en el texto original.
9. Devuelve ÚNICAMENTE el texto mejorado: sin títulos, sin explicaciones,
   sin formato markdown, listo para copiar y pegar en la tesis."""


# ====================================================================== #
#  JUEZ LLM (LLM-as-judge) — selección de secciones, calificación, G-Eval #
# ====================================================================== #

def build_seleccion_secciones_prompt(
    question: str, context: str, indice_secciones: str
) -> str:
    """
    Pide al JUEZ razonar qué secciones de la rúbrica aplican a la PARTE de la
    tesis que se está evaluando. Si se evalúa la tesis completa, deben aplicar
    todas; si es una parte (p.ej. 'Formulación del problema'), solo las
    secciones pertinentes.
    """
    return f"""Eres el JUEZ de una rúbrica metodológica de tesis (Hernández-Sampieri 2018).

TAREA: Decidir QUÉ secciones de la rúbrica corresponden evaluar para la parte de
la tesis que se te entrega. NO califiques todavía: solo selecciona las secciones
aplicables y justifica brevemente.

REGLAS:
- Si el texto/pregunta abarca la TESIS COMPLETA, selecciona todas las secciones.
- Si es una PARTE concreta (p.ej. solo la formulación del problema), selecciona
  ÚNICAMENTE las secciones que evalúan esa parte. No incluyas secciones de
  referencias, marco teórico, instrumentos, etc. si no están presentes.
- Razona a partir del CONTENIDO real del texto, no solo del título.

=== PREGUNTA / INSTRUCCIÓN DEL EVALUADOR ===
{question}

=== TEXTO DE LA TESIS A EVALUAR (fragmento) ===
{context[:2500]}

=== ÍNDICE DE SECCIONES DE LA RÚBRICA (número · nombre · pts máx) ===
{indice_secciones}

RESPONDE ÚNICAMENTE en JSON válido:
{{
  "secciones_aplicables": [3],
  "razon": "1-2 oraciones explicando por qué esas secciones (y no otras)."
}}"""


def build_score_rubrica_prompt(
    texto: str, bloque_rubrica: str, etiqueta_texto: str = "TEXTO A CALIFICAR"
) -> str:
    """
    Pide al JUEZ calificar un texto contra las secciones seleccionadas de la
    rúbrica, ítem por ítem, según la regla pts_max / 50% / 0. Se usa tanto para
    la ENTRADA (decidir umbral del Redactor + 'pre' del Gain) como para la
    SALIDA ('post' del Gain). El MISMO juez califica ambos para que el Gain sea
    comparable.
    """
    return f"""Eres el JUEZ de una rúbrica metodológica de tesis (Hernández-Sampieri 2018).

TAREA: Calificar el texto contra CADA ítem de las secciones de la rúbrica que se
listan. Para cada ítem asigna:
  - el puntaje MÁXIMO del ítem si el criterio SE CUMPLE COMPLETAMENTE,
  - la MITAD del máximo si se cumple PARCIALMENTE,
  - 0 si NO se cumple.
No inventes ítems ni excedas el máximo de cada ítem. Sé estricto y objetivo.

=== {etiqueta_texto} ===
{texto}

=== SECCIONES DE LA RÚBRICA A APLICAR (con el máximo de cada ítem) ===
{bloque_rubrica}

RESPONDE ÚNICAMENTE en JSON válido. Las claves de 'items' y 'justificaciones'
deben ser los ids de los ítems (p.ej. "3.1"):
{{
  "items": {{ "3.1": 1.5, "3.2": 1.0 }},
  "justificaciones": {{ "3.1": "por qué ese puntaje en 1 oración" }}
}}"""


def build_geval_prompt(texto_salida: str, bloque_rubrica: str) -> str:
    """
    G-Eval (estilo form-filling con cadena de razonamiento): el JUEZ evalúa la
    CALIDAD del TEXTO DE SALIDA (reescrito por el Redactor) en una escala 1-5,
    usando la rúbrica especializada como criterio. Es la métrica PRIMARIA de
    calidad y NO se mezcla con la escala de puntos de la rúbrica.
    """
    return f"""Eres el JUEZ (LLM-as-judge, estilo G-Eval) de calidad metodológica de tesis.

TAREA: Evaluar la CALIDAD del siguiente TEXTO DE SALIDA (una sección de tesis ya
reescrita) en una escala de 1 a 5, tomando como criterio la rúbrica especializada.

PASOS DE EVALUACIÓN (razónalos antes de puntuar):
1. Lee el texto de salida con atención.
2. Para cada sección de la rúbrica aplicable, valora qué tan bien la cumple el texto.
3. Integra esas valoraciones en un juicio global de calidad.

ESCALA (1-5):
  5 = Excelente: cumple plenamente los criterios de la rúbrica, redacción precisa.
  4 = Bueno: cumple la mayoría con observaciones menores.
  3 = Aceptable: cumple parcialmente; deficiencias notorias pero no graves.
  2 = Deficiente: incumple varios criterios clave.
  1 = Insuficiente: no cumple los criterios mínimos.

=== TEXTO DE SALIDA A EVALUAR ===
{texto_salida}

=== RÚBRICA (criterios de referencia) ===
{bloque_rubrica}

RESPONDE ÚNICAMENTE en JSON válido:
{{
  "razonamiento": "2-4 oraciones justificando la nota según la rúbrica.",
  "score": 4
}}"""


def build_context_precision_prompt(
    question: str, response: str, fragmentos: str
) -> str:
    """
    Veredicto de relevancia por fragmento para Context Precision (componente
    RAG, variante SIN referencia). Replica el juicio de
    `llm_context_precision_without_reference` de Ragas: para CADA fragmento
    recuperado de los libros, decide si fue ÚTIL para responder la pregunta.
    El Average Precision se calcula después de forma determinista a partir de
    estos veredictos (no lo calcula el LLM).
    """
    resp_block = (
        f"\n=== RESPUESTA/EVALUACIÓN PRODUCIDA ===\n{response}\n" if response else ""
    )
    return f"""Eres un verificador de relevancia de contexto para un sistema RAG.

TAREA: Para CADA fragmento recuperado de los libros de metodología, decide si el
fragmento fue ÚTIL/RELEVANTE para responder la PREGUNTA (y, si se da, para
sustentar la respuesta producida). Responde 1 si es relevante, 0 si no lo es.
Juzga la utilidad real del contenido, no su parecido superficial.

=== PREGUNTA ===
{question}
{resp_block}
=== FRAGMENTOS RECUPERADOS (en orden) ===
{fragmentos}

RESPONDE ÚNICAMENTE en JSON válido. 'veredictos' debe tener un objeto por
fragmento, en el MISMO orden, con su índice 1-based:
{{
  "veredictos": [
    {{"idx": 1, "relevante": 1, "razon": "1 frase"}},
    {{"idx": 2, "relevante": 0, "razon": "1 frase"}}
  ]
}}"""
