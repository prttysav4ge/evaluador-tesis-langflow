"""
Servicio de la rúbrica especializada (Hernández-Sampieri 2018).

Responsabilidades (SIN LLM — solo datos + aritmética determinista):
  - Cargar y cachear `rubrica.json` (100 pts, 15 secciones).
  - Exponer las secciones y sus ítems.
  - Formatear un bloque de rúbrica para inyectar en el prompt del juez,
    acotado a las secciones SELECCIONADAS dinámicamente.
  - Calcular la nota de una PARTE evaluada: suma de los ítems de las secciones
    seleccionadas sobre el máximo de ESAS secciones (no sobre los 100 pts).

El razonamiento de QUÉ secciones aplican (selección dinámica) vive en
`services.judge_service` porque requiere un LLM. Este módulo solo provee los
datos y la matemática.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# rubrica.json vive en la raíz del repo (junto a streamlit_app.py).
_RUBRICA_PATH = Path(__file__).resolve().parent.parent / "rubrica.json"


@lru_cache(maxsize=1)
def load_rubrica() -> Dict[str, Any]:
    """Carga y cachea la rúbrica estructurada desde rubrica.json."""
    with open(_RUBRICA_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info(
        "📋 Rúbrica cargada: %s secciones, %s pts totales",
        data.get("numero_de_secciones"),
        data.get("puntaje_total"),
    )
    return data


def get_secciones() -> List[Dict[str, Any]]:
    """Lista completa de las 15 secciones (cada una con sus ítems)."""
    return load_rubrica().get("secciones", [])


def get_seccion(numero: int) -> Optional[Dict[str, Any]]:
    """Devuelve la sección por su número (1-15), o None si no existe."""
    for sec in get_secciones():
        if int(sec.get("numero", -1)) == int(numero):
            return sec
    return None


def seccion_index() -> List[Dict[str, Any]]:
    """
    Índice liviano {numero, nombre, puntaje_maximo} de las 15 secciones.
    Se usa para que el juez razone qué secciones aplican sin pagar el costo
    de tokens de todos los ítems.
    """
    return [
        {
            "numero": sec.get("numero"),
            "nombre": sec.get("nombre"),
            "puntaje_maximo": sec.get("puntaje_maximo"),
        }
        for sec in get_secciones()
    ]


def normalizar_numeros(numeros: List[Any]) -> List[int]:
    """
    Limpia una lista de números de sección propuesta (p.ej. por el juez):
    descarta inválidos/fuera de rango, deduplica y ordena.
    """
    validos = {int(s.get("numero")) for s in get_secciones()}
    salida: List[int] = []
    for n in numeros:
        try:
            ni = int(n)
        except (TypeError, ValueError):
            continue
        if ni in validos and ni not in salida:
            salida.append(ni)
    return sorted(salida)


def format_sections_block(numeros: List[int]) -> str:
    """
    Construye el bloque de texto de la rúbrica acotado a las secciones
    seleccionadas, listo para inyectar en el prompt del juez. Incluye cada
    ítem con su id, criterio y puntaje máximo.
    """
    secciones = [get_seccion(n) for n in normalizar_numeros(numeros)]
    secciones = [s for s in secciones if s]
    if not secciones:
        return "(ninguna sección seleccionada)"

    partes: List[str] = []
    for sec in secciones:
        cabecera = (
            f"### Sección {sec['numero']}. {sec['nombre']} "
            f"(máx {sec['puntaje_maximo']} pts)"
        )
        items = [
            f"  - [{it['id']}] (máx {it['pts_max']} pts) {it['criterio']}"
            for it in sec.get("items", [])
        ]
        partes.append(cabecera + "\n" + "\n".join(items))
    return "\n\n".join(partes)


def max_de_secciones(numeros: List[int]) -> float:
    """Suma del puntaje máximo de las secciones seleccionadas."""
    total = 0.0
    for n in normalizar_numeros(numeros):
        sec = get_seccion(n)
        if sec:
            total += float(sec.get("puntaje_maximo", 0))
    return round(total, 2)


def item_ids_de_secciones(numeros: List[int]) -> List[str]:
    """Lista de ids de ítems ('3.1', '3.2', …) de las secciones seleccionadas."""
    ids: List[str] = []
    for n in normalizar_numeros(numeros):
        sec = get_seccion(n)
        if not sec:
            continue
        ids.extend(it["id"] for it in sec.get("items", []))
    return ids


def max_de_item(item_id: str) -> Optional[float]:
    """Puntaje máximo de un ítem concreto por su id ('3.1')."""
    for sec in get_secciones():
        for it in sec.get("items", []):
            if it.get("id") == item_id:
                return float(it.get("pts_max", 0))
    return None


def compute_part_grade(
    numeros: List[int],
    item_scores: Dict[str, float],
) -> Dict[str, Any]:
    """
    Calcula la nota de la PARTE evaluada.

    Args:
        numeros: secciones seleccionadas (las que aplican a esta parte).
        item_scores: {item_id: puntaje_obtenido}. Los ítems ausentes cuentan 0.
            Cada puntaje se acota (clamp) a [0, pts_max] del ítem para evitar
            que el juez infle por encima del máximo.

    Returns:
        {
          "obtenido": float,        # suma de puntos obtenidos
          "maximo": float,          # suma del máximo de las secciones
          "porcentaje": float,      # obtenido / maximo  (0-1); 0 si max=0
          "secciones": [int],       # secciones consideradas (normalizadas)
          "por_item": {id: {"obtenido", "maximo"}},
        }
    """
    secciones = normalizar_numeros(numeros)
    maximo = max_de_secciones(secciones)
    ids_validos = set(item_ids_de_secciones(secciones))

    por_item: Dict[str, Dict[str, float]] = {}
    obtenido = 0.0
    for item_id in ids_validos:
        tope = max_de_item(item_id) or 0.0
        bruto = float(item_scores.get(item_id, 0) or 0)
        punt = max(0.0, min(bruto, tope))   # clamp a [0, pts_max]
        por_item[item_id] = {"obtenido": round(punt, 3), "maximo": tope}
        obtenido += punt

    porcentaje = (obtenido / maximo) if maximo > 0 else 0.0
    return {
        "obtenido": round(obtenido, 3),
        "maximo": round(maximo, 3),
        "porcentaje": round(porcentaje, 4),
        "secciones": secciones,
        "por_item": por_item,
    }


def build_item_reasoning(
    numeros: List[int],
    grade: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Construye el desglose por ítem que el Redactor debe exponer: por cada ítem
    de las secciones seleccionadas → punto de la rúbrica + criterio +
    calificación obtenida (sobre su máximo) + justificación del juez.

    Args:
        numeros: secciones seleccionadas.
        grade:   salida de compute_part_grade enriquecida por el juez (incluye
                 `por_item` y, si está, `justificaciones`).
    """
    por_item = grade.get("por_item", {}) or {}
    justificaciones = grade.get("justificaciones", {}) or {}
    salida: List[Dict[str, Any]] = []
    for n in normalizar_numeros(numeros):
        sec = get_seccion(n)
        if not sec:
            continue
        for it in sec.get("items", []):
            iid = it["id"]
            info = por_item.get(iid, {})
            salida.append({
                "seccion": sec["numero"],
                "seccion_nombre": sec["nombre"],
                "punto": iid,
                "criterio": it["criterio"],
                "obtenido": info.get("obtenido", 0.0),
                "maximo": info.get("maximo", float(it.get("pts_max", 0))),
                "justificacion": justificaciones.get(iid, ""),
            })
    return salida
