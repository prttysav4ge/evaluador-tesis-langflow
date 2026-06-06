"""
Cliente HTTP para la API de Langflow — reemplazo local de Flowise.

Expone EXACTAMENTE el mismo método público que flowise/client.py para ser un
drop-in: misma firma, mismos parámetros, mismo tipo de retorno (dict).

    async call_chatflow(question, context, reference_context="",
                        session_id=None, override_config=None,
                        previous_iteration=None) -> dict

Diferencias internas respecto a Flowise:
  - Endpoint:  POST {LANGFLOW_URL}/api/v1/run/{LANGFLOW_FLOW_ID}
  - Headers:   Content-Type + x-api-key (se omite x-api-key si está vacío,
               p.ej. cuando LANGFLOW_AUTO_LOGIN=true)
  - Payload:   el JSON del backend viaja dentro de `input_value` (Langflow lo
               entrega al primer nodo del flujo como chat input).
  - Respuesta: el JSON final se extrae de
               data["outputs"][0]["outputs"][0]["results"]["message"]["text"]
               y se parsea con json.loads().

Errores: SIEMPRE se lanza RuntimeError (timeout, HTTP >= 400, JSON inválido o
estructura inesperada). El llamador en routes/query.py
(_call_langflow_with_fallback) captura RuntimeError y cae a los agentes Python.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class LangflowClient:
    """Wrapper sobre la API HTTP de Langflow, compatible con FlowiseClient."""

    def __init__(self) -> None:
        from app.config import settings

        self.base_url = settings.LANGFLOW_URL.rstrip("/")
        self.flow_id = settings.LANGFLOW_FLOW_ID
        self.api_key = settings.LANGFLOW_API_KEY
        self.superuser = settings.LANGFLOW_SUPERUSER
        self.superuser_password = settings.LANGFLOW_SUPERUSER_PASSWORD
        # 120 s — los flujos de Langflow con 6 LLM encadenados son lentos; si no
        # responde aquí, mejor caer al fallback Python que bloquear al cliente.
        self.timeout = 120.0
        # API key efectiva en uso. En modo superuser se deriva en runtime y se
        # cachea aquí; ante 401/403 se invalida y se vuelve a derivar (útil cuando
        # el servidor Langflow reinició y perdió la key — p.ej. un HF Space free).
        self._cached_api_key: Optional[str] = settings.LANGFLOW_API_KEY or None

    async def _login_and_create_key(self, client: "httpx.AsyncClient") -> str:
        """Login con el superusuario y creación de una API key (modo producción)."""
        resp = await client.post(
            f"{self.base_url}/api/v1/login",
            data={"username": self.superuser, "password": self.superuser_password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Login a Langflow falló (HTTP {resp.status_code}): {resp.text[:200]!r}"
            )
        token = resp.json().get("access_token")
        if not token:
            raise RuntimeError("Langflow no devolvió access_token en el login.")
        resp2 = await client.post(
            f"{self.base_url}/api/v1/api_key/",
            json={"name": "streamlit-evaluador"},
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp2.status_code not in (200, 201):
            raise RuntimeError(
                f"Creación de API key falló (HTTP {resp2.status_code}): {resp2.text[:200]!r}"
            )
        key = resp2.json().get("api_key")
        if not key:
            raise RuntimeError("Langflow no devolvió 'api_key' al crearla.")
        return key

    async def _ensure_api_key(self, client: "httpx.AsyncClient") -> Optional[str]:
        """
        Devuelve la API key a usar en x-api-key, según el modo configurado:
          1) LANGFLOW_API_KEY fija → se usa tal cual.
          2) Superusuario          → se deriva (login + crear key) y se cachea.
          3) Sin credenciales      → None (Langflow local con AUTO_LOGIN + skip).
        """
        if self.api_key:
            return self.api_key
        if self.superuser and self.superuser_password:
            if not self._cached_api_key:
                self._cached_api_key = await self._login_and_create_key(client)
                logger.info("🔑 API key de Langflow obtenida vía login de superusuario.")
            return self._cached_api_key
        return None

    def _headers(self, api_key: Optional[str]) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def _build_tweaks(
        self,
        question: str,
        context: str,
        reference_context: str,
        previous_iteration: Optional[str],
        override_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Construye el dict `tweaks` que inyecta los campos de cada request en los
        campos (variables) de los nodos Prompt del flujo de Langflow, en lugar de
        parsear un JSON dentro del canvas.

        Los IDs de nodo son FIJOS y los define build_flow.py al crear el flujo:
          PromptSupervisor / PromptInvestigador / PromptAuditor /
          PromptMetodologo / PromptRedactor / PromptSintesis

        Cada nodo solo recibe las variables que su prompt realmente usa (paridad
        con services/agent_service.py + prompts/agent_prompts.py). El Redactor
        recibe el contexto recortado a 800 chars, igual que build_redactor_prompt.
        """
        ctx  = context or ""
        refs = reference_context or ""
        prev = previous_iteration or ""

        tweaks: Dict[str, Any] = {
            "PromptSupervisor":   {"question": question, "context": ctx},
            "PromptInvestigador": {"question": question, "context": ctx,
                                   "reference_context": refs},
            "PromptAuditor":      {"question": question, "context": ctx},
            "PromptMetodologo":   {"question": question, "context": ctx,
                                   "reference_context": refs},
            "PromptRedactor":     {"question": question, "context": ctx[:800]},
            "PromptSintesis":     {"question": question, "previous_iteration": prev},
        }

        # Cualquier override_config explícito (p.ej. tweaks manuales) se fusiona
        # por encima, sin pisar lo anterior salvo colisión deliberada.
        if override_config:
            for node_id, fields in override_config.items():
                if isinstance(fields, dict):
                    tweaks.setdefault(node_id, {}).update(fields)
                else:
                    tweaks[node_id] = fields
        return tweaks

    async def call_chatflow(
        self,
        question: str,
        context: str,
        reference_context: str = "",
        session_id: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
        previous_iteration: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Llama al flujo de Langflow y devuelve el JSON final ya parseado (dict).

        Lanza RuntimeError ante cualquier fallo para que el fallback a agentes
        Python en routes/query.py funcione igual que con el cliente Flowise.
        """
        if not self.flow_id:
            raise RuntimeError(
                "LANGFLOW_FLOW_ID no está configurado. Crea el flujo en la UI de "
                "Langflow (http://localhost:7860) y copia su ID a .env."
            )

        payload: Dict[str, Any] = {
            # input_value es irrelevante (el flujo no tiene Chat Input cableado;
            # los datos entran por tweaks), pero el endpoint /run lo espera.
            "input_value": question,
            "output_type": "chat",
            "input_type":  "chat",
            "tweaks": self._build_tweaks(
                question, context, reference_context,
                previous_iteration, override_config,
            ),
        }
        if session_id:
            payload["session_id"] = session_id

        url = f"{self.base_url}/api/v1/run/{self.flow_id}"
        logger.info(f"📡 Llamando a Langflow: {url}")
        logger.debug(f"   context_chars: {len(context or '')}")

        # ── Request (con auth y reintento ante key inválida) ────────────────
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                api_key = await self._ensure_api_key(client)
                response = await client.post(
                    url, json=payload, headers=self._headers(api_key)
                )
                # Si la key se derivó del superusuario y el servidor la rechaza
                # (401/403) —típico tras un reinicio del Space que vació la DB—,
                # la invalidamos, derivamos una nueva y reintentamos UNA vez.
                derived = not self.api_key and bool(self.superuser)
                if response.status_code in (401, 403) and derived:
                    logger.info(
                        "🔁 Langflow rechazó la API key (HTTP "
                        f"{response.status_code}); re-derivando y reintentando…"
                    )
                    self._cached_api_key = None
                    api_key = await self._ensure_api_key(client)
                    response = await client.post(
                        url, json=payload, headers=self._headers(api_key)
                    )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
            raise RuntimeError(
                f"Langflow inaccesible ({type(exc).__name__}): {exc}"
            ) from exc

        ct = response.headers.get("content-type", "desconocido")
        logger.info(
            f"Langflow HTTP {response.status_code} | content-type: {ct} "
            f"| body: {len(response.content)} bytes"
        )

        # ── Estado HTTP ───────────────────────────────────────────────────
        if response.status_code >= 400:
            raise RuntimeError(
                f"Langflow devolvió HTTP {response.status_code}: "
                f"{response.text[:300]!r}"
            )

        # ── Parseo de la envoltura de Langflow ────────────────────────────
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Langflow no devolvió JSON válido. Content-Type: {ct}. "
                f"Primeros 300 chars: {response.text[:300]!r}"
            ) from exc

        try:
            text = data["outputs"][0]["outputs"][0]["results"]["message"]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Estructura de respuesta de Langflow inesperada "
                "(no se encontró outputs[0].outputs[0].results.message.text): "
                f"{json.dumps(data, ensure_ascii=False)[:300]!r}"
            ) from exc

        # ── El text del nodo final debe ser el JSON de la síntesis ────────
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                "El texto devuelto por Langflow no es JSON válido: "
                f"{str(text)[:300]!r}"
            ) from exc

    async def health_check(self) -> bool:
        """Verifica que el servidor Langflow esté levantado."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/health")
                return r.status_code == 200
        except Exception:
            return False


# Instancia singleton (mismo patrón que flowise_client)
langflow_client = LangflowClient()
