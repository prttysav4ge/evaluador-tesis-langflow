"""
Configuración central del proyecto via variables de entorno.
Carga automáticamente el archivo .env
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # ----- SERVIDOR -----
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # ----- LLM -----
    # El proveedor se elige automáticamente según las claves disponibles:
    # Groq → OpenAI → Ollama (en ese orden de prioridad).
    # Puedes forzar uno con LLM_PROVIDER=groq|openai|ollama.
    LLM_PROVIDER: str = "auto"   # "auto" | "groq" | "openai" | "ollama"

    # Groq (recomendado: usa los mismos modelos que Flowise, sin coste extra)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"   # mismo modelo que el Agentflow

    # ----- JUEZ LLM (LLM-as-judge / G-Eval) -----
    # Modelo que actúa como JUEZ de la rúbrica. DEBE ser distinto del modelo
    # generador (GROQ_MODEL, p.ej. llama-4-scout) para evitar sesgo de
    # autoevaluación. Se usa para: (a) seleccionar dinámicamente las secciones
    # de la rúbrica aplicables, (b) calificar ENTRADA y SALIDA contra la rúbrica
    # (alimenta el umbral del Redactor y el Gain Score), (c) el G-Eval 1-5 de
    # calidad del texto de salida. Default: el modelo Groq más liviano (menos
    # tokens). Se eligió llama-3.3-70b-versatile porque es el único que calificó
    # de forma calibrada en las pruebas (el 8b daba notas erráticas 0%↔65% sobre
    # el mismo texto; gpt-oss-20b daba 100% indulgente). Es distinto del modelo
    # generador (scout), así que mantiene la independencia anti-sesgo. Solo se
    # invoca on-demand (no en cada evaluación), por lo que el coste extra es acotado.
    GEVAL_JUDGE_MODEL: str = "llama-3.3-70b-versatile"
    # Umbral (fracción 0-1) sobre el máximo de las secciones evaluadas: si la
    # nota de la ENTRADA lo alcanza, el Redactor NO reescribe (solo pule).
    REDACTOR_UMBRAL: float = 0.90

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # ----- LANGFLOW (motor de flujos local en Docker) -----
    # El backend corre en el host y alcanza a Langflow por el puerto publicado
    # del contenedor (ver docker-compose.yml).
    USE_LANGFLOW: bool = True
    LANGFLOW_URL: str = "http://localhost:7860"
    # Acepta el UUID del flujo o su endpoint_name estable (p.ej. "evaluador-tesis").
    LANGFLOW_FLOW_ID: str = ""
    # Auth del endpoint /run. Tres modos (ver langflow/client.py):
    #   1) LANGFLOW_API_KEY fija    → se envía como x-api-key (puede expirar si el
    #      servidor tiene almacenamiento efímero, p.ej. un HF Space gratis).
    #   2) Credenciales de superuser → el cliente hace login y CREA una API key en
    #      runtime; la regenera si el servidor reinició (robusto en HF Space).
    #   3) Ninguna                  → sin auth (Langflow local con AUTO_LOGIN + skip).
    LANGFLOW_API_KEY: str = ""
    LANGFLOW_SUPERUSER: str = ""
    LANGFLOW_SUPERUSER_PASSWORD: str = ""

    # ----- EMBEDDINGS -----
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"

    # ----- CHROMADB -----
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "academic_thesis"

    # ----- RAG -----
    TOP_K: int = 5

    # ----- CHUNKING -----
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
