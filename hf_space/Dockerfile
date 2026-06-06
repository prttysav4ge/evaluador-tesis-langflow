# ============================================================================
# Imagen del Hugging Face Space: Langflow con el flujo del Evaluador HORNEADO.
# Validado con Langflow 1.9.5 (pin por reproducibilidad — :latest puede romper).
# ============================================================================
FROM langflowai/langflow:1.9.5

# Flujo horneado SIN secretos. La GROQ_API_KEY NO viaja aquí: cada nodo Groq la
# resuelve como variable global "GROQ_API_KEY", que Langflow crea desde el
# entorno (ver LANGFLOW_VARIABLES_TO_GET_FROM_ENVIRONMENT). Generado por
# bake_flow.py. Se carga al arrancar vía LANGFLOW_LOAD_FLOWS_PATH.
COPY flows/ /app/flows/

# Config NO secreta. Los SECRETOS llegan como Secrets del Space (variables de
# entorno en runtime) y NO se hornean en la imagen:
#   - GROQ_API_KEY
#   - LANGFLOW_SUPERUSER
#   - LANGFLOW_SUPERUSER_PASSWORD
#   - LANGFLOW_SECRET_KEY   (recomendado: clave fija para cifrar variables)
ENV LANGFLOW_AUTO_LOGIN=false \
    LANGFLOW_LOAD_FLOWS_PATH=/app/flows \
    LANGFLOW_CONFIG_DIR=/tmp/langflow \
    LANGFLOW_DATABASE_URL=sqlite:////tmp/langflow/langflow.db \
    LANGFLOW_VARIABLES_TO_GET_FROM_ENVIRONMENT=GROQ_API_KEY \
    LANGFLOW_HOST=0.0.0.0 \
    LANGFLOW_PORT=7860 \
    DO_NOT_TRACK=true

EXPOSE 7860
