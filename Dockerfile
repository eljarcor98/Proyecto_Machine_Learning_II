# ==============================================================================
# Dockerfile para la Aplicación Streamlit de Predicción de Deserción con SVM
# Optimizado para arquitecturas x86_64 y ARM64 (Raspberry Pi 4 / 5)
# ==============================================================================

FROM python:3.11-slim

# Variables de entorno para Python y Streamlit
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Instalar dependencias mínimas del sistema requeridas para librerías científicas en ARM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar dependencias e instalarlas aprovechando el caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación y los artefactos de modelos y datos
COPY app.py .
COPY src/ ./src/
COPY modelos/ ./modelos/
COPY datos/ ./datos/
COPY resultados/ ./resultados/

# Exponer el puerto interno de Streamlit
EXPOSE 8501

# Verificación de salud del contenedor
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de ejecución
ENTRYPOINT ["streamlit", "run", "app.py"]
