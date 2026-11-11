"""
Módulo de Configuración Central del Proyecto.
==============================================
Define la arquitectura de rutas canónicas basadas en pathlib.Path, el esquema
de variables predictoras (numéricas y categóricas), el mapeo de la variable
objetivo según la norma SPADIES 3.0 de Colombia, y las paletas estéticas
utilizadas de manera transversal en los gráficos.

Autor: Arnold Santiago Torres
Machine Learning II - 2026-II
"""

from pathlib import Path

# ==============================================================================
# 1. RESOLUCIÓN DINÁMICA DE RUTAS DEL SISTEMA DE ARCHIVOS
# ==============================================================================
# UTILS_DIR: Directorio actual (src/utils)
UTILS_DIR = Path(__file__).resolve().parent

# SRC_DIR: Directorio raíz del código fuente (src/)
SRC_DIR = UTILS_DIR.parent

# PROJECT_ROOT: Raíz del repositorio del proyecto
PROJECT_ROOT = SRC_DIR.parent

# Rutas a directorios de datos
DATA_DIR = PROJECT_ROOT / "datos"
DATA_RAW_DIR = DATA_DIR / "brutos"
DATA_PROCESSED_DIR = DATA_DIR / "procesados"

# Rutas a directorios de salidas y evidencias experimentales
RESULTS_DIR = PROJECT_ROOT / "resultados"
FIGURES_DIR = RESULTS_DIR / "figuras"
METRICS_DIR = RESULTS_DIR / "metricas"

# Ruta para artefactos de modelos serializados (.joblib)
MODELS_DIR = PROJECT_ROOT / "modelos"

# Rutas a archivos por defecto del flujo principal
DEFAULT_DATA_FILE = DATA_RAW_DIR / "estudiantes_sinteticos_11500.csv"
DEFAULT_MODEL_FILE = MODELS_DIR / "svm_pipeline_reproducible.joblib"
DEFAULT_METRICS_FILE = METRICS_DIR / "resultados_svm_normalizacion.csv"

# ==============================================================================
# 2. DEFINICIÓN DEL ESQUEMA DE CARACTERÍSTICAS (FEATURES)
# ==============================================================================
# Variables Categóricas (requieren codificación One-Hot o categórica)
CAT_COLS = [
    'sexo',                    # Género biológico del estudiante (Hombre/Mujer)
    'estado_civil',            # Estado conyugal (Soltero, Unión libre, Casado, etc.)
    'zona_residencia',         # Sector habitacional de procedencia (Urbana/Rural)
    'departamento',            # Departamento geográfico de Colombia
    'nivel_educativo_padre',   # Nivel máximo alcanzado por el padre
    'nivel_educativo_madre',   # Nivel máximo alcanzado por la madre
    'sector_ies',              # Naturaleza de la institución (Oficial/Pública vs Privada)
    'nivel_formacion',         # Tipo de programa (Tecnico profesional, Tecnologico, Universitario)
    'metodologia',             # Modalidad pedagógica (Presencial, Virtual, Distancia tradicional)
    'area_conocimiento'        # Campo de estudio (Ingeniería, Ciencias Sociales, Salud, etc.)
]

# Variables Numéricas Continuas y Discretas (sujetas a evaluación de normalizadores)
NUM_COLS = [
    'edad_ingreso',            # Edad biológica al momento de ingreso a la educación superior
    'promedio_academico',      # Calificación ponderada acumulada (escala estándar 0.0 - 5.0)
    'materias_reprobadas',     # Conteo acumulado de asignaturas no aprobadas
    'distancia_hogar_ies_km',  # Distancia física en kilómetros del hogar al campus (alta asimetría)
    'puntaje_icfes_percentil', # Rendimiento estandarizado en examen de Estado Saber 11 (percentil 0 - 100)
    'semestre_cursado',        # Número de periodos lectivos completados por el estudiante
    'estrato_socioeconomico',   # Nivel socioeconómico estratificado (1 al 6)
    'anio_registro',           # Periodo cronológico de matrícula institucional
    'trabaja_mientras_estudia',# Indicador binario (1: Trabaja activamente, 0: No trabaja)
    'beneficiario_icetex',     # Indicador binario de apoyo crediticio estatal (1: Sí, 0: No)
    'beneficiario_beca'        # Indicador binario de subsidio/beca de matrícula (1: Sí, 0: No)
]

# Unión ordenada de todas las características para la matriz de diseño X
ALL_FEATURE_COLS = NUM_COLS + CAT_COLS

# ==============================================================================
# 3. FORMULACIÓN MULTICLASE SPADIES 3.0
# ==============================================================================
# Clases académicas oficiales según el Ministerio de Educación Nacional de Colombia
CLASSES = ['Graduado', 'En curso', 'Desertor']

# Mapeos bidireccionales de etiquetas cualitativas a índices enteros
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}
INVERSE_CLASS_MAP = {i: c for i, c in enumerate(CLASSES)}

# ==============================================================================
# 4. PALETAS ESTÉTICAS PARA VISUALIZACIONES CIENTÍFICAS
# ==============================================================================
# Paleta cromática semántica para diagramas de barras, tortas y boxplots
CLASS_PALETTE = {
    'Graduado': '#2ca02c',     # Verde: culminación exitosa
    'En curso': '#2b5c8f',     # Azul: persistencia activa
    'Desertor': '#d95f02'      # Naranja/Rojo: deserción / desvinculación
}

# Paleta cromática para dispersión en proyecciones PCA 2D
SCATTER_PALETTE = ['#1b9e77', '#d95f02', '#7570b3']
