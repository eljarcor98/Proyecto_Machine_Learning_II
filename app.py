
"""
Aplicación interactiva Streamlit.
==================================
Predicción multiclase del estado académico mediante SVM.

Modelos:
- SVM RBF + StandardScaler.
- SVM Poly-3 + PowerTransformer.

Los pipelines incluyen sus propios transformadores.
No se realiza un nuevo entrenamiento desde Streamlit.

IMPORTANTE:
Los modelos fueron entrenados con un dataset sintético.
Sus predicciones tienen carácter experimental y educativo.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.utils.config import (
    PROJECT_ROOT,
    DEFAULT_DATA_FILE,
    ALL_FEATURE_COLS,
    NUM_COLS,
    CAT_COLS,
    CLASSES
)


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Predicción de Deserción Estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELS_DIR = PROJECT_ROOT / "modelos"

MODEL_RBF_PATH = MODELS_DIR / "svm_pipeline_rbf.joblib"
MODEL_POLY_PATH = MODELS_DIR / "svm_pipeline_poly3.joblib"

DISCRETE_NUM_COLS = [
    "edad_ingreso",
    "materias_reprobadas",
    "semestre_cursado",
    "estrato_socioeconomico",
    "anio_registro"
]

BINARY_COLS = [
    "trabaja_mientras_estudia",
    "beneficiario_icetex",
    "beneficiario_beca"
]


# ============================================================
# 2. CARGA DE RECURSOS
# ============================================================

@st.cache_data
def load_reference_data():
    """Carga y valida el dataset de referencia para obtener rangos y categorías de entrada."""
    if not DEFAULT_DATA_FILE.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset: {DEFAULT_DATA_FILE}"
        )

    df = pd.read_csv(DEFAULT_DATA_FILE)

    required = set(ALL_FEATURE_COLS)
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas en el dataset: {sorted(missing)}"
        )

    return df


@st.cache_resource
def load_models():
    """Carga en memoria los pipelines serializados de SVM (RBF y Poly-3)."""
    missing_files = [
        str(path)
        for path in [MODEL_RBF_PATH, MODEL_POLY_PATH]
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "No se encontraron los modelos entrenados:\n"
            + "\n".join(missing_files)
        )

    model_rbf = joblib.load(MODEL_RBF_PATH)
    model_poly = joblib.load(MODEL_POLY_PATH)

    return model_rbf, model_poly


# ============================================================
# 3. ESTILOS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 750;
        color: #174A72;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f1f5f9;
        border: 1px solid #dbe4ee;
        text-align: center;
    }

    .result-label {
        font-size: 1.15rem;
        font-weight: 700;
        color: #174A72;
    }

    .result-value {
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .warning-text {
        color: #9a6700;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. ENCABEZADO
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Sistema de Predicción de Deserción Estudiantil'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Comparación de modelos SVM con diferentes configuraciones '
    'de preprocesamiento'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    Esta aplicación permite ingresar las características de un
    estudiante y obtener una predicción de su estado académico
    mediante dos modelos de Máquinas de Soporte Vectorial.
    """
)

st.divider()


# ============================================================
# 5. CARGAR DATASET Y MODELOS
# ============================================================

try:

    reference_df = load_reference_data()

    model_rbf, model_poly = load_models()

except Exception as error:

    st.error(
        "No fue posible cargar los recursos necesarios."
    )

    st.exception(error)

    st.stop()


# ============================================================
# 6. VALIDACIÓN DE COMPATIBILIDAD
# ============================================================

for model_name, model in [
    ("SVM RBF", model_rbf),
    ("SVM Poly-3", model_poly)
]:

    if hasattr(model, "feature_names_in_"):

        model_features = list(model.feature_names_in_)

        if model_features != list(ALL_FEATURE_COLS):

            st.error(
                f"Las columnas esperadas por {model_name} "
                "no coinciden con la configuración actual."
            )

            st.write("Columnas esperadas:", model_features)
            st.write("Columnas configuradas:", ALL_FEATURE_COLS)

            st.stop()


# ============================================================
# 7. PANEL LATERAL
# ============================================================

with st.sidebar:

    st.title("🎓 Panel del proyecto")

    st.markdown(
        """
        **Machine Learning II**

        Predicción multiclase del estado académico mediante SVM.
        """
    )

    st.divider()

    st.markdown("### Modelos disponibles")

    st.success("SVM RBF — StandardScaler")

    st.success("SVM Poly-3 — PowerTransformer")

    st.divider()

    st.markdown("### Clases de predicción")

    st.markdown("- 🟢 Graduado")
    st.markdown("- 🔵 En curso")
    st.markdown("- 🟠 Desertor")

    st.divider()

    st.caption(
        "Proyecto académico desarrollado con datos sintéticos "
        "inspirados en el contexto SPADIES 3.0."
    )


# ============================================================
# 8. FUNCIONES AUXILIARES PARA EL FORMULARIO
# ============================================================

def get_numeric_default(column):
    """Extrae la serie numérica limpia de una columna de referencia."""
    values = pd.to_numeric(
        reference_df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        raise ValueError(
            f"No existen valores válidos para {column}."
        )

    return values


def get_categorical_options(column):
    """Obtiene y ordena la lista de categorías únicas disponibles para una variable."""
    values = (
        reference_df[column]
        .dropna()
        .unique()
        .tolist()
    )

    values = sorted(values, key=lambda value: str(value))

    if not values:
        raise ValueError(
            f"No existen categorías disponibles para {column}."
        )

    return values


def numeric_input(column, label):
    """Renderiza un control interactivo de entrada numérica adaptado al rango y tipo de la variable."""
    values = get_numeric_default(column)

    minimum = float(values.min())
    maximum = float(values.max())
    default = float(values.median())

    if minimum == maximum:
        return float(minimum)

    if column in DISCRETE_NUM_COLS:

        return st.number_input(
            label,
            min_value=int(np.floor(minimum)),
            max_value=int(np.ceil(maximum)),
            value=int(round(default)),
            step=1,
            key=f"input_{column}"
        )

    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=float(np.clip(default, minimum, maximum)),
        step=0.1,
        format="%.2f",
        key=f"input_{column}"
    )


def categorical_input(column, label):
    """Renderiza un selector desplegable con las opciones válidas para una variable categórica."""
    options = get_categorical_options(column)

    return st.selectbox(
        label,
        options=options,
        format_func=lambda value: str(value),
        key=f"input_{column}"
    )


def binary_input(column, label):
    """Renderiza un selector binario con etiquetas amigables No (0) y Sí (1)."""
    return st.selectbox(
        label,
        options=[0, 1],
        format_func=lambda value: (
            "No (0)" if value == 0 else "Sí (1)"
        ),
        key=f"input_{column}"
    )


# ============================================================
# 9. FORMULARIO DE ENTRADA
# ============================================================

st.header("1. Características del estudiante")

st.write(
    "Completa los campos para realizar la predicción. "
    "Los valores iniciales corresponden a valores representativos "
    "del dataset experimental."
)

with st.form("student_form"):

    st.subheader("Información demográfica")

    col1, col2 = st.columns(2)

    with col1:

        sexo = categorical_input(
            "sexo",
            "Sexo"
        )

        estado_civil = categorical_input(
            "estado_civil",
            "Estado civil"
        )

        zona_residencia = categorical_input(
            "zona_residencia",
            "Zona de residencia"
        )

        departamento = categorical_input(
            "departamento",
            "Departamento"
        )

        edad_ingreso = numeric_input(
            "edad_ingreso",
            "Edad de ingreso"
        )

    with col2:

        estrato_socioeconomico = numeric_input(
            "estrato_socioeconomico",
            "Estrato socioeconómico"
        )

        nivel_educativo_padre = categorical_input(
            "nivel_educativo_padre",
            "Nivel educativo del padre"
        )

        nivel_educativo_madre = categorical_input(
            "nivel_educativo_madre",
            "Nivel educativo de la madre"
        )

        sector_ies = categorical_input(
            "sector_ies",
            "Sector de la institución"
        )

        trabaja_mientras_estudia = binary_input(
            "trabaja_mientras_estudia",
            "¿Trabaja mientras estudia?"
        )

    st.divider()

    st.subheader("Información académica")

    col3, col4 = st.columns(2)

    with col3:

        nivel_formacion = categorical_input(
            "nivel_formacion",
            "Nivel de formación"
        )

        metodologia = categorical_input(
            "metodologia",
            "Metodología"
        )

        area_conocimiento = categorical_input(
            "area_conocimiento",
            "Área de conocimiento"
        )

        semestre_cursado = numeric_input(
            "semestre_cursado",
            "Semestre cursado"
        )

        promedio_academico = numeric_input(
            "promedio_academico",
            "Promedio académico"
        )

    with col4:

        materias_reprobadas = numeric_input(
            "materias_reprobadas",
            "Materias reprobadas"
        )

        puntaje_icfes_percentil = numeric_input(
            "puntaje_icfes_percentil",
            "Puntaje ICFES percentil"
        )

        distancia_hogar_ies_km = numeric_input(
            "distancia_hogar_ies_km",
            "Distancia hogar - institución (km)"
        )

        anio_registro = numeric_input(
            "anio_registro",
            "Año de registro"
        )

    st.divider()

    st.subheader("Información socioeconómica")

    col5, col6 = st.columns(2)

    with col5:

        beneficiario_icetex = binary_input(
            "beneficiario_icetex",
            "¿Es beneficiario de ICETEX?"
        )

    with col6:

        beneficiario_beca = binary_input(
            "beneficiario_beca",
            "¿Es beneficiario de una beca?"
        )

    submitted = st.form_submit_button(
        "🔍 Realizar predicción",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# 10. CONSTRUIR DATAFRAME DE ENTRADA
# ============================================================

if submitted:

    student_data = {

        "sexo": sexo,
        "estado_civil": estado_civil,
        "zona_residencia": zona_residencia,
        "departamento": departamento,
        "nivel_educativo_padre": nivel_educativo_padre,
        "nivel_educativo_madre": nivel_educativo_madre,
        "sector_ies": sector_ies,
        "nivel_formacion": nivel_formacion,
        "metodologia": metodologia,
        "area_conocimiento": area_conocimiento,

        "edad_ingreso": edad_ingreso,
        "promedio_academico": promedio_academico,
        "materias_reprobadas": materias_reprobadas,
        "distancia_hogar_ies_km": distancia_hogar_ies_km,
        "puntaje_icfes_percentil": puntaje_icfes_percentil,
        "semestre_cursado": semestre_cursado,
        "estrato_socioeconomico": estrato_socioeconomico,
        "anio_registro": anio_registro,
        "trabaja_mientras_estudia": trabaja_mientras_estudia,
        "beneficiario_icetex": beneficiario_icetex,
        "beneficiario_beca": beneficiario_beca
    }

    input_df = pd.DataFrame(
        [student_data],
        columns=ALL_FEATURE_COLS
    )

    st.session_state["student_input"] = input_df

    st.session_state["prediction_ready"] = True


# ============================================================
# 11. GENERACIÓN DE PREDICCIONES
# ============================================================

if st.session_state.get("prediction_ready", False):

    input_df = st.session_state["student_input"]

    st.divider()

    st.header("2. Resultados de la predicción")

    try:

        with st.spinner(
            "Evaluando las características con ambos modelos SVM..."
        ):

            prediction_rbf = model_rbf.predict(input_df)[0]

            prediction_poly = model_poly.predict(input_df)[0]

        if prediction_rbf not in CLASSES:
            raise ValueError(
                f"Clase no reconocida por RBF: {prediction_rbf}"
            )

        if prediction_poly not in CLASSES:
            raise ValueError(
                f"Clase no reconocida por Poly-3: {prediction_poly}"
            )

        # ----------------------------------------------------
        # Mostrar resultados individuales
        # ----------------------------------------------------

        col_rbf, col_poly = st.columns(2)

        with col_rbf:

            st.markdown(
                """
                <div class="result-card">
                    <div class="result-label">SVM RBF</div>
                    <div class="result-value">"""
                + str(prediction_rbf)
                + """</div>
                <p>StandardScaler + OneHotEncoder</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_poly:

            st.markdown(
                """
                <div class="result-card">
                    <div class="result-label">SVM Poly-3</div>
                    <div class="result-value">"""
                + str(prediction_poly)
                + """</div>
                <p>PowerTransformer + OneHotEncoder</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        # ----------------------------------------------------
        # Comparación de resultados
        # ----------------------------------------------------

        st.subheader("Comparación de los modelos")

        if prediction_rbf == prediction_poly:

            st.success(
                f"Ambos modelos coinciden en la predicción: "
                f"**{prediction_rbf}**."
            )

        else:

            st.warning(
                "Los modelos presentan predicciones diferentes. "
                "Esto refleja que sus funciones de decisión y "
                "transformaciones no son idénticas."
            )

        comparison_df = pd.DataFrame({

            "Modelo": [
                "SVM RBF",
                "SVM Poly-3"
            ],

            "Configuración": [
                "StandardScaler",
                "PowerTransformer"
            ],

            "Predicción": [
                prediction_rbf,
                prediction_poly
            ]

        })

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Advertencia metodológica
        # ----------------------------------------------------

        st.divider()

        st.info(
            """
            **Interpretación metodológica**

            Las predicciones corresponden a modelos entrenados con
            un conjunto de datos sintético inspirado en el contexto
            colombiano de SPADIES 3.0.

            La variable objetivo fue construida mediante reglas
            experimentales basadas en las características académicas
            y socioeconómicas.

            Por tanto, los resultados permiten ilustrar el
            funcionamiento de los clasificadores, pero no deben
            interpretarse como diagnósticos reales de estudiantes
            ni utilizarse para tomar decisiones institucionales.
            """
        )

        # ----------------------------------------------------
        # Mostrar datos procesados de entrada
        # ----------------------------------------------------

        with st.expander("Consultar características ingresadas"):

            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:

        st.error(
            "Se produjo un error durante la predicción."
        )

        st.exception(error)


# ============================================================
# 12. PIE DE PÁGINA
# ============================================================

st.divider()

st.caption(
    "Machine Learning II | 2026-II | Ciencia de Datos | "
    "Universidad Externado de Colombia"
)

st.caption(
    "Modelos experimentales SVM multiclase — RBF y Poly-3."
)
