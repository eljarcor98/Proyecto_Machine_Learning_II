"""
Módulo de Ingesta, Limpieza y Formulación de Datos SPADIES 3.0.
==============================================================
Contiene las funciones encargadas de:
1. Localizar y cargar el dataset de estudiantes desde múltiples rutas relativas o absolutas.
2. Formular la variable objetivo multiclase ('estado_final') conforme a los lineamientos del
   Sistema de Prevención de la Deserción en la Educación Superior (SPADIES 3.0) de Colombia.
3. Realizar un diagnóstico estadístico riguroso de asimetría (skewness) y valores atípicos
   mediante el criterio del Rango Intercuartílico (IQR).
4. Generar particiones de entrenamiento y prueba estratificadas y reproducibles, evitando la fuga
   de información (data leakage).

Autor: Arnold Santiago Torres
Machine Learning II - 2026-II
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import train_test_split

from src.utils.config import (
    ALL_FEATURE_COLS,
    CAT_COLS,
    CLASS_MAP,
    CLASSES,
    DEFAULT_DATA_FILE,
    NUM_COLS,
    PROJECT_ROOT
)


def load_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Localiza y carga en memoria el archivo CSV con los datos de los estudiantes.
    Implementa una lista de rutas de contingencia para garantizar portabilidad
    tanto si se ejecuta desde la raíz del proyecto como desde un subdirectorio o notebook.

    Parámetros:
    -----------
    filepath : str o Path, opcional
        Ruta específica hacia el archivo CSV. Si es None, busca en las ubicaciones estándar.

    Retorna:
    --------
    pd.DataFrame:
        DataFrame con los datos crudos cargados.

    Lanza:
    ------
    FileNotFoundError:
        Si el archivo no se localiza en ninguna de las rutas de contingencia.
    """
    candidates = []
    if filepath is not None:
        candidates.append(Path(filepath))

    # Lista ordenada de rutas candidatas prioritarias
    candidates.extend([
        DEFAULT_DATA_FILE,
        PROJECT_ROOT / "Archivos" / "estudiantes_sinteticos_11500.csv",
        PROJECT_ROOT / "estudiantes_sinteticos_11500.csv",
        PROJECT_ROOT / "avances nicolas" / "estudiantes_sinteticos_11500.csv"
    ])

    # Explorar rutas hasta encontrar una válida en el disco
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)

    # Si ninguna ruta existe, levantar excepción descriptiva
    raise FileNotFoundError(
        f"No se localizó el archivo de datos en ninguna de las rutas esperadas: {[str(p) for p in candidates]}"
    )


def create_spadies_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la formulación multiclase estandarizada según la metodología SPADIES 3.0
    del Ministerio de Educación Nacional de Colombia.

    Construcción del Índice de Riesgo Ponderado (R_i en [0.0, 8.0]):
    -----------------------------------------------------------------
    - Rendimiento Académico Crítico: (promedio_academico < 3.0) -> +2.5 puntos
    - Repitencia Múltiple: (materias_reprobadas >= 2)          -> +2.5 puntos
    - Brecha de Ingreso: (puntaje_icfes_percentil < 40)         -> +1.0 puntos
    - Sobrecarga Laboral: (trabaja_mientras_estudia == 1)      -> +1.0 puntos
    - Vulnerabilidad de Financiación: (beneficiario_beca == 0) -> +0.5 puntos
    - Vulnerabilidad Socioeconómica: (estrato <= 2)             -> +0.5 puntos

    Criterios de Asignación de Clase (estado_final):
    ------------------------------------------------
    1. 'Desertor': Estudiantes con índice de riesgo acumulado R_i >= 3.5 puntos.
    2. 'Graduado': Estudiantes sin condición de deserción que completaron el número
                   nominal de semestres (duración según nivel de formación) y mantuvieron
                   un promedio acumulado >= 3.2.
    3. 'En curso': Estudiantes activos persistentes que continúan cursando su programa.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con las columnas académicas, demográficas y socioeconómicas del estudiante.

    Retorna:
    --------
    pd.DataFrame:
        Copia del DataFrame original con las nuevas columnas:
        - 'estado_final': Etiqueta textual ('Graduado', 'En curso', 'Desertor').
        - 'target': Entero mapeado según CLASS_MAP (0, 1, 2).
    """
    df = df.copy()

    # Mapeo de duración nominal en semestres según el nivel académico
    duracion_map = {'Tecnico profesional': 4, 'Tecnologico': 6, 'Universitario': 10}
    duracion_nominal = df['nivel_formacion'].map(duracion_map).fillna(8)

    # Cálculo vectorial del Índice de Riesgo Ponderado
    riesgo = (
        (df['promedio_academico'] < 3.0).astype(int) * 2.5 +
        (df['materias_reprobadas'] >= 2).astype(int) * 2.5 +
        (df['puntaje_icfes_percentil'] < 40).astype(int) * 1.0 +
        (df['trabaja_mientras_estudia'] == 1).astype(int) * 1.0 +
        (df['beneficiario_beca'] == 0).astype(int) * 0.5 +
        (df['estrato_socioeconomico'] <= 2).astype(int) * 0.5
    )

    # Definición de máscaras booleanas para cada estado
    is_desertor = (riesgo >= 3.5)
    is_graduado = (~is_desertor) & (df['semestre_cursado'] >= duracion_nominal) & (df['promedio_academico'] >= 3.2)

    # Asignación jerárquica de la clase
    df['estado_final'] = np.where(is_desertor, 'Desertor', np.where(is_graduado, 'Graduado', 'En curso'))

    # Mapeo a variable target numérica para algoritmos de Scikit-Learn
    df['target'] = df['estado_final'].map(CLASS_MAP)

    return df


def compute_outlier_diagnostics(
    df: pd.DataFrame,
    num_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calcula el diagnóstico cuantitativo de distribución, asimetría estadística y
    valores atípicos según la regla de Tukey del Rango Intercuartílico (IQR).

    Límites del IQR:
    ----------------
    - Q1: Percentil 25
    - Q3: Percentil 75
    - IQR = Q3 - Q1
    - Límite Inferior = Q1 - 1.5 * IQR
    - Límite Superior = Q3 + 1.5 * IQR

    Parámetros:
    -----------
    df : pd.DataFrame
        Conjunto de datos a diagnosticar.
    num_cols : list of str, opcional
        Lista de variables numéricas a evaluar. Por defecto toma NUM_COLS.

    Retorna:
    --------
    pd.DataFrame:
        Tabla resumen con Media, Mediana, IQR, Conteo de Outliers, Porcentaje y Coeficiente de Asimetría (Skewness).
    """
    if num_cols is None:
        num_cols = NUM_COLS

    diag_rows = []

    # Iterar sobre cada variable continua para calcular métricas de dispersión
    for col in num_cols:
        if col not in df.columns:
            continue

        data_col = df[col].dropna()

        # Cálculo de cuartiles y rango intercuartílico
        q25, q75 = np.percentile(data_col, [25, 75])
        iqr = q75 - q25

        # Fronteras de detección de valores atípicos
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        # Conteo y porcentaje de observaciones que caen por fuera de las fronteras
        n_outliers = int(np.sum((data_col < lower_bound) | (data_col > upper_bound)))
        pct_outliers = (n_outliers / len(data_col)) * 100

        # Coeficiente de asimetría de Fisher-Pearson
        sk = float(skew(data_col))

        diag_rows.append({
            'Variable': col,
            'Media': float(data_col.mean()),
            'Mediana': float(data_col.median()),
            'IQR': float(iqr),
            'Outliers Count': n_outliers,
            'Outliers %': float(pct_outliers),
            'Skewness': sk
        })

    return pd.DataFrame(diag_rows)


def get_train_test_split(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Separa las matrices de características predictoras y el vector objetivo,
    realizando una partición estratificada para preservar la proporción relativa
    de las 3 clases SPADIES en ambos subconjuntos.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame preprocesado. Si no contiene 'estado_final', se calcula automáticamente.
    feature_cols : list of str, opcional
        Lista ordenada de columnas a incluir en la matriz de diseño X. Por defecto ALL_FEATURE_COLS.
    test_size : float, por defecto 0.2
        Proporción del conjunto de datos destinada a la partición de prueba (ej. 20%).
    random_state : int, por defecto 42
        Semilla pseudoaleatoria para reproducibilidad matemática estricta.

    Retorna:
    --------
    tuple:
        (X_train, X_test, y_train, y_test, X_raw, y)
    """
    # Asegurar existencia de las variables objetivo
    if 'estado_final' not in df.columns or 'target' not in df.columns:
        df = create_spadies_target(df)

    if feature_cols is None:
        feature_cols = ALL_FEATURE_COLS

    # Extraer la matriz de diseño X y el vector unidimensional y
    X_raw = df[feature_cols].copy()
    y = df['target'].values

    # Realizar partición estratificada con Scikit-Learn
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, X_raw, y
