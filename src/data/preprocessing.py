
"""
Módulo de Ingesta, Limpieza y Formulación de Datos.
===================================================
Contiene las funciones encargadas de:

1. Localizar y cargar el dataset de estudiantes.
2. Construir la variable objetivo sintética multiclase:
   Graduado, En curso y Desertor.
3. Calcular diagnósticos estadísticos de asimetría y valores atípicos
   mediante el criterio del Rango Intercuartílico (IQR).
4. Generar particiones estratificadas y reproducibles para entrenamiento
   y prueba, manteniendo las etiquetas originales.

IMPORTANTE:
La variable objetivo se construye mediante reglas experimentales
sintéticas inspiradas en factores académicos y socioeconómicos.
No representa una implementación oficial de SPADIES 3.0.

Machine Learning II - 2026-II
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from scipy.stats import skew
from sklearn.model_selection import train_test_split

from src.utils.config import (
    ALL_FEATURE_COLS,
    DEFAULT_DATA_FILE,
    NUM_COLS,
    PROJECT_ROOT
)


# ============================================================
# 1. CARGA DE DATOS
# ============================================================

def load_data(
    filepath: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Localiza y carga el archivo CSV del dataset de estudiantes.

    Si no se proporciona una ruta específica, busca en las ubicaciones
    estándar del proyecto.

    Parameters
    ----------
    filepath : str o Path, opcional
        Ruta del archivo CSV.

    Returns
    -------
    pd.DataFrame
        Dataset cargado.

    Raises
    ------
    FileNotFoundError
        Si no se encuentra el archivo.
    """

    candidates = []

    if filepath is not None:
        candidates.append(Path(filepath).expanduser())

    candidates.extend([
        DEFAULT_DATA_FILE,
        PROJECT_ROOT / "Archivos" / "estudiantes_sinteticos_11500.csv",
        PROJECT_ROOT / "estudiantes_sinteticos_11500.csv",
        PROJECT_ROOT / "avances nicolas" / "estudiantes_sinteticos_11500.csv"
    ])

    # Eliminar rutas duplicadas conservando su orden.
    unique_candidates = list(dict.fromkeys(candidates))

    for path in unique_candidates:

        if path.is_file():

            df = pd.read_csv(path)

            if df.empty:
                raise ValueError(
                    f"El archivo de datos está vacío: {path}"
                )

            print(f"Dataset cargado correctamente desde: {path}")

            return df

    searched_paths = "\n".join(
        f"- {path}" for path in unique_candidates
    )

    raise FileNotFoundError(
        "No se encontró el archivo CSV del dataset.\n"
        f"Rutas examinadas:\n{searched_paths}"
    )


# ============================================================
# 2. CONSTRUCCIÓN DE LA VARIABLE OBJETIVO
# ============================================================

def create_spadies_target(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Construye la variable objetivo sintética de tres clases:

    - Graduado
    - En curso
    - Desertor

    La clasificación utiliza una regla experimental de riesgo académico
    y socioeconómico. No constituye una regla oficial de SPADIES 3.0.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.

    Returns
    -------
    pd.DataFrame
        Copia del dataset con la columna estado_final.

    Raises
    ------
    ValueError
        Si faltan columnas necesarias o existen valores nulos
        en las variables utilizadas para construir el objetivo.
    """

    df = df.copy()

    required_cols = [
        "nivel_formacion",
        "promedio_academico",
        "materias_reprobadas",
        "puntaje_icfes_percentil",
        "trabaja_mientras_estudia",
        "beneficiario_beca",
        "estrato_socioeconomico",
        "semestre_cursado"
    ]

    # --------------------------------------------------------
    # Validación de columnas
    # --------------------------------------------------------

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "No se puede construir estado_final. "
            f"Faltan las siguientes columnas: {missing_cols}"
        )

    # --------------------------------------------------------
    # Validación de valores nulos
    # --------------------------------------------------------

    null_cols = [
        col for col in required_cols
        if df[col].isnull().any()
    ]

    if null_cols:
        raise ValueError(
            "Existen valores nulos en las variables necesarias "
            f"para construir estado_final: {null_cols}"
        )

    # --------------------------------------------------------
    # Validación de variables numéricas
    # --------------------------------------------------------

    numeric_cols = [
        "promedio_academico",
        "materias_reprobadas",
        "puntaje_icfes_percentil",
        "trabaja_mientras_estudia",
        "beneficiario_beca",
        "estrato_socioeconomico",
        "semestre_cursado"
    ]

    for col in numeric_cols:

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if converted.isnull().any():
            raise ValueError(
                f"La columna '{col}' contiene valores no numéricos."
            )

        if not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ValueError(
                f"La columna '{col}' contiene valores infinitos."
            )

        df[col] = converted

    # --------------------------------------------------------
    # Duración nominal según nivel de formación
    # --------------------------------------------------------

    duracion_map = {
        "Tecnico profesional": 4,
        "Tecnologico": 6,
        "Universitario": 10
    }

    duracion_nominal = (
        df["nivel_formacion"]
        .map(duracion_map)
        .fillna(8)
    )

    # --------------------------------------------------------
    # Índice sintético de riesgo
    # --------------------------------------------------------

    riesgo = (
        (df["promedio_academico"] < 3.0).astype(int) * 2.5
        + (df["materias_reprobadas"] >= 2).astype(int) * 2.5
        + (df["puntaje_icfes_percentil"] < 40).astype(int) * 1.0
        + (df["trabaja_mientras_estudia"] == 1).astype(int) * 1.0
        + (df["beneficiario_beca"] == 0).astype(int) * 0.5
        + (df["estrato_socioeconomico"] <= 2).astype(int) * 0.5
    )

    # --------------------------------------------------------
    # Clasificación de desertores
    # --------------------------------------------------------

    is_desertor = riesgo >= 3.5

    # --------------------------------------------------------
    # Clasificación de graduados
    # --------------------------------------------------------

    is_graduado = (
        (~is_desertor)
        & (df["semestre_cursado"] >= duracion_nominal)
        & (df["promedio_academico"] >= 3.2)
    )

    # --------------------------------------------------------
    # Asignación de las tres clases
    # --------------------------------------------------------

    df["estado_final"] = np.select(
        [is_desertor, is_graduado],
        ["Desertor", "Graduado"],
        default="En curso"
    )

    return df


# ============================================================
# 3. DIAGNÓSTICO DE VALORES ATÍPICOS Y ASIMETRÍA
# ============================================================

def compute_outlier_diagnostics(
    df: pd.DataFrame,
    num_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas, asimetría y valores atípicos
    utilizando el criterio del Rango Intercuartílico (IQR).

    Límites:
        Q1 = Percentil 25
        Q3 = Percentil 75
        IQR = Q3 - Q1

        Límite inferior = Q1 - 1.5 * IQR
        Límite superior = Q3 + 1.5 * IQR

    Parameters
    ----------
    df : pd.DataFrame
        Dataset que será diagnosticado.

    num_cols : list, opcional
        Variables numéricas que se analizarán.
        Por defecto se utilizan NUM_COLS.

    Returns
    -------
    pd.DataFrame
        Tabla con estadísticas descriptivas y diagnóstico IQR.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "El parámetro df debe ser un pandas DataFrame."
        )

    if num_cols is None:
        num_cols = NUM_COLS

    diag_rows = []

    for col in num_cols:

        if col not in df.columns:
            continue

        data_col = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        # No calcular estadísticas si no existen datos válidos.
        if data_col.empty:
            diag_rows.append({
                "Variable": col,
                "Media": np.nan,
                "Mediana": np.nan,
                "IQR": np.nan,
                "Outliers Count": 0,
                "Outliers %": np.nan,
                "Skewness": np.nan
            })

            continue

        values = data_col.to_numpy(dtype=float)

        # Cuartiles e IQR.
        q25, q75 = np.percentile(
            values,
            [25, 75]
        )

        iqr = q75 - q25

        # Límites de Tukey.
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        # Identificación de valores atípicos.
        outlier_mask = (
            (values < lower_bound)
            | (values > upper_bound)
        )

        n_outliers = int(outlier_mask.sum())

        pct_outliers = (
            (n_outliers / len(values)) * 100
        )

        # Asimetría de Fisher-Pearson.
        # Para variables constantes o muestras pequeñas se evita
        # producir valores indefinidos.
        if len(values) >= 3 and np.std(values) > 0:

            skewness = float(
                skew(
                    values,
                    bias=False,
                    nan_policy="omit"
                )
            )

        else:
            skewness = 0.0

        diag_rows.append({
            "Variable": col,
            "Media": float(np.mean(values)),
            "Mediana": float(np.median(values)),
            "IQR": float(iqr),
            "Outliers Count": n_outliers,
            "Outliers %": float(pct_outliers),
            "Skewness": skewness
        })

    columns = [
        "Variable",
        "Media",
        "Mediana",
        "IQR",
        "Outliers Count",
        "Outliers %",
        "Skewness"
    ]

    return pd.DataFrame(
        diag_rows,
        columns=columns
    )


# ============================================================
# 4. PARTICIÓN ESTRATIFICADA TRAIN / TEST
# ============================================================

def get_train_test_split(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray
]:
    """
    Separa las variables predictoras y la variable objetivo mediante
    una partición estratificada y reproducible.

    La variable objetivo utilizada es estado_final, conservando
    las etiquetas textuales:

        Graduado
        En curso
        Desertor

    No requiere una columna target codificada numéricamente.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con las variables predictoras.

    feature_cols : list, opcional
        Columnas predictoras.
        Por defecto utiliza ALL_FEATURE_COLS.

    test_size : float
        Proporción destinada al conjunto de prueba.
        Por defecto 0.20.

    random_state : int
        Semilla de reproducibilidad.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, X_raw, y

    Raises
    ------
    ValueError
        Si faltan columnas, existen valores nulos en las variables
        necesarias o no se puede realizar una partición estratificada.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "El parámetro df debe ser un pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "No se puede dividir un dataset vacío."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size debe ser un valor entre 0 y 1."
        )

    if feature_cols is None:
        feature_cols = list(ALL_FEATURE_COLS)

    if not feature_cols:
        raise ValueError(
            "La lista de variables predictoras está vacía."
        )

    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError(
            "La lista feature_cols contiene columnas duplicadas."
        )

    # --------------------------------------------------------
    # Validar columnas predictoras
    # --------------------------------------------------------

    missing_features = [
        col for col in feature_cols
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Faltan variables predictoras en el dataset: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Construir variable objetivo si es necesario
    # --------------------------------------------------------

    if "estado_final" not in df.columns:

        df = create_spadies_target(df)

    # --------------------------------------------------------
    # Validar objetivo
    # --------------------------------------------------------

    if df["estado_final"].isnull().any():
        raise ValueError(
            "La variable estado_final contiene valores nulos."
        )

    valid_classes = {
        "Graduado",
        "En curso",
        "Desertor"
    }

    observed_classes = set(
        df["estado_final"].unique()
    )

    unexpected_classes = observed_classes - valid_classes

    if unexpected_classes:
        raise ValueError(
            "Se encontraron etiquetas no reconocidas en estado_final: "
            f"{unexpected_classes}"
        )

    # --------------------------------------------------------
    # Extraer variables predictoras y objetivo
    # --------------------------------------------------------

    X_raw = df[feature_cols].copy()

    y = df["estado_final"].astype(str).to_numpy()

    # --------------------------------------------------------
    # Validar valores nulos de las variables predictoras
    # --------------------------------------------------------

    null_features = X_raw.columns[
        X_raw.isnull().any()
    ].tolist()

    if null_features:
        raise ValueError(
            "Existen valores nulos en las variables predictoras: "
            f"{null_features}. Deben tratarse antes de entrenar el SVM."
        )

    # --------------------------------------------------------
    # Validar frecuencia mínima de las clases
    # --------------------------------------------------------

    class_counts = pd.Series(y).value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "Se necesitan al menos dos clases para realizar "
            "una clasificación multiclase."
        )

    if (class_counts < 2).any():
        raise ValueError(
            "Cada clase debe tener al menos dos observaciones "
            "para realizar una partición estratificada."
        )

    # --------------------------------------------------------
    # Partición estratificada
    # --------------------------------------------------------

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X_raw,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    except ValueError as exc:

        raise ValueError(
            "No fue posible realizar la partición estratificada. "
            "Verifica el número de observaciones por clase y "
            "el tamaño del conjunto de prueba."
        ) from exc

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_raw,
        y
    )
