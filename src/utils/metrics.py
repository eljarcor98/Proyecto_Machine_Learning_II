
"""
Módulo de Métricas y Evaluación Estadística.
=============================================
Provee funciones para:

1. Cálculo de métricas multiclase: Accuracy, F1 ponderado,
   F1 macro y Recall de la clase Desertor.
2. Construcción de tablas comparativas de experimentos Monte Carlo.
3. Comparaciones pareadas frente al baseline.
4. Corrección de Holm para comparaciones múltiples.
5. Auditoría descriptiva del Recall de Desertor por subgrupos.
6. Exportación de métricas a archivos CSV.

Nota metodológica:
------------------
Las comparaciones estadísticas utilizan resultados pareados porque
todas las configuraciones se evalúan sobre las mismas particiones
en cada trial.

Los p-valores deben interpretarse con cautela debido al posible
solapamiento entre particiones Monte Carlo. Por tanto, las pruebas
estadísticas se consideran exploratorias.

Las etiquetas objetivo se manejan como cadenas:
    - Graduado
    - En curso
    - Desertor

Autor: Arnold Santiago Torres
Machine Learning II - 2026-II
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from scipy.stats import ttest_rel

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
)

from src.utils.config import (
    CLASSES,
    DEFAULT_METRICS_FILE,
    METRICS_DIR
)


# ============================================================
# 1. MÉTRICAS DE CLASIFICACIÓN MULTICLASE
# ============================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    desertor_label: str = "Desertor"
) -> Dict[str, Any]:
    """
    Calcula métricas de clasificación multiclase.

    Las etiquetas se manejan como cadenas y se utiliza un orden
    fijo para calcular la matriz de confusión y el reporte.

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas reales.

    y_pred : np.ndarray
        Etiquetas predichas.

    target_names : list of str, optional
        Orden de las clases. Por defecto utiliza CLASSES.

    desertor_label : str
        Nombre textual de la clase Desertor.

    Returns
    -------
    dict
        accuracy:
            Exactitud global.

        f1_weighted:
            F1 ponderado por la frecuencia de cada clase.

        f1_macro:
            Promedio no ponderado del F1 de las clases.

        recall_desertor:
            Recall específico de la clase Desertor.

        confusion_matrix:
            Matriz de confusión con orden fijo de clases.

        classification_report:
            Reporte detallado de clasificación.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # --------------------------------------------------------
    # Validaciones
    # --------------------------------------------------------

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            "y_true e y_pred deben ser vectores unidimensionales."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true e y_pred deben tener la misma longitud."
        )

    if len(y_true) == 0:
        raise ValueError(
            "No se pueden calcular métricas sin observaciones."
        )

    if target_names is None:
        target_names = list(CLASSES)
    else:
        target_names = list(target_names)

    if len(target_names) == 0:
        raise ValueError(
            "target_names no puede estar vacío."
        )

    if len(set(target_names)) != len(target_names):
        raise ValueError(
            "target_names contiene clases duplicadas."
        )

    if desertor_label not in target_names:
        raise ValueError(
            f"La etiqueta '{desertor_label}' no pertenece "
            f"a las clases definidas: {target_names}."
        )

    observed_labels = set(np.unique(y_true))
    predicted_labels = set(np.unique(y_pred))

    allowed_labels = set(target_names)

    unknown_true = observed_labels - allowed_labels
    unknown_pred = predicted_labels - allowed_labels

    if unknown_true:
        raise ValueError(
            f"Se encontraron etiquetas reales desconocidas: {unknown_true}"
        )

    if unknown_pred:
        raise ValueError(
            f"Se encontraron etiquetas predichas desconocidas: {unknown_pred}"
        )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    acc = accuracy_score(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # F1 Weighted
    # --------------------------------------------------------

    f1_wei = f1_score(
        y_true,
        y_pred,
        labels=target_names,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # F1 Macro
    # --------------------------------------------------------

    f1_mac = f1_score(
        y_true,
        y_pred,
        labels=target_names,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Recall específico de Desertor
    # --------------------------------------------------------

    rec_des = recall_score(
        y_true,
        y_pred,
        labels=[desertor_label],
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Matriz de confusión
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=target_names
    )

    # --------------------------------------------------------
    # Reporte de clasificación
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=target_names,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1_wei),
        "f1_macro": float(f1_mac),
        "recall_desertor": float(rec_des),
        "confusion_matrix": cm,
        "classification_report": report
    }


# ============================================================
# 2. CORRECCIÓN DE HOLM-BONFERRONI
# ============================================================

def _holm_adjustment(
    p_values: List[float]
) -> np.ndarray:
    """
    Aplica el procedimiento de Holm-Bonferroni.

    Controla el error familiar al realizar múltiples comparaciones.

    Los valores no finitos se conservan como NaN.

    Parameters
    ----------
    p_values : list
        P-valores sin ajustar.

    Returns
    -------
    np.ndarray
        P-valores ajustados mediante Holm.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
        dtype=float
    )

    valid_idx = np.where(
        np.isfinite(p_values)
    )[0]

    if len(valid_idx) == 0:
        return adjusted

    valid_p = p_values[valid_idx]

    if np.any((valid_p < 0) | (valid_p > 1)):
        raise ValueError(
            "Los p-valores deben estar entre 0 y 1."
        )

    # Ordenar p-valores de menor a mayor.
    order = np.argsort(valid_p)

    sorted_p = valid_p[order]

    m = len(sorted_p)

    # Ajuste escalonado de Holm.
    sorted_adjusted = np.maximum.accumulate(
        (m - np.arange(m)) * sorted_p
    )

    sorted_adjusted = np.clip(
        sorted_adjusted,
        0,
        1
    )

    # Recuperar el orden original.
    adjusted_valid = np.empty(
        m,
        dtype=float
    )

    adjusted_valid[order] = sorted_adjusted

    adjusted[valid_idx] = adjusted_valid

    return adjusted


# ============================================================
# 3. TABLA COMPARATIVA DE EXPERIMENTOS MONTE CARLO
# ============================================================

def build_summary_comparison_table(
    results_weighted: Dict[str, List[float]],
    results_macro: Dict[str, List[float]],
    results_desertor: Dict[str, List[float]],
    baseline_name: str = "No Normalization (Baseline)"
) -> pd.DataFrame:
    """
    Construye una tabla comparativa de los experimentos Monte Carlo.

    Las comparaciones estadísticas se realizan mediante t-test
    pareado frente al baseline, dado que todas las configuraciones
    comparten las mismas particiones dentro de cada trial.

    La variable principal para ordenar los resultados es F1 macro.

    Se reportan:

    - Media y desviación estándar de F1 weighted.
    - Media y desviación estándar de F1 macro.
    - Media y desviación estándar de F1 Desertor.
    - Diferencia de F1 weighted frente al baseline.
    - Estadístico t.
    - P-valor pareado.
    - P-valor ajustado mediante Holm.

    IMPORTANTE:
    Los p-valores son exploratorios porque las particiones Monte Carlo
    pueden compartir observaciones entre distintos trials.

    Parameters
    ----------
    results_weighted : dict
        Resultados de F1 weighted por configuración.

    results_macro : dict
        Resultados de F1 macro por configuración.

    results_desertor : dict
        Resultados de F1 de la clase Desertor por configuración.

    baseline_name : str
        Nombre de la configuración de referencia.

    Returns
    -------
    pd.DataFrame
        Tabla comparativa ordenada por F1 macro promedio.
    """

    # --------------------------------------------------------
    # Validar diccionarios
    # --------------------------------------------------------

    if not isinstance(results_weighted, dict):
        raise TypeError(
            "results_weighted debe ser un diccionario."
        )

    if not isinstance(results_macro, dict):
        raise TypeError(
            "results_macro debe ser un diccionario."
        )

    if not isinstance(results_desertor, dict):
        raise TypeError(
            "results_desertor debe ser un diccionario."
        )

    if not results_weighted:
        raise ValueError(
            "No existen resultados experimentales."
        )

    methods_weighted = set(results_weighted.keys())
    methods_macro = set(results_macro.keys())
    methods_desertor = set(results_desertor.keys())

    if not (
        methods_weighted
        == methods_macro
        == methods_desertor
    ):
        raise ValueError(
            "Los tres diccionarios deben contener exactamente "
            "las mismas configuraciones."
        )

    if baseline_name not in results_weighted:
        raise ValueError(
            f"No se encontró el baseline '{baseline_name}'."
        )

    # --------------------------------------------------------
    # Convertir resultados en arrays numéricos
    # --------------------------------------------------------

    arrays_weighted = {}
    arrays_macro = {}
    arrays_desertor = {}

    for method in results_weighted:

        arrays_weighted[method] = np.asarray(
            results_weighted[method],
            dtype=float
        )

        arrays_macro[method] = np.asarray(
            results_macro[method],
            dtype=float
        )

        arrays_desertor[method] = np.asarray(
            results_desertor[method],
            dtype=float
        )

        if not (
            arrays_weighted[method].ndim == 1
            and arrays_macro[method].ndim == 1
            and arrays_desertor[method].ndim == 1
        ):
            raise ValueError(
                f"Los resultados de '{method}' deben ser "
                "vectores unidimensionales."
            )

        if not (
            len(arrays_weighted[method])
            == len(arrays_macro[method])
            == len(arrays_desertor[method])
        ):
            raise ValueError(
                f"Los resultados de '{method}' tienen "
                "diferente número de trials."
            )

        if len(arrays_weighted[method]) == 0:
            raise ValueError(
                f"La configuración '{method}' no contiene resultados."
            )

    # --------------------------------------------------------
    # Verificar alineación de los trials
    # --------------------------------------------------------

    trial_counts = {
        len(values)
        for values in arrays_weighted.values()
    }

    if len(trial_counts) != 1:
        raise ValueError(
            "Todas las configuraciones deben tener el mismo "
            "número de trials para realizar comparaciones pareadas."
        )

    baseline_scores = arrays_weighted[baseline_name]

    if not np.isfinite(baseline_scores).any():
        raise ValueError(
            "El baseline no contiene resultados F1 weighted válidos."
        )

    baseline_mean = float(
        np.nanmean(baseline_scores)
    )

    # --------------------------------------------------------
    # Construcción de filas
    # --------------------------------------------------------

    summary_rows = []

    raw_p_values = []
    comparison_indices = []

    for method_name in arrays_weighted:

        scores_weighted = arrays_weighted[method_name]
        scores_macro = arrays_macro[method_name]
        scores_desertor = arrays_desertor[method_name]

        # ----------------------------------------------------
        # F1 Weighted
        # ----------------------------------------------------

        mean_weighted = (
            float(np.nanmean(scores_weighted))
            if np.isfinite(scores_weighted).any()
            else np.nan
        )

        std_weighted = (
            float(np.nanstd(scores_weighted, ddof=1))
            if np.isfinite(scores_weighted).sum() >= 2
            else np.nan
        )

        # ----------------------------------------------------
        # F1 Macro
        # ----------------------------------------------------

        mean_macro = (
            float(np.nanmean(scores_macro))
            if np.isfinite(scores_macro).any()
            else np.nan
        )

        std_macro = (
            float(np.nanstd(scores_macro, ddof=1))
            if np.isfinite(scores_macro).sum() >= 2
            else np.nan
        )

        # ----------------------------------------------------
        # F1 Desertor
        # ----------------------------------------------------

        mean_desertor = (
            float(np.nanmean(scores_desertor))
            if np.isfinite(scores_desertor).any()
            else np.nan
        )

        std_desertor = (
            float(np.nanstd(scores_desertor, ddof=1))
            if np.isfinite(scores_desertor).sum() >= 2
            else np.nan
        )

        # ----------------------------------------------------
        # Diferencia frente al baseline
        # ----------------------------------------------------

        delta = (
            mean_weighted - baseline_mean
            if np.isfinite(mean_weighted)
            else np.nan
        )

        # ----------------------------------------------------
        # Comparación pareada
        # ----------------------------------------------------

        if method_name == baseline_name:

            t_stat = 0.0
            p_value = 1.0

        else:

            valid_mask = (
                np.isfinite(scores_weighted)
                & np.isfinite(baseline_scores)
            )

            method_valid = scores_weighted[valid_mask]
            baseline_valid = baseline_scores[valid_mask]

            if len(method_valid) < 2:

                t_stat = np.nan
                p_value = np.nan

            elif np.allclose(
                method_valid,
                baseline_valid
            ):

                t_stat = 0.0
                p_value = 1.0

            else:

                test_result = ttest_rel(
                    method_valid,
                    baseline_valid,
                    nan_policy="omit"
                )

                t_stat = float(
                    test_result.statistic
                )

                p_value = float(
                    test_result.pvalue
                )

                if not np.isfinite(t_stat):
                    t_stat = np.nan

                if not np.isfinite(p_value):
                    p_value = np.nan

            comparison_indices.append(
                len(summary_rows)
            )

            raw_p_values.append(
                p_value
            )

        # ----------------------------------------------------
        # Registrar fila
        # ----------------------------------------------------

        summary_rows.append({

            "Normalization Method": method_name,

            "Average F1 (Weighted)": mean_weighted,

            "Std F1 (Weighted)": std_weighted,

            "Average F1 (Macro)": mean_macro,

            "Std F1 (Macro)": std_macro,

            "Average F1 (Desertor)": mean_desertor,

            "Std F1 (Desertor)": std_desertor,

            "F1 Delta vs Baseline": delta,

            "T-Statistic": t_stat,

            "P-value vs Baseline": p_value

        })

    # --------------------------------------------------------
    # Corrección de Holm
    # --------------------------------------------------------

    adjusted_p_values = _holm_adjustment(
        raw_p_values
    )

    for idx, adjusted_p in zip(
        comparison_indices,
        adjusted_p_values
    ):

        summary_rows[idx][
            "P-value Holm vs Baseline"
        ] = (
            float(adjusted_p)
            if np.isfinite(adjusted_p)
            else np.nan
        )

    # El baseline se compara consigo mismo.
    for row in summary_rows:

        if row["Normalization Method"] == baseline_name:

            row["P-value Holm vs Baseline"] = 1.0

    # --------------------------------------------------------
    # DataFrame final
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = summary_df.sort_values(
        by="Average F1 (Macro)",
        ascending=False,
        na_position="last"
    )

    return summary_df.reset_index(drop=True)


# ============================================================
# 4. EXPORTACIÓN DE TABLAS A CSV
# ============================================================

def export_summary_to_csv(
    summary_df: pd.DataFrame,
    filepath: Optional[Union[str, Path]] = None
) -> Path:
    """
    Exporta la tabla comparativa a un archivo CSV.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Tabla comparativa de resultados.

    filepath : str o Path, opcional
        Ruta de destino. Si no se especifica, utiliza DEFAULT_METRICS_FILE.

    Returns
    -------
    Path
        Ruta del archivo CSV generado.
    """

    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError(
            "summary_df debe ser un DataFrame de pandas."
        )

    if summary_df.empty:
        raise ValueError(
            "No se puede exportar una tabla vacía."
        )

    if filepath is None:

        METRICS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        target_path = Path(
            DEFAULT_METRICS_FILE
        )

    else:

        target_path = Path(
            filepath
        )

        target_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    summary_df.to_csv(
        target_path,
        index=False,
        encoding="utf-8-sig"
    )

    return target_path


# ============================================================
# 5. AUDITORÍA DESCRIPTIVA DEL RECALL POR SUBGRUPOS
# ============================================================

def compute_subgroup_recall(
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_cols: Optional[List[str]] = None,
    desertor_label: str = "Desertor",
    min_samples: int = 10
) -> pd.DataFrame:
    """
    Calcula descriptivamente el Recall de Desertor por subgrupos.

    Se consideran únicamente los estudiantes cuya clase real
    corresponde a Desertor.

    El Recall de cada subgrupo se calcula como:

        Recall = Desertores correctamente identificados
                 ----------------------------------------
                 Total de desertores reales del subgrupo

    IMPORTANTE:
    Esta función constituye una auditoría descriptiva. Las diferencias
    observadas no demuestran por sí solas discriminación, causalidad
    ni ausencia de equidad algorítmica.

    Parameters
    ----------
    X_test : pd.DataFrame
        Variables originales del conjunto de evaluación.

    y_true : np.ndarray
        Etiquetas reales.

    y_pred : np.ndarray
        Etiquetas predichas.

    subgroup_cols : list of str, opcional
        Variables categóricas por las cuales agrupar.

    desertor_label : str
        Etiqueta textual de la clase Desertor.

    min_samples : int
        Número mínimo de desertores reales para reportar un subgrupo.

    Returns
    -------
    pd.DataFrame
        Tabla con Variable, Subgrupo, N Desertores y Recall (Desertor).
    """

    # --------------------------------------------------------
    # Validaciones
    # --------------------------------------------------------

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            "X_test debe ser un DataFrame de pandas."
        )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            "y_true e y_pred deben ser vectores unidimensionales."
        )

    if not (
        len(X_test)
        == len(y_true)
        == len(y_pred)
    ):
        raise ValueError(
            "X_test, y_true e y_pred deben tener igual "
            "número de registros."
        )

    if min_samples < 1:
        raise ValueError(
            "min_samples debe ser mayor o igual a 1."
        )

    if desertor_label not in CLASSES:
        raise ValueError(
            f"La etiqueta '{desertor_label}' no pertenece "
            f"a las clases definidas: {CLASSES}."
        )

    if subgroup_cols is None:

        subgroup_cols = [
            "sexo",
            "zona_residencia",
            "sector_ies",
            "metodologia"
        ]

    # Eliminar nombres repetidos sin cambiar el orden.
    subgroup_cols = list(
        dict.fromkeys(subgroup_cols)
    )

    # --------------------------------------------------------
    # Construir tabla de evaluación
    # --------------------------------------------------------

    eval_df = X_test.copy()

    eval_df["real"] = y_true
    eval_df["pred"] = y_pred

    # Solo estudiantes cuya clase real es Desertor.
    desertores = eval_df[
        eval_df["real"] == desertor_label
    ]

    subgroup_rows = []

    # --------------------------------------------------------
    # Evaluar cada variable categórica
    # --------------------------------------------------------

    for col in subgroup_cols:

        if col not in desertores.columns:
            continue

        for group, subgroup in desertores.groupby(
            col,
            dropna=False
        ):

            n_desertores = len(subgroup)

            if n_desertores < min_samples:
                continue

            # Proporción de desertores correctamente identificados.
            recall_group = float(
                (
                    subgroup["pred"] == desertor_label
                ).mean()
            )

            subgroup_rows.append({

                "Variable": col,

                "Subgrupo": (
                    "Sin dato"
                    if pd.isna(group)
                    else str(group)
                ),

                "N Desertores": int(n_desertores),

                "Recall (Desertor)": recall_group

            })

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    return pd.DataFrame(
        subgroup_rows,
        columns=[
            "Variable",
            "Subgrupo",
            "N Desertores",
            "Recall (Desertor)"
        ]
    )

