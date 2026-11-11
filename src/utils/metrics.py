"""
Módulo de Métricas y Evaluación Estadística.
=============================================
Provee funciones especializadas para:
1. Cálculo de métricas multiclase (Accuracy, F1-Ponderado, F1-Macro, Recall por clase).
2. Construcción de tablas estadísticas comparativas de repeticiones experimentales (trials).
3. Prueba de significancia estadística (t-test de Welch) de cada escalador frente al baseline.
4. Auditoría de equidad algorítmica y sesgo (desglose de Recall de deserción por subgrupos).
5. Exportación estandarizada de métricas a archivos CSV.

Autor: Arnold Santiago Torres
Machine Learning II - 2026-II
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
)

from src.utils.config import (
    CLASSES,
    CLASS_MAP,
    DEFAULT_METRICS_FILE,
    METRICS_DIR
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    desertor_label: int = CLASS_MAP['Desertor']
) -> Dict[str, Any]:
    """
    Calcula un compendio integral de métricas multiclase para evaluación de modelos SVM.

    Parámetros:
    -----------
    y_true : np.ndarray
        Vector unidimensional con las etiquetas verdaderas de clase (enteros).
    y_pred : np.ndarray
        Vector unidimensional con las predicciones del clasificador (enteros).
    target_names : list of str, opcional
        Nombres cualitativos de las clases. Si es None, toma ['Graduado', 'En curso', 'Desertor'].
    desertor_label : int, por defecto 2
        Índice de la clase de interés prioritario ('Desertor') para cálculo de sensibilidad focal.

    Retorna:
    --------
    dict:
        Diccionario con las métricas escalares ('accuracy', 'f1_weighted', 'f1_macro', 'recall_desertor'),
        la matriz de confusión cruda ('confusion_matrix') y el reporte de texto ('classification_report').
    """
    if target_names is None:
        target_names = CLASSES

    # Accuracy global: proporción total de instancias correctamente clasificadas
    acc = accuracy_score(y_true, y_pred)

    # F1 ponderado (Weighted): compensa el desbalance de clases ponderando por el soporte de cada clase
    f1_wei = f1_score(y_true, y_pred, average='weighted')

    # F1 macro: promedio no ponderado de las 3 clases, tratando a cada una con igual importancia
    f1_mac = f1_score(y_true, y_pred, average='macro')

    # Recall de la clase crítica ('Desertor'): porcentaje de estudiantes que realmente desertan identificados
    rec_des = recall_score(y_true, y_pred, labels=[desertor_label], average='macro')

    # Matriz de confusión absoluta (3x3)
    cm = confusion_matrix(y_true, y_pred)

    # Reporte detallado de clasificación por clase con 4 cifras decimales
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)

    return {
        'accuracy': float(acc),
        'f1_weighted': float(f1_wei),
        'f1_macro': float(f1_mac),
        'recall_desertor': float(rec_des),
        'confusion_matrix': cm,
        'classification_report': report
    }


def build_summary_comparison_table(
    results_weighted: Dict[str, List[float]],
    results_macro: Dict[str, List[float]],
    results_desertor: Dict[str, List[float]],
    baseline_name: str = 'No Normalization (Baseline)'
) -> pd.DataFrame:
    """
    Sintetiza estadísticamente las repeticiones (trials) de validación independiente,
    evaluando la hipótesis nula de equivalencia de medias frente al método baseline (sin normalizar)
    mediante la prueba t-test de Student de dos muestras independientes (Welch, varianzas desiguales).

    Parámetros:
    -----------
    results_weighted : dict de {str: list of float}
        Puntuaciones de F1-Score Ponderado obtenidas en cada repetición por escalador.
    results_macro : dict de {str: list of float}
        Puntuaciones de F1-Score Macro obtenidas en cada repetición por escalador.
    results_desertor : dict de {str: list of float}
        Puntuaciones de Recall / F1 de la clase 'Desertor' obtenidas en cada repetición.
    baseline_name : str, por defecto 'No Normalization (Baseline)'
        Identificador del método de referencia frente al cual se calculan los deltas y el p-valor.

    Retorna:
    --------
    pd.DataFrame:
        Tabla ordenada descendentemente por F1-Score ponderado promedio, con medias, desviaciones estándar,
        deltas frente al baseline, estadísticos t y p-valores correspondientes.
    """
    # Extraer puntuaciones de la distribución de referencia (baseline)
    baseline_scores_wei = results_weighted.get(baseline_name, [])
    mean_baseline = np.mean(baseline_scores_wei) if len(baseline_scores_wei) > 0 else 0.0

    summary_rows = []

    # Iterar secuencialmente sobre cada técnica de normalización evaluada
    for method_name, scores_wei in results_weighted.items():
        scores_mac = results_macro.get(method_name, [])
        scores_des = results_desertor.get(method_name, [])

        # Estadísticos descriptivos (media y desviación estándar muestral)
        mean_f1_wei = float(np.mean(scores_wei))
        std_f1_wei = float(np.std(scores_wei))
        mean_f1_mac = float(np.mean(scores_mac)) if scores_mac else np.nan
        mean_f1_des = float(np.mean(scores_des)) if scores_des else np.nan

        # Contraste de hipótesis: Welch t-test (asume varianzas posiblemente heterogéneas)
        if method_name != baseline_name and len(baseline_scores_wei) > 0:
            t_stat, p_val = ttest_ind(scores_wei, baseline_scores_wei, equal_var=False)
            delta = mean_f1_wei - mean_baseline
        else:
            # Caso de auto-comparación del baseline: delta nulo y significancia unitaria
            t_stat, p_val, delta = 0.0, 1.0, 0.0

        summary_rows.append({
            'Normalization Method': method_name,
            'Average F1 (Weighted)': mean_f1_wei,
            'Average F1 (Macro)': mean_f1_mac,
            'Average F1 (Desertor)': mean_f1_des,
            'Std Dev': std_f1_wei,
            'F1 Delta vs Baseline': float(delta),
            'T-Statistic': float(t_stat),
            'P-value vs Baseline': float(p_val)
        })

    # Ordenar de mayor a menor efectividad predictiva en F1-Score Ponderado
    summary_df = pd.DataFrame(summary_rows).sort_values(by='Average F1 (Weighted)', ascending=False)
    return summary_df.reset_index(drop=True)


def export_summary_to_csv(
    summary_df: pd.DataFrame,
    filepath: Optional[Union[str, Path]] = None
) -> Path:
    """
    Persiste la tabla comparativa de desempeño en formato CSV estándar.

    Parámetros:
    -----------
    summary_df : pd.DataFrame
        DataFrame estructurado con las métricas consolidadas.
    filepath : str o Path, opcional
        Ruta destino del archivo CSV. Si no se indica, utiliza DEFAULT_METRICS_FILE.

    Retorna:
    --------
    Path:
        Ruta absoluta del archivo exportado.
    """
    if filepath is None:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        target_path = DEFAULT_METRICS_FILE
    else:
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(target_path, index=False)
    return target_path


def compute_subgroup_recall(
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_cols: Optional[List[str]] = None,
    desertor_label: int = CLASS_MAP['Desertor'],
    min_samples: int = 10
) -> pd.DataFrame:
    """
    Auditoría de Equidad Algorítmica (Fairness Audit):
    Evalúa si la capacidad de detección del modelo (Recall de deserción) se degrada
    o sesga significativamente al analizar subgrupos sociodemográficos particulares.

    Parámetros:
    -----------
    X_test : pd.DataFrame
        Matriz de características originales de los estudiantes del conjunto de prueba.
    y_true : np.ndarray
        Vector con las etiquetas verdaderas reales.
    y_pred : np.ndarray
        Vector con las predicciones generadas por el modelo evaluado.
    subgroup_cols : list of str, opcional
        Columnas sobre las cuales estratificar la auditoría (género, residencia, sector IES, etc.).
    desertor_label : int, por defecto 2
        Identificador de la clase 'Desertor'.
    min_samples : int, por defecto 10
        Umbral mínimo de estudiantes desertores para que el subgrupo sea estadísticamente informativo.

    Retorna:
    --------
    pd.DataFrame:
        Tabla con variable, subgrupo, tamaño de muestra y tasa de acierto (Recall).
    """
    if subgroup_cols is None:
        subgroup_cols = ['sexo', 'zona_residencia', 'sector_ies', 'metodologia']

    # Crear copia local con variables predictoras y estados observados/predichos
    eval_df = X_test.copy()
    eval_df['real'] = y_true
    eval_df['pred'] = y_pred

    # Filtrar únicamente la población que realmente se encuentra en deserción
    desertores = eval_df[eval_df['real'] == desertor_label]
    subgroup_rows = []

    # Ciclo iterativo sobre cada dimensión sociodemográfica analizada
    for col in subgroup_cols:
        if col not in eval_df.columns:
            continue

        # Agrupar por las categorías de la variable y calcular la tasa de acierto en cada una
        for grupo, sub in desertores.groupby(col):
            # Se descartan subgrupos con soporte muestral insignificante
            if len(sub) >= min_samples:
                # El recall del subgrupo es la media de predicciones acertadas como desertor
                rec_g = (sub['pred'] == desertor_label).mean()
                subgroup_rows.append({
                    'Variable': col,
                    'Subgrupo': str(grupo),
                    'N Desertores': int(len(sub)),
                    'Recall (Desertor)': float(rec_g)
                })

    return pd.DataFrame(subgroup_rows)
