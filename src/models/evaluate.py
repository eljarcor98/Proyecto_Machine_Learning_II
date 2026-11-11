"""
Módulo de Evaluación de Modelos, Experimentos Paralelos y Serialización.
========================================================================
Gestiona la ejecución de experimentos Monte Carlo multiclase para evaluación de métodos
de normalización, la estimación de importancia de variables por permutación y la
serialización/deserialización de modelos entrenados con joblib.

Principales Capacidades:
------------------------
1. _evaluate_single_trial: Evaluación de una partición independiente para los 8 escaladores.
2. run_multiclass_trials: Orquestador paralelizado (con joblib) de N repeticiones sin fuga de datos.
3. fit_best_pipeline: Ajuste final del pipeline óptimo sobre la partición de entrenamiento.
4. save_model / load_model: Persistencia y carga segura de artefactos serializados (.joblib).
5. compute_permutation_importance: Estimación no paramétrica de relevancia de características.

Autor: Arnold Santiago Torres
Machine Learning II - 2026-II
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.pipeline import build_svm_pipeline, get_normalization_scalers
from src.utils.config import (
    CAT_COLS,
    CLASS_MAP,
    DEFAULT_MODEL_FILE,
    MODELS_DIR,
    NUM_COLS
)
from src.utils.metrics import build_summary_comparison_table


def _evaluate_single_trial(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    trial_idx: int,
    scalers_dict: Dict[str, Any],
    num_cols: List[str],
    cat_cols: List[str],
    class_map: Dict[str, int],
    test_size: float = 0.2
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Función de trabajo atómica para paralelización: evalúa una partición independiente (trial)
    entrenando un pipeline independiente para cada uno de los escaladores provistos.

    Parámetros:
    -----------
    X_raw : pd.DataFrame
        Matriz completa de características (sin dividir).
    y : np.ndarray
        Vector completo de etiquetas de clase.
    trial_idx : int
        Índice del ensayo actual (utilizado como offset de la semilla pseudoaleatoria).
    scalers_dict : dict de {str: scaler_instance}
        Diccionario con los estimadores de normalización a evaluar.
    num_cols : list of str
        Variables numéricas.
    cat_cols : list of str
        Variables categóricas.
    class_map : dict de {str: int}
        Mapeo de etiquetas textuales a índices enteros.
    test_size : float, por defecto 0.2
        Proporción reservada para el conjunto de prueba de la iteración.

    Retorna:
    --------
    tuple:
        (trial_scores_weighted, trial_scores_macro, trial_scores_desertor)
    """
    # Generar partición estratificada única para esta iteración (Monte Carlo Cross-Validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=42 + trial_idx,
        stratify=y
    )

    trial_scores_weighted = {}
    trial_scores_macro = {}
    trial_scores_desertor = {}

    idx_des = class_map['Desertor']

    # Iterar sobre cada técnica de normalización dentro de la partición actual
    for method_name, scaler_proto in scalers_dict.items():
        try:
            # Identificar si corresponde a la variante ordinal
            is_ordinal = (method_name == 'Ordinal Encoder')

            # Ensamblar pipeline exclusivo para esta partición
            pipeline = build_svm_pipeline(
                scaler=scaler_proto,
                num_cols=num_cols,
                cat_cols=cat_cols,
                is_ordinal=is_ordinal,
                random_state=42
            )

            # Ajuste estricto sobre X_train (evita fuga de datos)
            pipeline.fit(X_train, y_train)

            # Inferencia sobre datos nunca antes vistos (X_test)
            preds = pipeline.predict(X_test)

            # Cálculo de métricas de desempeño
            trial_scores_weighted[method_name] = float(f1_score(y_test, preds, average='weighted'))
            trial_scores_macro[method_name] = float(f1_score(y_test, preds, average='macro'))
            trial_scores_desertor[method_name] = float(f1_score(y_test, preds, labels=[idx_des], average='macro'))

        except Exception:
            # Captura de fallas numéricas en transformaciones extremas
            trial_scores_weighted[method_name] = np.nan
            trial_scores_macro[method_name] = np.nan
            trial_scores_desertor[method_name] = np.nan

    return trial_scores_weighted, trial_scores_macro, trial_scores_desertor


def run_multiclass_trials(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    n_trials: int = 50,
    n_jobs: int = -1,
    scalers_dict: Optional[Dict[str, Any]] = None,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    class_map: Optional[Dict[str, int]] = None,
    test_size: float = 0.2
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], pd.DataFrame]:
    """
    Ejecuta el experimento de N particiones independientes en paralelo utilizando joblib.
    Consolida las métricas de todas las iteraciones y construye la tabla comparativa con significancia t-test.

    Parámetros:
    -----------
    X_raw : pd.DataFrame
        Matriz de características predictoras.
    y : np.ndarray
        Vector de etiquetas verdaderas.
    n_trials : int, por defecto 50
        Número de iteraciones independientes de validación cruzada Monte Carlo.
    n_jobs : int, por defecto -1
        Número de núcleos de CPU para ejecución concurrente (-1 utiliza todos los disponibles).
    scalers_dict : dict, opcional
        Catálogo de escaladores a evaluar. Por defecto get_normalization_scalers().
    num_cols : list of str, opcional
        Variables continuas y discretas numéricas.
    cat_cols : list of str, opcional
        Variables cualitativas categóricas.
    class_map : dict, opcional
        Mapeo de nombres a enteros.
    test_size : float, por defecto 0.2
        Fracción destinada a prueba en cada iteración.

    Retorna:
    --------
    tuple:
        (results_weighted, results_macro, results_desertor, summary_df)
    """
    if scalers_dict is None:
        scalers_dict = get_normalization_scalers()
    if num_cols is None:
        num_cols = NUM_COLS
    if cat_cols is None:
        cat_cols = CAT_COLS
    if class_map is None:
        class_map = CLASS_MAP

    # Despachar las N iteraciones a través del pool de procesos de joblib
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_single_trial)(
            X_raw=X_raw,
            y=y,
            trial_idx=t,
            scalers_dict=scalers_dict,
            num_cols=num_cols,
            cat_cols=cat_cols,
            class_map=class_map,
            test_size=test_size
        )
        for t in range(n_trials)
    )

    # Inicializar contenedores de recolección de resultados por técnica
    results_weighted = {method: [] for method in scalers_dict.keys()}
    results_macro = {method: [] for method in scalers_dict.keys()}
    results_desertor = {method: [] for method in scalers_dict.keys()}

    # Acumular las métricas devueltas por cada proceso concurrente
    for res_wei, res_mac, res_des in parallel_results:
        for method_name in scalers_dict.keys():
            results_weighted[method_name].append(res_wei[method_name])
            results_macro[method_name].append(res_mac[method_name])
            results_desertor[method_name].append(res_des[method_name])

    # Construir tabla estadística con t-test vs Baseline
    summary_df = build_summary_comparison_table(
        results_weighted=results_weighted,
        results_macro=results_macro,
        results_desertor=results_desertor,
        baseline_name='No Normalization (Baseline)'
    )

    return results_weighted, results_macro, results_desertor, summary_df


def fit_best_pipeline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    scaler: Any = None,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    probability: bool = False,
    random_state: int = 42
) -> Pipeline:
    """
    Ajusta el pipeline óptimo sobre la partición completa de entrenamiento.

    Parámetros:
    -----------
    X_train : pd.DataFrame
        Matriz de características de entrenamiento.
    y_train : np.ndarray
        Vector de etiquetas de entrenamiento.
    scaler : Any, opcional
        Método de escalamiento seleccionado. Por defecto StandardScaler().
    num_cols : list of str, opcional
        Variables numéricas.
    cat_cols : list of str, opcional
        Variables categóricas.
    probability : bool, por defecto False
        Habilitar estimación probabilística de Platt.
    random_state : int, por defecto 42
        Semilla aleatoria.

    Retorna:
    --------
    Pipeline:
        Pipeline ajustado y listo para inferencia o persistencia en disco.
    """
    if scaler is None:
        scaler = StandardScaler()

    # Construir pipeline con preprocesador y clasificador
    pipeline = build_svm_pipeline(
        scaler=scaler,
        num_cols=num_cols,
        cat_cols=cat_cols,
        probability=probability,
        random_state=random_state
    )

    # Ajustar pipeline sobre los datos de entrenamiento
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(
    pipeline: Pipeline,
    filepath: Optional[Union[str, Path]] = None
) -> Path:
    """
    Serializa el pipeline ajustado en disco utilizando joblib.

    Parámetros:
    -----------
    pipeline : Pipeline
        Pipeline ajustado de Scikit-Learn.
    filepath : str o Path, opcional
        Ruta destino para guardar el artefacto (.joblib). Por defecto DEFAULT_MODEL_FILE.

    Retorna:
    --------
    Path:
        Ruta absoluta del archivo guardado.
    """
    if filepath is None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        target_path = DEFAULT_MODEL_FILE
    else:
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)

    dump(pipeline, target_path)
    return target_path


def load_model(filepath: Optional[Union[str, Path]] = None) -> Pipeline:
    """
    Carga un pipeline previamente serializado desde el disco.

    Parámetros:
    -----------
    filepath : str o Path, opcional
        Ruta del archivo .joblib a cargar. Por defecto DEFAULT_MODEL_FILE.

    Retorna:
    --------
    Pipeline:
        Pipeline deserializado listo para predicciones.

    Lanza:
    ------
    FileNotFoundError:
        Si el archivo no existe en la ruta especificada.
    """
    if filepath is None:
        target_path = DEFAULT_MODEL_FILE
    else:
        target_path = Path(filepath)

    if not target_path.exists():
        raise FileNotFoundError(f"No existe el archivo de modelo serializado en: {target_path}")

    return load(target_path)


def compute_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_repeats: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Calcula la Importancia de Variables por Permutación (Permutation Feature Importance).
    Metodología no paramétrica que permuta aleatoriamente los valores de cada variable
    en el conjunto de prueba y cuantifica la caída resultante en el rendimiento (F1-Score).

    Parámetros:
    -----------
    pipeline : Pipeline
        Pipeline ajustado a evaluar.
    X_test : pd.DataFrame
        Matriz de características de prueba.
    y_test : np.ndarray
        Vector de etiquetas verdaderas de prueba.
    n_repeats : int, por defecto 5
        Número de veces que se baraja cada variable para estimar la varianza de la métrica.
    random_state : int, por defecto 42
        Semilla para el generador de permutaciones.
    n_jobs : int, por defecto -1
        Núcleos paralelos para el cómputo de permutaciones.

    Retorna:
    --------
    pd.DataFrame:
        DataFrame con las variables ordenadas descendentemente por importancia media ('Importance_Mean')
        junto con su desviación estándar ('Importance_Std').
    """
    # Ejecutar la inspección de permutaciones de Scikit-Learn
    perm_importance = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs
    )

    # Ordenar índices de mayor a menor impacto en el clasificador
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    feature_names = X_test.columns

    # Estructurar resultados en DataFrame ordenado
    imp_df = pd.DataFrame({
        'Feature': feature_names[sorted_idx],
        'Importance_Mean': perm_importance.importances_mean[sorted_idx],
        'Importance_Std': perm_importance.importances_std[sorted_idx]
    })

    return imp_df.reset_index(drop=True)
