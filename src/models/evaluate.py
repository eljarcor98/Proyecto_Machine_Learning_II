"""
Módulo de Evaluación de Modelos, Experimentos Monte Carlo y Serialización.
============================================================================

Incluye:
1. Evaluación repetida de ocho configuraciones SVM.
2. Comparación de kernels RBF y polinomial.
3. DummyClassifier como baseline predictiva independiente.
4. Selección de normalizador mediante F1 Macro.
5. Entrenamiento final y serialización.
6. Importancia de variables por permutación.

El conjunto de test final debe permanecer fuera de este módulo durante
la selección experimental.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from joblib import Parallel, delayed, dump, load

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance

from src.models.pipeline import (
    build_svm_pipeline,
    get_normalization_scalers
)

from src.utils.config import (
    CLASSES,
    NUM_COLS,
    CAT_COLS
)

from src.utils.metrics import build_summary_comparison_table


# =============================================================================
# CONSTANTES
# =============================================================================

DUMMY_NAME = "DummyClassifier (Most Frequent)"
SVM_BASELINE_NAME = "No Normalization (Baseline)"


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def _get_scaler_information(scaler_name, scaler_info):
    """
    Obtiene el transformador y determina si la configuración es ordinal.

    Compatible con:
    - Transformadores sklearn devueltos directamente.
    - Diccionarios con las claves 'scaler' e 'is_ordinal'.
    """

    if isinstance(scaler_info, dict):
        scaler = scaler_info.get("scaler")
        is_ordinal = scaler_info.get(
            "is_ordinal",
            scaler_name.lower().startswith("ordinal")
        )
    else:
        scaler = scaler_info
        is_ordinal = scaler_name.lower().startswith("ordinal")

    return scaler, is_ordinal


def _calculate_trial_metrics(y_true, y_pred):
    """
    Calcula las tres métricas utilizadas durante los experimentos.
    """

    return (
        f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        f1_score(
            y_true,
            y_pred,
            labels=["Desertor"],
            average="macro",
            zero_division=0
        )
    )


# =============================================================================
# 1. EVALUACIÓN DE UN TRIAL
# =============================================================================

def _evaluate_single_trial(
    X_dev,
    y_dev,
    trial_idx,
    scalers_dict,
    num_cols,
    cat_cols,
    test_size=0.2,
    kernel="rbf",
    degree=3,
    coef0=1.0
):
    """
    Ejecuta una partición estratificada y evalúa:

    - Las ocho configuraciones SVM.
    - DummyClassifier con estrategia most_frequent.

    Todos los modelos utilizan exactamente la misma partición.
    """

    y_dev = pd.Series(y_dev).reset_index(drop=True)

    if y_dev.isna().any():
        raise ValueError("y_dev contiene valores nulos.")

    y_dev = y_dev.astype(str)

    X_dev = X_dev.reset_index(drop=True)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_dev,
        y_dev,
        test_size=test_size,
        random_state=42 + trial_idx,
        stratify=y_dev
    )

    results_weighted = {}
    results_macro = {}
    results_desertor = {}

    # -------------------------------------------------------------------------
    # SVM: configuraciones de normalización
    # -------------------------------------------------------------------------

    for scaler_name, scaler_info in scalers_dict.items():

        scaler, is_ordinal = _get_scaler_information(
            scaler_name,
            scaler_info
        )

        model = build_svm_pipeline(
            scaler=scaler,
            num_cols=num_cols,
            cat_cols=cat_cols,
            kernel=kernel,
            degree=degree,
            coef0=coef0,
            probability=False,
            random_state=42 + trial_idx,
            is_ordinal=is_ordinal
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_valid)

        (
            score_weighted,
            score_macro,
            score_desertor
        ) = _calculate_trial_metrics(
            y_valid,
            predictions
        )

        results_weighted[scaler_name] = score_weighted
        results_macro[scaler_name] = score_macro
        results_desertor[scaler_name] = score_desertor

    # -------------------------------------------------------------------------
    # DummyClassifier: baseline predictiva independiente
    # -------------------------------------------------------------------------

    dummy = DummyClassifier(
        strategy="most_frequent"
    )

    dummy.fit(X_train, y_train)

    dummy_predictions = dummy.predict(X_valid)

    (
        dummy_weighted,
        dummy_macro,
        dummy_desertor
    ) = _calculate_trial_metrics(
        y_valid,
        dummy_predictions
    )

    results_weighted[DUMMY_NAME] = dummy_weighted
    results_macro[DUMMY_NAME] = dummy_macro
    results_desertor[DUMMY_NAME] = dummy_desertor

    return (
        results_weighted,
        results_macro,
        results_desertor
    )


# =============================================================================
# 2. EXPERIMENTOS MONTE CARLO
# =============================================================================

def run_multiclass_trials(
    X_dev,
    y_dev,
    n_trials=50,
    n_jobs=-1,
    scalers_dict=None,
    num_cols=None,
    cat_cols=None,
    test_size=0.2,
    kernel="rbf",
    degree=3,
    coef0=1.0
):
    """
    Ejecuta experimentos Monte Carlo para las ocho configuraciones SVM
    y DummyClassifier.

    Retorna:
        results_weighted
        results_macro
        results_desertor
        summary_df
    """

    if scalers_dict is None:
        scalers_dict = get_normalization_scalers()

    if num_cols is None:
        num_cols = NUM_COLS

    if cat_cols is None:
        cat_cols = CAT_COLS

    if not isinstance(X_dev, pd.DataFrame):
        raise TypeError("X_dev debe ser un DataFrame.")

    if len(X_dev) != len(y_dev):
        raise ValueError(
            "X_dev e y_dev tienen diferente número de registros."
        )

    if n_trials < 2:
        raise ValueError("n_trials debe ser al menos 2.")

    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")

    if not isinstance(scalers_dict, dict) or not scalers_dict:
        raise ValueError(
            "scalers_dict debe ser un diccionario no vacío."
        )

    if SVM_BASELINE_NAME not in scalers_dict:
        raise ValueError(
            f"Falta la configuración obligatoria: {SVM_BASELINE_NAME}"
        )

    y_dev = pd.Series(y_dev).reset_index(drop=True)

    if y_dev.isna().any():
        raise ValueError("y_dev contiene valores nulos.")

    y_dev = y_dev.astype(str)

    if set(y_dev.unique()) != set(CLASSES):
        raise ValueError(
            "El conjunto de desarrollo debe contener las tres clases: "
            f"{CLASSES}"
        )

    if len(y_dev) != len(X_dev):
        raise ValueError(
            "X_dev e y_dev tienen diferente número de registros."
        )

    print(
        f"\nEjecutando {n_trials} trials para kernel {kernel.upper()}..."
    )

    trial_results = Parallel(
        n_jobs=n_jobs,
        verbose=5
    )(
        delayed(_evaluate_single_trial)(
            X_dev=X_dev,
            y_dev=y_dev,
            trial_idx=trial_idx,
            scalers_dict=scalers_dict,
            num_cols=num_cols,
            cat_cols=cat_cols,
            test_size=test_size,
            kernel=kernel,
            degree=degree,
            coef0=coef0
        )
        for trial_idx in range(n_trials)
    )

    if not trial_results:
        raise RuntimeError(
            "No se obtuvieron resultados de los experimentos."
        )

    results_weighted = {
        name: [trial[0][name] for trial in trial_results]
        for name in trial_results[0][0]
    }

    results_macro = {
        name: [trial[1][name] for trial in trial_results]
        for name in trial_results[0][1]
    }

    results_desertor = {
        name: [trial[2][name] for trial in trial_results]
        for name in trial_results[0][2]
    }

    summary_df = build_summary_comparison_table(
        results_weighted=results_weighted,
        results_macro=results_macro,
        results_desertor=results_desertor,
        baseline_name=SVM_BASELINE_NAME
    )

    print(
        f"-> Finalizados {n_trials} trials para {kernel.upper()}."
    )

    return (
        results_weighted,
        results_macro,
        results_desertor,
        summary_df
    )


# =============================================================================
# 3. SELECCIÓN DE LA MEJOR CONFIGURACIÓN SVM
# =============================================================================

def select_best_configuration(
    results_macro,
    scalers_dict
):
    """
    Selecciona exclusivamente entre las configuraciones SVM.

    DummyClassifier queda excluido porque no es un normalizador
    ni una configuración SVM.
    """

    svm_names = [
        name
        for name in scalers_dict
        if name in results_macro
    ]

    if not svm_names:
        raise ValueError(
            "No hay configuraciones SVM disponibles para seleccionar."
        )

    mean_scores = {}

    for name in svm_names:

        scores = np.asarray(
            results_macro[name],
            dtype=float
        )

        if scores.size == 0 or np.isnan(scores).all():
            raise ValueError(
                f"No existen puntuaciones válidas para {name}."
            )

        mean_scores[name] = float(np.nanmean(scores))

    best_name = max(
        mean_scores,
        key=mean_scores.get
    )

    best_info = scalers_dict[best_name]

    best_scaler, best_ordinal = _get_scaler_information(
        best_name,
        best_info
    )

    return (
        best_name,
        best_scaler,
        best_ordinal
    )


# =============================================================================
# 4. ENTRENAMIENTO FINAL SVM
# =============================================================================

def fit_best_pipeline(
    X_train,
    y_train,
    scaler=None,
    num_cols=None,
    cat_cols=None,
    kernel="rbf",
    degree=3,
    coef0=1.0,
    probability=False,
    random_state=42,
    is_ordinal=False
):
    """
    Entrena el pipeline SVM seleccionado utilizando todo el conjunto
    de desarrollo.
    """

    if num_cols is None:
        num_cols = NUM_COLS

    if cat_cols is None:
        cat_cols = CAT_COLS

    if scaler is None:
        raise ValueError(
            "Debe proporcionarse el normalizador seleccionado."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train e y_train tienen diferente número de registros."
        )

    model = build_svm_pipeline(
        scaler=scaler,
        num_cols=num_cols,
        cat_cols=cat_cols,
        kernel=kernel,
        degree=degree,
        coef0=coef0,
        probability=probability,
        random_state=random_state,
        is_ordinal=is_ordinal
    )

    model.fit(
        X_train,
        pd.Series(y_train).astype(str)
    )

    return model


# =============================================================================
# 5. ENTRENAMIENTO FINAL DUMMY
# =============================================================================

def fit_dummy_classifier(
    X_train,
    y_train,
    strategy="most_frequent"
):
    """
    Entrena el baseline predictivo DummyClassifier.

    No utiliza las características para aprender relaciones predictivas.
    La estrategia most_frequent predice siempre la clase mayoritaria
    observada en el conjunto de entrenamiento.
    """

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train e y_train tienen diferente número de registros."
        )

    model = DummyClassifier(
        strategy=strategy
    )

    model.fit(
        X_train,
        pd.Series(y_train).astype(str)
    )

    return model


# =============================================================================
# 6. SERIALIZACIÓN
# =============================================================================

def save_model(
    model,
    filepath
):
    """Guarda un modelo utilizando joblib."""

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dump(model, filepath)

    return filepath


def load_model(filepath):
    """Carga un modelo previamente serializado."""

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {filepath}"
        )

    return load(filepath)


# =============================================================================
# 7. IMPORTANCIA POR PERMUTACIÓN
# =============================================================================

def compute_permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42,
    n_jobs=-1
):
    """
    Calcula importancia por permutación usando F1 Macro.

    Este análisis es posterior a la evaluación y no debe utilizarse
    para seleccionar o reajustar el modelo.
    """

    if n_repeats < 1:
        raise ValueError("n_repeats debe ser al menos 1.")

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test debe ser un DataFrame.")

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test e y_test tienen diferente número de registros."
        )

    result = permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance Mean": result.importances_mean,
        "Importance Std": result.importances_std
    })

    importance_df = importance_df.sort_values(
        by="Importance Mean",
        ascending=False
    ).reset_index(drop=True)

    return importance_df