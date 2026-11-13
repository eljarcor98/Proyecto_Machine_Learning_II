
"""
Pipeline Orquestador del Experimento SVM
========================================

Predicción multiclase de deserción estudiantil sobre dataset sintético
inspirado en el contexto colombiano de SPADIES 3.0.

Características:
    - Ocho configuraciones de normalización y codificación.
    - Comparación de kernels RBF y polinomial de grado 3.
    - Experimentos Monte Carlo sobre el conjunto de desarrollo.
    - DummyClassifier como baseline predictiva independiente.
    - Selección de configuración mediante F1 Macro.
    - Test final independiente.
    - Permutation Importance con F1 Macro.
    - Explicabilidad SHAP, LIME y PDP para RBF y Poly-3.
    - Análisis PCA y visualizaciones de decisión.
    - Auditoría descriptiva del recall de Desertor por subgrupos.

IMPORTANTE:
Los registros y las etiquetas son sintéticos. No representan registros
administrativos oficiales de SPADIES 3.0.

Machine Learning II - 2026-II
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier


# =============================================================================
# 0. CONFIGURACIÓN DE RUTAS E IMPORTACIONES
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# PREPROCESAMIENTO
# -----------------------------------------------------------------------------

from src.data import (
    compute_outlier_diagnostics,
    create_spadies_target,
    get_train_test_split,
    load_data
)


# -----------------------------------------------------------------------------
# EVALUACIÓN Y ENTRENAMIENTO
# -----------------------------------------------------------------------------

from src.models.evaluate import (
    DUMMY_NAME,
    run_multiclass_trials,
    select_best_configuration,
    fit_best_pipeline,
    fit_dummy_classifier,
    compute_permutation_importance,
    save_model
)


# -----------------------------------------------------------------------------
# CONFIGURACIONES DE NORMALIZACIÓN
# -----------------------------------------------------------------------------

from src.models.pipeline import (
    get_normalization_scalers
)


# -----------------------------------------------------------------------------
# MÉTRICAS Y CONFIGURACIÓN GENERAL
# -----------------------------------------------------------------------------

from src.utils import (
    CLASSES,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    NUM_COLS,
    compute_classification_metrics,
    compute_subgroup_recall,
    export_summary_to_csv
)


# -----------------------------------------------------------------------------
# VISUALIZACIONES
# -----------------------------------------------------------------------------

from src.visualization.plots import (
    plot_eda_distributions,
    plot_key_variables_distribution,
    plot_scaler_impact,
    plot_normalization_comparison,
    plot_permutation_importance,
    plot_pca_hyperplane,
    plot_confusion_matrices,
    plot_decision_boundary_top2,
    plot_decision_function_distribution,
    plot_classification_metrics,
    plot_trial_score_distributions,
    plot_desertor_metrics
)


# -----------------------------------------------------------------------------
# EXPLICABILIDAD: SHAP, LIME Y PDP
# -----------------------------------------------------------------------------

from src.models.interpretability import (
    run_model_interpretability
)


# =============================================================================
# 1. FUNCIONES AUXILIARES
# =============================================================================

def print_section(title):
    """Imprime un encabezado uniforme para cada etapa."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80, flush=True)


def print_summary_table(summary_df, title):
    """Imprime una tabla comparativa con formato numérico."""

    print(f"\n{title}")

    if summary_df is None or summary_df.empty:
        print("No hay resultados disponibles para mostrar.")
        return

    numeric_formats = {
        "Average F1 (Weighted)": "{:.4f}",
        "Std F1 (Weighted)": "{:.4f}",
        "Average F1 (Macro)": "{:.4f}",
        "Std F1 (Macro)": "{:.4f}",
        "Average F1 (Desertor)": "{:.4f}",
        "Std F1 (Desertor)": "{:.4f}",
        "F1 Delta vs Baseline": "{:+.4f}",
        "T-Statistic": "{:.3f}",
        "P-value vs Baseline": "{:.6f}",
        "P-value Holm vs Baseline": "{:.6f}",
        "P-value Holm": "{:.6f}"
    }

    formatters = {
        column: (
            lambda value, fmt=fmt:
                fmt.format(value) if pd.notna(value) else "NaN"
        )
        for column, fmt in numeric_formats.items()
        if column in summary_df.columns
    }

    print(
        summary_df.to_string(
            index=False,
            formatters=formatters
        )
    )


def print_final_metrics(metrics, model_name):
    """Presenta las métricas de evaluación de un modelo."""

    print(f"\nREPORTE DE CLASIFICACIÓN [{model_name}]")

    print(metrics["classification_report"])

    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"F1 Weighted:       {metrics['f1_weighted']:.4f}")
    print(f"F1 Macro:          {metrics['f1_macro']:.4f}")
    print(f"Recall Desertor:   {metrics['recall_desertor']:.4f}")


def save_dataframe(df, filepath):
    """Guarda un DataFrame y crea el directorio de destino."""

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        filepath,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"-> Archivo guardado: {filepath}")

    return filepath


def print_figure_result(result, description):
    """
    Procesa el resultado de una función de visualización.

    Las funciones de plots.py retornan normalmente (fig, ax)
    o (fig, axes). Después de guardar, se cierra la figura.
    """

    if isinstance(result, tuple) and len(result) >= 1:

        fig = result[0]

        print(
            f"-> {description}: figura generada y guardada."
        )

        import matplotlib.pyplot as plt

        plt.close(fig)

    elif result is not None:

        print(f"-> {description}: {result}")

    else:

        print(
            f"-> No se obtuvo resultado para: {description}"
        )


def validate_dataset(df):
    """Comprueba que el dataset tenga las clases y columnas esperadas."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "load_data() debe devolver un pandas DataFrame."
        )

    if df.empty:
        raise ValueError("El dataset está vacío.")

    if "estado_final" not in df.columns:
        raise ValueError(
            "No se encontró la columna objetivo 'estado_final'."
        )

    if df["estado_final"].isna().any():
        raise ValueError(
            "La variable objetivo contiene valores nulos."
        )

    missing_classes = set(CLASSES) - set(
        df["estado_final"].unique()
    )

    if missing_classes:
        raise ValueError(
            "Faltan clases objetivo en el dataset: "
            f"{sorted(missing_classes)}"
        )

    return True


def validate_experiment_results(
    results_weighted,
    results_macro,
    results_desertor,
    summary_df,
    experiment_name
):
    """Comprueba que los resultados del experimento sean consistentes."""

    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError(
            f"El resumen de {experiment_name} debe ser un DataFrame."
        )

    if summary_df.empty:
        raise ValueError(
            f"El resumen de {experiment_name} está vacío."
        )

    dictionaries = [
        results_weighted,
        results_macro,
        results_desertor
    ]

    reference_keys = set(results_macro.keys())

    for results in dictionaries:

        if set(results.keys()) != reference_keys:
            raise ValueError(
                f"Las configuraciones de resultados de "
                f"{experiment_name} no coinciden."
            )

    for name in reference_keys:

        lengths = [
            len(results[name])
            for results in dictionaries
        ]

        if len(set(lengths)) != 1:
            raise ValueError(
                f"El número de trials no coincide para "
                f"{name} en {experiment_name}."
            )

    required_columns = {
        "Normalization Method",
        "Average F1 (Macro)"
    }

    missing_columns = required_columns - set(summary_df.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en el resumen de {experiment_name}: "
            f"{sorted(missing_columns)}"
        )

    if DUMMY_NAME not in reference_keys:
        raise ValueError(
            f"No se encontró {DUMMY_NAME} en los resultados."
        )

    return True


def validate_prediction_classes(y_true, y_pred):
    """Comprueba que las predicciones utilicen las etiquetas esperadas."""

    true_classes = set(pd.Series(y_true).dropna().unique())
    predicted_classes = set(pd.Series(y_pred).dropna().unique())

    expected_classes = set(CLASSES)

    if not true_classes.issubset(expected_classes):
        raise ValueError(
            "Se encontraron etiquetas desconocidas en y_true."
        )

    if not predicted_classes.issubset(expected_classes):
        raise ValueError(
            "Se encontraron etiquetas desconocidas en y_pred."
        )


def create_final_metrics_row(
    model_name,
    configuration,
    metrics
):
    """Construye una fila uniforme para las métricas finales."""

    return {
        "Model": model_name,
        "Best Configuration": configuration,
        "Accuracy": metrics["accuracy"],
        "F1 Weighted": metrics["f1_weighted"],
        "F1 Macro": metrics["f1_macro"],
        "Recall Desertor": metrics["recall_desertor"]
    }


# =============================================================================
# 2. FUNCIÓN PRINCIPAL
# =============================================================================

def main(n_trials=50, n_jobs=-1):
    """
    Ejecuta el pipeline experimental completo de predicción de deserción con SVM.

    Parámetros
    ----------
    n_trials : int, opcional (por defecto 50)
        Número de particiones en la validación Monte Carlo Repeated Holdout.
    n_jobs : int, opcional (por defecto -1)
        Número de procesos paralelos para la ejecución concurrente de trials.
    """

    print("=" * 80)
    print("PROYECTO MACHINE LEARNING II: PREDICCIÓN DE DESERCIÓN CON SVM")
    print("Dataset sintético inspirado en el contexto SPADIES 3.0")
    print("=" * 80, flush=True)

    # -------------------------------------------------------------------------
    # CREACIÓN DE DIRECTORIOS
    # -------------------------------------------------------------------------

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. CARGA Y PREPARACIÓN DEL DATASET
    # -------------------------------------------------------------------------

    print_section("1. CARGA Y PREPARACIÓN DEL DATASET")

    raw_df = load_data()

    print(
        f"-> Dimensiones iniciales: "
        f"{raw_df.shape[0]} estudiantes, "
        f"{raw_df.shape[1]} columnas."
    )

    df = create_spadies_target(raw_df)

    validate_dataset(df)

    print(
        f"-> Valores nulos totales: "
        f"{df.isnull().sum().sum()}"
    )

    print("\nDistribución de la variable objetivo:")
    print(df["estado_final"].value_counts())

    print("\nDistribución porcentual:")

    print(
        (
            df["estado_final"].value_counts(normalize=True) * 100
        ).round(2).astype(str) + "%"
    )

    print(
        "\nNOTA: La variable objetivo es sintética y se construye "
        "a partir de características del dataset."
    )

    # -------------------------------------------------------------------------
    # 2. DIAGNÓSTICO ESTADÍSTICO
    # -------------------------------------------------------------------------

    print_section("2. DIAGNÓSTICO ESTADÍSTICO DE VARIABLES")

    diag_df = compute_outlier_diagnostics(
        df,
        NUM_COLS
    )

    if diag_df is not None and not diag_df.empty:

        print(
            diag_df.to_string(
                index=False,
                formatters={
                    "Media": "{:.2f}".format,
                    "Mediana": "{:.2f}".format,
                    "IQR": "{:.2f}".format,
                    "Outliers %": "{:.2f}%".format,
                    "Skewness": "{:.2f}".format
                }
            )
        )

    else:

        print("-> No se generaron resultados de diagnóstico.")

    # -------------------------------------------------------------------------
    # 3. VISUALIZACIONES EXPLORATORIAS
    # -------------------------------------------------------------------------

    print_section(
        "3. GENERACIÓN DE VISUALIZACIONES EXPLORATORIAS"
    )

    result = plot_eda_distributions(
        df,
        output_path=(
            FIGURES_DIR / "eda_distribucion_correlaciones.png"
        )
    )

    print_figure_result(
        result,
        "Distribuciones y correlaciones"
    )

    result = plot_key_variables_distribution(
        df,
        output_path=(
            FIGURES_DIR / "eda_variables_clave.png"
        )
    )

    print_figure_result(
        result,
        "Distribución de variables académicas"
    )

    result = plot_scaler_impact(
        df,
        feature="distancia_hogar_ies_km",
        output_path=(
            FIGURES_DIR / "impacto_transformadores.png"
        )
    )

    print_figure_result(
        result,
        "Impacto de los transformadores"
    )

    print(
        "-> Las correlaciones y distribuciones son descriptivas; "
        "no representan relaciones causales."
    )

    # -------------------------------------------------------------------------
    # 4. DIVISIÓN ÚNICA DEL DATASET
    # -------------------------------------------------------------------------

    print_section(
        "4. DIVISIÓN DEL DATASET: DESARROLLO 80% / TEST FINAL 20%"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_raw,
        y
    ) = get_train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    X_dev = X_train.copy()
    y_dev = y_train.copy()

    print(f"-> Registros para desarrollo: {len(X_dev)}")
    print(f"-> Registros para test final: {len(X_test)}")

    print(
        "-> El test final permanece reservado durante "
        "los experimentos Monte Carlo."
    )

    # -------------------------------------------------------------------------
    # 5. CONFIGURACIONES DE NORMALIZACIÓN
    # -------------------------------------------------------------------------

    print_section("5. CONFIGURACIONES DE NORMALIZACIÓN")

    scalers_dict = get_normalization_scalers()

    if len(scalers_dict) != 8:
        raise ValueError(
            "El experimento requiere exactamente 8 configuraciones SVM. "
            f"Se encontraron {len(scalers_dict)}."
        )

    print(
        f"-> Configuraciones SVM disponibles: {len(scalers_dict)}"
    )

    for name in scalers_dict:
        print(f"   - {name}")

    print(f"   - {DUMMY_NAME} (baseline predictiva independiente)")

    # -------------------------------------------------------------------------
    # 6. EXPERIMENTO CON KERNEL RBF
    # -------------------------------------------------------------------------

    print_section(
        f"6. EXPERIMENTO RBF: 8 CONFIGURACIONES SVM + DUMMY, "
        f"{n_trials} TRIALS"
    )

    (
        results_wei_rbf,
        results_mac_rbf,
        results_des_rbf,
        summary_rbf
    ) = run_multiclass_trials(
        X_dev=X_dev,
        y_dev=y_dev,
        n_trials=n_trials,
        n_jobs=n_jobs,
        test_size=0.2,
        kernel="rbf"
    )

    validate_experiment_results(
        results_wei_rbf,
        results_mac_rbf,
        results_des_rbf,
        summary_rbf,
        "RBF"
    )

    print("-> Experimento RBF finalizado.")

    print_summary_table(
        summary_rbf,
        "TABLA COMPARATIVA RBF"
    )

    csv_rbf = export_summary_to_csv(
        summary_rbf,
        filepath=(
            METRICS_DIR / "resultados_svm_normalizacion_rbf.csv"
        )
    )

    print(f"-> Resultados RBF exportados: {csv_rbf}")

    summary_rbf_svm = summary_rbf[
        summary_rbf["Normalization Method"] != DUMMY_NAME
    ].copy()

    result = plot_normalization_comparison(
        summary_rbf_svm,
        n_trials=n_trials,
        metric_col="Average F1 (Macro)",
        output_path=(
            FIGURES_DIR / "svm_normalizacion_vs_baseline_rbf.png"
        )
    )

    print_figure_result(
        result,
        "Comparación de normalización RBF"
    )

    # -------------------------------------------------------------------------
    # 7. DISTRIBUCIÓN DE RESULTADOS MONTE CARLO RBF
    # -------------------------------------------------------------------------

    print_section(
        "7. DISTRIBUCIÓN DE PUNTUACIONES F1 - RBF"
    )

    result = plot_trial_score_distributions(
        results_mac_rbf,
        output_path=(
            FIGURES_DIR / "svm_distribucion_trials_rbf.png"
        ),
        metric_name="F1 Macro RBF"
    )

    print_figure_result(
        result,
        "Distribución de puntuaciones Monte Carlo RBF"
    )

    # -------------------------------------------------------------------------
    # 8. SELECCIÓN DE LA MEJOR CONFIGURACIÓN RBF
    # -------------------------------------------------------------------------

    print_section(
        "8. SELECCIÓN DE LA MEJOR CONFIGURACIÓN RBF"
    )

    (
        best_name_rbf,
        best_scaler_rbf,
        best_ordinal_rbf
    ) = select_best_configuration(
        results_mac_rbf,
        scalers_dict
    )

    print(f"-> Mejor configuración RBF: {best_name_rbf}")

    # -------------------------------------------------------------------------
    # 9. EXPERIMENTO CON KERNEL POLINOMIAL GRADO 3
    # -------------------------------------------------------------------------

    print_section(
        f"9. EXPERIMENTO POLY-3: 8 CONFIGURACIONES SVM + DUMMY, "
        f"{n_trials} TRIALS"
    )

    (
        results_wei_poly,
        results_mac_poly,
        results_des_poly,
        summary_poly
    ) = run_multiclass_trials(
        X_dev=X_dev,
        y_dev=y_dev,
        n_trials=n_trials,
        n_jobs=n_jobs,
        test_size=0.2,
        kernel="poly",
        degree=3,
        coef0=1.0
    )

    validate_experiment_results(
        results_wei_poly,
        results_mac_poly,
        results_des_poly,
        summary_poly,
        "Poly-3"
    )

    print("-> Experimento Poly-3 finalizado.")

    print_summary_table(
        summary_poly,
        "TABLA COMPARATIVA POLY-3"
    )

    csv_poly = export_summary_to_csv(
        summary_poly,
        filepath=(
            METRICS_DIR / "resultados_svm_normalizacion_poly3.csv"
        )
    )

    print(f"-> Resultados Poly-3 exportados: {csv_poly}")

    summary_poly_svm = summary_poly[
        summary_poly["Normalization Method"] != DUMMY_NAME
    ].copy()

    result = plot_normalization_comparison(
        summary_poly_svm,
        n_trials=n_trials,
        metric_col="Average F1 (Macro)",
        output_path=(
            FIGURES_DIR / "svm_normalizacion_vs_baseline_poly3.png"
        )
    )

    print_figure_result(
        result,
        "Comparación de normalización Poly-3"
    )

    # -------------------------------------------------------------------------
    # 10. DISTRIBUCIÓN DE RESULTADOS MONTE CARLO POLY-3
    # -------------------------------------------------------------------------

    print_section(
        "10. DISTRIBUCIÓN DE PUNTUACIONES F1 - POLY-3"
    )

    result = plot_trial_score_distributions(
        results_mac_poly,
        output_path=(
            FIGURES_DIR / "svm_distribucion_trials_poly3.png"
        ),
        metric_name="F1 Macro Poly-3"
    )

    print_figure_result(
        result,
        "Distribución de puntuaciones Monte Carlo Poly-3"
    )

    # -------------------------------------------------------------------------
    # 11. SELECCIÓN DE LA MEJOR CONFIGURACIÓN POLY-3
    # -------------------------------------------------------------------------

    print_section(
        "11. SELECCIÓN DE LA MEJOR CONFIGURACIÓN POLY-3"
    )

    (
        best_name_poly,
        best_scaler_poly,
        best_ordinal_poly
    ) = select_best_configuration(
        results_mac_poly,
        scalers_dict
    )

    print(f"-> Mejor configuración Poly-3: {best_name_poly}")

    # -------------------------------------------------------------------------
    # 12. ENTRENAMIENTO DE LOS MODELOS FINALES
    # -------------------------------------------------------------------------

    print_section(
        "12. ENTRENAMIENTO FINAL CON EL CONJUNTO DE DESARROLLO"
    )

    print("Entrenando modelo SVM RBF...")

    best_pipeline = fit_best_pipeline(
        X_dev,
        y_dev,
        scaler=best_scaler_rbf,
        kernel="rbf",
        probability=False,
        is_ordinal=best_ordinal_rbf
    )

    print("Entrenando modelo SVM Poly-3...")

    best_pipeline_poly = fit_best_pipeline(
        X_dev,
        y_dev,
        scaler=best_scaler_poly,
        kernel="poly",
        degree=3,
        coef0=1.0,
        probability=False,
        is_ordinal=best_ordinal_poly
    )

    print("Entrenando DummyClassifier...")

    dummy_pipeline = fit_dummy_classifier(
        X_dev,
        y_dev,
        strategy="most_frequent"
    )

    print("-> Los tres modelos fueron entrenados.")

    # -------------------------------------------------------------------------
    # 13. SERIALIZACIÓN DE LOS MODELOS
    # -------------------------------------------------------------------------

    print_section(
        "13. SERIALIZACIÓN DE LOS MODELOS"
    )

    saved_model_path = save_model(
        best_pipeline,
        filepath=MODELS_DIR / "svm_pipeline_rbf.joblib"
    )

    saved_poly_path = save_model(
        best_pipeline_poly,
        filepath=MODELS_DIR / "svm_pipeline_poly3.joblib"
    )

    saved_dummy_path = save_model(
        dummy_pipeline,
        filepath=MODELS_DIR / "dummy_classifier.joblib"
    )

    print(f"-> Modelo RBF guardado: {saved_model_path}")
    print(f"-> Modelo Poly-3 guardado: {saved_poly_path}")
    print(f"-> Modelo Dummy guardado: {saved_dummy_path}")

    print(
        "-> Clase mayoritaria aprendida por DummyClassifier: "
        f"{dummy_pipeline.classes_[np.argmax(dummy_pipeline.class_prior_)]}"
    )

    # -------------------------------------------------------------------------
    # 14. EVALUACIÓN FINAL EN EL TEST RESERVADO
    # -------------------------------------------------------------------------

    print_section(
        f"14. EVALUACIÓN FINAL SOBRE TEST ({len(y_test)} ESTUDIANTES)"
    )

    y_pred_rbf = best_pipeline.predict(X_test)
    y_pred_poly = best_pipeline_poly.predict(X_test)
    y_pred_dummy = dummy_pipeline.predict(X_test)

    validate_prediction_classes(y_test, y_pred_rbf)
    validate_prediction_classes(y_test, y_pred_poly)
    validate_prediction_classes(y_test, y_pred_dummy)

    metrics_rbf = compute_classification_metrics(
        y_test,
        y_pred_rbf,
        target_names=CLASSES
    )

    metrics_poly = compute_classification_metrics(
        y_test,
        y_pred_poly,
        target_names=CLASSES
    )

    metrics_dummy = compute_classification_metrics(
        y_test,
        y_pred_dummy,
        target_names=CLASSES
    )

    print_final_metrics(
        metrics_rbf,
        "SVM RBF"
    )

    print_final_metrics(
        metrics_poly,
        "SVM POLY-3"
    )

    print_final_metrics(
        metrics_dummy,
        "DUMMY CLASSIFIER"
    )

    # -------------------------------------------------------------------------
    # 15. EXPORTACIÓN DE MÉTRICAS FINALES
    # -------------------------------------------------------------------------

    print_section(
        "15. EXPORTACIÓN DE MÉTRICAS FINALES"
    )

    final_metrics_df = pd.DataFrame([
        create_final_metrics_row(
            model_name="SVM RBF",
            configuration=best_name_rbf,
            metrics=metrics_rbf
        ),
        create_final_metrics_row(
            model_name="SVM Poly-3",
            configuration=best_name_poly,
            metrics=metrics_poly
        ),
        create_final_metrics_row(
            model_name=DUMMY_NAME,
            configuration="Most Frequent",
            metrics=metrics_dummy
        )
    ])

    final_metrics_path = save_dataframe(
        final_metrics_df,
        METRICS_DIR / "metricas_finales_test.csv"
    )

    print("\nCOMPARACIÓN FINAL EN TEST:")

    print(
        final_metrics_df.to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # 16. MATRICES DE CONFUSIÓN
    # -------------------------------------------------------------------------

    print_section("16. MATRICES DE CONFUSIÓN")

    result = plot_confusion_matrices(
        metrics_rbf["confusion_matrix"],
        classes=CLASSES,
        output_path=(
            FIGURES_DIR / "svm_matriz_confusion_rbf.png"
        )
    )

    print_figure_result(
        result,
        "Matriz de confusión RBF"
    )

    result = plot_confusion_matrices(
        metrics_poly["confusion_matrix"],
        classes=CLASSES,
        output_path=(
            FIGURES_DIR / "svm_matriz_confusion_poly3.png"
        )
    )

    print_figure_result(
        result,
        "Matriz de confusión Poly-3"
    )

    result = plot_confusion_matrices(
        metrics_dummy["confusion_matrix"],
        classes=CLASSES,
        output_path=(
            FIGURES_DIR / "dummy_matriz_confusion.png"
        )
    )

    print_figure_result(
        result,
        "Matriz de confusión DummyClassifier"
    )

    # -------------------------------------------------------------------------
    # 17. MÉTRICAS DE CLASIFICACIÓN POR CLASE
    # -------------------------------------------------------------------------

    print_section(
        "17. MÉTRICAS DE CLASIFICACIÓN POR CLASE"
    )

    result = plot_classification_metrics(
        y_test,
        y_pred_rbf,
        classes=CLASSES,
        output_path=(
            FIGURES_DIR / "svm_metricas_clasificacion_rbf.png"
        )
    )

    print_figure_result(
        result,
        "Métricas de clasificación por clase RBF"
    )

    result = plot_classification_metrics(
        y_test,
        y_pred_poly,
        classes=CLASSES,
        output_path=(
            FIGURES_DIR / "svm_metricas_clasificacion_poly3.png"
        )
    )

    print_figure_result(
        result,
        "Métricas de clasificación por clase Poly-3"
    )

    result = plot_classification_metrics(
        y_test,
        y_pred_dummy,
        classes=CLASSES,
        output_path=(
            FIGURES_DIR / "dummy_metricas_clasificacion.png"
        )
    )

    print_figure_result(
        result,
        "Métricas de clasificación por clase Dummy"
    )

    # -------------------------------------------------------------------------
    # 18. MÉTRICAS ESPECÍFICAS DE LA CLASE DESERTOR
    # -------------------------------------------------------------------------

    print_section(
        "18. EVALUACIÓN ESPECÍFICA DE LA CLASE DESERTOR"
    )

    result = plot_desertor_metrics(
        y_test,
        y_pred_rbf,
        output_path=(
            FIGURES_DIR / "svm_metricas_desertor_rbf.png"
        )
    )

    print_figure_result(
        result,
        "Métricas de la clase Desertor RBF"
    )

    result = plot_desertor_metrics(
        y_test,
        y_pred_poly,
        output_path=(
            FIGURES_DIR / "svm_metricas_desertor_poly3.png"
        )
    )

    print_figure_result(
        result,
        "Métricas de la clase Desertor Poly-3"
    )

    result = plot_desertor_metrics(
        y_test,
        y_pred_dummy,
        output_path=(
            FIGURES_DIR / "dummy_metricas_desertor.png"
        )
    )

    print_figure_result(
        result,
        "Métricas de la clase Desertor Dummy"
    )

    # -------------------------------------------------------------------------
    # 19. IMPORTANCIA DE VARIABLES POR PERMUTACIÓN
    # -------------------------------------------------------------------------

    print_section(
        "19. IMPORTANCIA DE VARIABLES POR PERMUTACIÓN"
    )

    print("Calculando importancia para RBF...")

    imp_rbf = compute_permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42
    )

    print("Calculando importancia para Poly-3...")

    imp_poly = compute_permutation_importance(
        best_pipeline_poly,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42
    )

    if imp_rbf is None or imp_rbf.empty:
        raise ValueError(
            "La importancia por permutación RBF no produjo resultados."
        )

    if imp_poly is None or imp_poly.empty:
        raise ValueError(
            "La importancia por permutación Poly-3 no produjo resultados."
        )

    print("\nTOP 15 VARIABLES - RBF:")

    print(imp_rbf.head(15).to_string(index=False))

    print("\nTOP 15 VARIABLES - POLY-3:")

    print(imp_poly.head(15).to_string(index=False))

    save_dataframe(
        imp_rbf,
        METRICS_DIR / "permutation_importance_rbf.csv"
    )

    save_dataframe(
        imp_poly,
        METRICS_DIR / "permutation_importance_poly3.csv"
    )

    result = plot_permutation_importance(
        imp_rbf,
        top_n=15,
        output_path=(
            FIGURES_DIR / "svm_importancia_variables_rbf.png"
        )
    )

    print_figure_result(
        result,
        "Importancia por permutación RBF"
    )

    result = plot_permutation_importance(
        imp_poly,
        top_n=15,
        output_path=(
            FIGURES_DIR / "svm_importancia_variables_poly3.png"
        )
    )

    print_figure_result(
        result,
        "Importancia por permutación Poly-3"
    )

    print(
        "\nNOTA: La importancia se calcula sobre el test "
        "como análisis explicativo posterior a la evaluación. "
        "No se utiliza para seleccionar ni reajustar los modelos."
    )

    print(
        "-> No se calcula permutation importance para DummyClassifier, "
        "porque su estrategia most_frequent no aprende relaciones "
        "entre las variables predictoras y la clase."
    )

    # -------------------------------------------------------------------------
    # 19.1. EXPLICABILIDAD SHAP, LIME Y PDP
    # -------------------------------------------------------------------------

    print_section(
        "19.1. EXPLICABILIDAD DE LOS MODELOS: SHAP, LIME Y PDP"
    )

    print(
        "-> Se utilizarán los modelos finales ya entrenados."
    )

    print(
        "-> El conjunto de desarrollo se utiliza como referencia "
        "y el conjunto de prueba como conjunto de explicación."
    )

    print(
        "-> No se realiza un nuevo entrenamiento ni se modifican "
        "las configuraciones seleccionadas."
    )

    print("\nEXPLICABILIDAD DEL MODELO RBF:")

    explanations_rbf = run_model_interpretability(
        model=best_pipeline,
        X_background=X_dev,
        X_explain=X_test,
        y_explain=y_test,
        model_name="RBF",
        numeric_features=NUM_COLS,
        figures_dir=FIGURES_DIR,
        metrics_dir=METRICS_DIR
    )

    print("\nEXPLICABILIDAD DEL MODELO POLY-3:")

    explanations_poly = run_model_interpretability(
        model=best_pipeline_poly,
        X_background=X_dev,
        X_explain=X_test,
        y_explain=y_test,
        model_name="Poly-3",
        numeric_features=NUM_COLS,
        figures_dir=FIGURES_DIR,
        metrics_dir=METRICS_DIR
    )

    print(
        "\n-> SHAP, LIME y PDP se calcularon para ambos "
        "modelos finales."
    )

    print(
        "-> Las explicaciones son descriptivas del comportamiento "
        "de los clasificadores sobre datos sintéticos."
    )

    # -------------------------------------------------------------------------
    # 20. VISUALIZACIÓN PCA
    # -------------------------------------------------------------------------

    print_section(
        "20. VISUALIZACIÓN DEL ESPACIO PCA"
    )

    result = plot_pca_hyperplane(
        best_pipeline,
        X_dev,
        y_dev,
        X_test,
        y_test,
        classes=CLASSES,
        output_path=(
            FIGURES_DIR / "svm_proyeccion_pca_rbf.png"
        ),
        model_name="SVM RBF"
    )

    print_figure_result(
        result,
        "Proyección PCA RBF"
    )

    result = plot_pca_hyperplane(
        best_pipeline_poly,
        X_dev,
        y_dev,
        X_test,
        y_test,
        classes=CLASSES,
        output_path=(
            FIGURES_DIR / "svm_proyeccion_pca_poly3.png"
        ),
        model_name="SVM Poly-3"
    )

    print_figure_result(
        result,
        "Proyección PCA Poly-3"
    )

    # -------------------------------------------------------------------------
    # 21. FRONTERAS DE DECISIÓN Y SCORES FUNCIONALES
    # -------------------------------------------------------------------------

    print_section(
        "21. VISUALIZACIONES DE DECISIÓN Y SCORES FUNCIONALES"
    )

    decision_df = X_dev.copy()

    decision_df["estado_final"] = np.asarray(y_dev)

    result = plot_decision_boundary_top2(
        decision_df,
        features=[
            "promedio_academico",
            "materias_reprobadas"
        ],
        sample_size=300,
        random_state=42,
        output_path=(
            FIGURES_DIR / "svm_frontera_binaria_exploratoria.png"
        )
    )

    print_figure_result(
        result,
        "Frontera binaria exploratoria"
    )

    result = plot_decision_function_distribution(
        decision_df,
        num_cols=NUM_COLS,
        random_state=42,
        output_path=(
            FIGURES_DIR / "svm_puntuaciones_decision_binaria.png"
        )
    )

    print_figure_result(
        result,
        "Distribución de puntuaciones de decisión"
    )

    print(
        "\n-> Las puntuaciones decision_function son scores "
        "funcionales del clasificador, no distancias geométricas "
        "directas."
    )

    print(
        "-> La frontera binaria corresponde a un modelo exploratorio "
        "independiente, no al modelo multiclase final."
    )

    # -------------------------------------------------------------------------
    # 22. AUDITORÍA DESCRIPTIVA POR SUBGRUPOS
    # -------------------------------------------------------------------------

    print_section(
        "22. AUDITORÍA DESCRIPTIVA POR SUBGRUPOS"
    )

    subgroup_rbf = compute_subgroup_recall(
        X_test,
        y_test,
        y_pred_rbf
    )

    subgroup_poly = compute_subgroup_recall(
        X_test,
        y_test,
        y_pred_poly
    )

    subgroup_dummy = compute_subgroup_recall(
        X_test,
        y_test,
        y_pred_dummy
    )

    print("\nRECALL DE DESERTOR POR SUBGRUPO - RBF:")

    if subgroup_rbf is not None and not subgroup_rbf.empty:
        print(subgroup_rbf.to_string(index=False))
    else:
        print("No se generaron resultados para RBF.")

    print("\nRECALL DE DESERTOR POR SUBGRUPO - POLY-3:")

    if subgroup_poly is not None and not subgroup_poly.empty:
        print(subgroup_poly.to_string(index=False))
    else:
        print("No se generaron resultados para Poly-3.")

    print("\nRECALL DE DESERTOR POR SUBGRUPO - DUMMY:")

    if subgroup_dummy is not None and not subgroup_dummy.empty:
        print(subgroup_dummy.to_string(index=False))
    else:
        print("No se generaron resultados para Dummy.")

    save_dataframe(
        subgroup_rbf,
        METRICS_DIR / "auditoria_subgrupos_rbf.csv"
    )

    save_dataframe(
        subgroup_poly,
        METRICS_DIR / "auditoria_subgrupos_poly3.csv"
    )

    save_dataframe(
        subgroup_dummy,
        METRICS_DIR / "auditoria_subgrupos_dummy.csv"
    )

    print(
        "\nNOTA: El análisis por subgrupos es descriptivo. "
        "No constituye por sí solo una evaluación completa "
        "de equidad algorítmica."
    )

    # -------------------------------------------------------------------------
    # 23. RESUMEN DE EJECUCIÓN
    # -------------------------------------------------------------------------

    print_section("23. EXPERIMENTO COMPLETADO")

    print("Configuración seleccionada RBF:")
    print(f"-> {best_name_rbf}")

    print("\nConfiguración seleccionada Poly-3:")
    print(f"-> {best_name_poly}")

    print("\nMÉTRICAS FINALES EN TEST:")

    print(
        f"RBF     | F1 Macro: {metrics_rbf['f1_macro']:.4f} "
        f"| F1 Weighted: {metrics_rbf['f1_weighted']:.4f} "
        f"| Recall Desertor: {metrics_rbf['recall_desertor']:.4f}"
    )

    print(
        f"Poly-3  | F1 Macro: {metrics_poly['f1_macro']:.4f} "
        f"| F1 Weighted: {metrics_poly['f1_weighted']:.4f} "
        f"| Recall Desertor: {metrics_poly['recall_desertor']:.4f}"
    )

    print(
        f"Dummy   | F1 Macro: {metrics_dummy['f1_macro']:.4f} "
        f"| F1 Weighted: {metrics_dummy['f1_weighted']:.4f} "
        f"| Recall Desertor: {metrics_dummy['recall_desertor']:.4f}"
    )

    print("\nARCHIVOS PRINCIPALES GENERADOS:")

    print(f"-> Métricas finales: {final_metrics_path}")
    print(f"-> Modelo RBF:       {saved_model_path}")
    print(f"-> Modelo Poly-3:    {saved_poly_path}")
    print(f"-> Modelo Dummy:     {saved_dummy_path}")

    print("\nDIRECTORIOS DE SALIDA:")

    print(f"-> Modelos:  {MODELS_DIR}")
    print(f"-> Métricas: {METRICS_DIR}")
    print(f"-> Figuras:  {FIGURES_DIR}")

    print("\nIMPORTANTE:")

    print(
        "Los resultados corresponden a un dataset sintético "
        "y no deben interpretarse como evidencia de desempeño "
        "sobre estudiantes reales."
    )

    print("=" * 80, flush=True)


# =============================================================================
# 24. PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Experimento SVM multiclase para predicción "
            "de deserción estudiantil."
        )
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Número de particiones Monte Carlo (default: 50)."
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Núcleos paralelos de Joblib (default: -1)."
    )

    args = parser.parse_args()

    if args.trials < 2:
        parser.error("--trials debe ser al menos 2.")

    if args.jobs == 0:
        parser.error("--jobs no puede ser 0.")

    try:

        main(
            n_trials=args.trials,
            n_jobs=args.jobs
        )

    except KeyboardInterrupt:

        print(
            "\n\nEjecución interrumpida por el usuario.",
            file=sys.stderr
        )

        sys.exit(130)

    except Exception as error:

        print(
            "\nERROR DURANTE LA EJECUCIÓN:",
            file=sys.stderr
        )

        print(
            f"{type(error).__name__}: {error}",
            file=sys.stderr
        )

        raise
