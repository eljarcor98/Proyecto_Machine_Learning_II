"""
Pipeline Orquestador del Experimento SVM:
Predicción de Deserción Estudiantil sobre Dataset Sintético (11,500 Estudiantes).
- Clasificación Multiclase SPADIES 3.0 (Graduado, En curso, Desertor)
- EDA Completo de Variables Socioeconómicas y Académicas
- Flujo Reproducible Sin Fuga de Datos (Scikit-Learn Pipeline + ColumnTransformer)
- Evaluación de 8 Normalizaciones vs Baseline sobre 50 particiones independientes
- Importancia de Variables en la Clasificación (Permutation Importance)
- Visualización del Hiperplano 2D (PCA) y Análisis de Desempeño por Subgrupo

Machine Learning II - 2026-II
"""

import argparse
import sys
import warnings
from pathlib import Path

# Asegurar que el directorio raíz del proyecto esté en el sys.path
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import (
    compute_outlier_diagnostics,
    create_spadies_target,
    get_train_test_split,
    load_data
)
from src.models import (
    compute_permutation_importance,
    fit_best_pipeline,
    run_multiclass_trials,
    save_model
)
from src.utils import (
    CLASSES,
    CLASS_MAP,
    DEFAULT_MODEL_FILE,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    NUM_COLS,
    compute_classification_metrics,
    compute_subgroup_recall,
    export_summary_to_csv
)
from src.visualization import (
    plot_confusion_matrices,
    plot_decision_boundary_top2,
    plot_decision_function_distribution,
    plot_eda_distributions,
    plot_key_variables_distribution,
    plot_normalization_comparison,
    plot_pca_hyperplane,
    plot_permutation_importance,
    plot_scaler_impact
)

warnings.filterwarnings('ignore')


def main(n_trials: int = 50, n_jobs: int = -1):
    """
    Función orquestadora principal del experimento de Máquinas de Soporte Vectorial.

    Ejecuta de manera secuencial y modular:
    1. Carga del conjunto de datos y etiquetado multiclase según SPADIES 3.0.
    2. Diagnóstico estadístico de valores atípicos (IQR) y asimetría (Skewness).
    3. Generación y guardado de gráficos exploratorios (EDA).
    4. Evaluación rigurosa sin fuga de datos de 8 métodos de normalización vs Baseline (N trials).
    5. Exportación de métricas consolidadas y contraste de hipótesis con t-test de Welch.
    6. Ajuste y serialización en disco del pipeline óptimo.
    7. Cálculo de relevancia de características por permutación (Permutation Importance).
    8. Visualización de fronteras de decisión y vectores de soporte en espacio latente PCA 2D.
    9. Evaluación en conjunto de prueba independiente (Matriz de confusión y métricas multiclase).
    10. Auditoría de sesgo algorítmico y equidad (Fairness Audit) desglosada por subgrupos.

    Parámetros:
    -----------
    n_trials : int, por defecto 50
        Número de repeticiones independientes de validación cruzada Monte Carlo.
    n_jobs : int, por defecto -1
        Cantidad de núcleos de procesamiento para aceleración paralela con joblib.
    """
    print("=" * 80)
    print("PROYECTO MACHINE LEARNING II: PREDICCIÓN DE DESERCIÓN CON SVM")
    print("Flujo Modular y Reproducible con Normalización Multiclase (SPADIES 3.0)")
    print("=" * 80, flush=True)

    # Crear directorios de salida
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. CARGA DE DATOS Y FORMULACIÓN DE OBJETIVO SPADIES 3.0
    # -------------------------------------------------------------
    print("\n1. CARGANDO Y PREPARANDO DATASET...")
    raw_df = load_data()
    print(f"-> Dimensiones iniciales: {raw_df.shape[0]} estudiantes, {raw_df.shape[1]} columnas.")

    df = create_spadies_target(raw_df)
    print(f"-> Valores nulos totales: {df.isnull().sum().sum()}")
    print("\nDistribución de la Variable Objetivo 'estado_final' (SPADIES):")
    print(df['estado_final'].value_counts())
    print((df['estado_final'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')

    # Diagnóstico de Outliers e IQR
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO ESTADÍSTICO DE VARIABLES CONTINUAS (OUTLIERS E IQR)")
    print("=" * 80)
    diag_df = compute_outlier_diagnostics(df, NUM_COLS)
    print(diag_df.to_string(index=False, formatters={
        'Media': '{:.2f}'.format,
        'Mediana': '{:.2f}'.format,
        'IQR': '{:.2f}'.format,
        'Outliers %': '{:.2f}%'.format,
        'Skewness': '{:.2f}'.format
    }))

    # -------------------------------------------------------------
    # 2. GENERACIÓN DE VISUALIZACIONES EDA
    # -------------------------------------------------------------
    print("\n2. GENERANDO VISUALIZACIONES EXPLORATORIAS (EDA)...")
    fig_eda, _ = plot_eda_distributions(df)
    print(f"-> [OK] Figura guardada: {FIGURES_DIR / 'eda_distribucion_correlaciones.png'}")

    fig_box, _ = plot_key_variables_distribution(df)
    print(f"-> [OK] Figura guardada: {FIGURES_DIR / 'eda_variables_clave.png'}")

    fig_scale, _ = plot_scaler_impact(df)
    print(f"-> [OK] Figura guardada: {FIGURES_DIR / 'eda_efecto_escaladores.png'}")

    # -------------------------------------------------------------
    # 3. EXPERIMENTO DE PIPELINES MULTICLASE (50 TRIALS)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"3. EVALUACIÓN DE 8 NORMALIZACIONES VS BASELINE ({n_trials} TRIALS SIN FUGA DE DATOS)")
    print("=" * 80, flush=True)

    X_train, X_test, y_train, y_test, X_raw, y = get_train_test_split(df, test_size=0.2, random_state=42)

    print(f"Iniciando {n_trials} particiones independientes con paralelización (n_jobs={n_jobs})...", flush=True)
    results_wei, results_mac, results_des, summary_df = run_multiclass_trials(
        X_raw=X_raw,
        y=y,
        n_trials=n_trials,
        n_jobs=n_jobs,
        test_size=0.2
    )
    print("-> Iteraciones finalizadas con éxito.")

    # Guardar métricas en CSV
    csv_path = export_summary_to_csv(summary_df)
    print(f"-> [OK] Resumen estadístico exportado a: {csv_path}")

    print("\nTABLA COMPARATIVA: PROMEDIO DE F1-SCORE MULTICLASE:")
    print(summary_df.to_string(index=False, formatters={
        'Average F1 (Weighted)': '{:.4f}'.format,
        'Average F1 (Macro)': '{:.4f}'.format,
        'Average F1 (Desertor)': '{:.4f}'.format,
        'Std Dev': '{:.4f}'.format,
        'F1 Delta vs Baseline': '{:+.4f}'.format,
        'T-Statistic': '{:.3f}'.format,
        'P-value vs Baseline': '{:.6f}'.format
    }))

    fig_norm, _ = plot_normalization_comparison(summary_df, n_trials=n_trials)
    print(f"-> [OK] Figura guardada: {FIGURES_DIR / 'svm_normalizacion_vs_baseline.png'}")

    # -------------------------------------------------------------
    # 4. AJUSTE DEL MODELO FINAL, SERIALIZACIÓN Y PERMUTATION IMPORTANCE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. AJUSTE DEL PIPELINE FINAL, SERIALIZACIÓN E IMPORTANCIA DE VARIABLES")
    print("=" * 80, flush=True)

    best_pipeline = fit_best_pipeline(X_train, y_train, scaler=StandardScaler(), probability=False)
    saved_model_path = save_model(best_pipeline)
    print(f"-> [OK] Pipeline serializado guardado en: {saved_model_path}")

    print("Calculando Permutation Importance sobre el conjunto de test...")
    imp_df = compute_permutation_importance(best_pipeline, X_test, y_test, n_repeats=5, random_state=42)
    print("\nTop 10 Variables más influyentes para el Pipeline SVM:")
    print(imp_df.head(10).to_string(index=False))

    fig_imp, _ = plot_permutation_importance(imp_df, top_n=15)
    print(f"-> [OK] Figura guardada: {FIGURES_DIR / 'svm_importancia_variables.png'}")

    # -------------------------------------------------------------
    # 5. VISUALIZACIÓN DEL HIPERPLANO 2D (PCA) Y EN VARIABLES REALES
    # -------------------------------------------------------------
    print("\n5. VISUALIZANDO HIPERPLANO MULTICLASE EN ESPACIO LATENTE (PCA 2D)...")
    fig_pca, _ = plot_pca_hyperplane(best_pipeline, X_train, y_train, X_test, y_test)
    print(f"-> [OK] Figura guardada: {FIGURES_DIR / 'svm_hiperplano_pca_2d.png'}")

    print("-> Generando visualización nítida del hiperplano en variables reales clave...")
    plot_decision_boundary_top2(df)
    print(f"-> [OK] Figura guardada: {FIGURES_DIR / 'svm_hiperplano_top2_variables.png'}")

    print("-> Generando distribución de distancia funcional f(x) y margen de vectores de soporte...")
    plot_decision_function_distribution(df)
    print(f"-> [OK] Figura guardada: {FIGURES_DIR / 'svm_distancia_hiperplano_estudiantes.png'}")

    # -------------------------------------------------------------
    # 6. EVALUACIÓN DETALLADA Y AUDITORÍA DE EQUIDAD POR SUBGRUPO
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"6. EVALUACIÓN EN CONJUNTO DE PRUEBA (TEST SET = {len(y_test)} ESTUDIANTES)")
    print("=" * 80, flush=True)

    y_pred = best_pipeline.predict(X_test)
    metrics = compute_classification_metrics(y_test, y_pred, target_names=CLASSES)

    print("\nREPORTE DE CLASIFICACIÓN DETALLADO:")
    print(metrics['classification_report'])
    print(f"Accuracy Total:      {metrics['accuracy']:.4f}")
    print(f"F1-Score Ponderado: {metrics['f1_weighted']:.4f}")
    print(f"F1-Score Macro:     {metrics['f1_macro']:.4f}")
    print(f"Recall (Desertor):  {metrics['recall_desertor']:.4f}")

    fig_cm, _ = plot_confusion_matrices(metrics['confusion_matrix'])
    print(f"-> [OK] Figura guardada: {FIGURES_DIR / 'svm_matriz_confusion_roc.png'}")

    print("\nDESGLOSE DE RECALL (DESERTOR) POR SUBGRUPO (AUDITORÍA DE SESGO):")
    subgroup_df = compute_subgroup_recall(X_test, y_test, y_pred)
    print(subgroup_df.to_string(index=False, formatters={'Recall (Desertor)': '{:.4f}'.format}))

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO EXITOSAMENTE CON ARQUITECTURA MODULAR.")
    print(f"Modelos serializados en:  {MODELS_DIR}")
    print(f"Métricas exportadas en:   {METRICS_DIR}")
    print(f"Figuras guardadas en:     {FIGURES_DIR}")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline Modular SVM: Predicción de Deserción Estudiantil SPADIES 3.0")
    parser.add_argument('--trials', type=int, default=50, help='Número de iteraciones independientes (default: 50)')
    parser.add_argument('--jobs', type=int, default=-1, help='Número de núcleos paralelos para joblib (default: -1)')
    args = parser.parse_args()

    main(n_trials=args.trials, n_jobs=args.jobs)
