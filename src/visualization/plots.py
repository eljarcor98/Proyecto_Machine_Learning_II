"""
Módulo de Visualización Científica del Experimento SVM.
======================================================
Provee funciones especializadas de graficación para la generación de evidencias empíricas
y figuras de publicación en alta resolución (300 DPI):

Figuras Soportadas:
-------------------
1. plot_eda_distributions: Distribución SPADIES 3 clases + Correlaciones Pearson vs Spearman.
2. plot_key_variables_distribution: Boxplots de variables clave (Promedio y Reprobadas) por estado.
3. plot_scaler_impact: Demostración empírica de 6 transformaciones sobre variable sesgada.
4. plot_normalization_comparison: Gráfico de barras comparativo de normalizaciones vs baseline.
5. plot_permutation_importance: Relevancia de características por permutación con barras de error.
6. plot_pca_hyperplane: Fronteras de decisión RBF y vectores de soporte en espacio latente PCA 2D.
7. plot_confusion_matrices: Matriz de confusión multiclase 3x3 (conteos absolutos y normalizada).

Autor: Arnold Santiago Torres
Machine Learning II - 2026-II
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler
)
from sklearn.svm import SVC

from src.utils.config import (
    CLASS_PALETTE,
    CLASSES,
    FIGURES_DIR,
    NUM_COLS,
    SCATTER_PALETTE
)

# Configuración estética base para gráficos científicos
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'Arial'


def plot_eda_distributions(
    df: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, Any]:
    """
    Figura 1: Genera un panel de 3 gráficos para el análisis exploratorio:
    - Subgráfico 1: Conteo absoluto y porcentual de las 3 clases SPADIES.
    - Subgráfico 2: Correlación lineal de Pearson de variables numéricas con la deserción.
    - Subgráfico 3: Correlación monótona de Spearman con la deserción.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con las columnas del dataset y la variable objetivo 'estado_final'.
    num_cols : list of str, opcional
        Lista de variables continuas y discretas a evaluar. Por defecto NUM_COLS.
    classes : list of str, opcional
        Nombres y orden de las clases a graficar. Por defecto CLASSES.
    output_path : str o Path, opcional
        Ruta para guardar la imagen generada (300 DPI). Si es None, usa FIGURES_DIR.

    Retorna:
    --------
    tuple:
        (fig, axes) objetos de Matplotlib.
    """
    if num_cols is None:
        num_cols = NUM_COLS
    if classes is None:
        classes = CLASSES

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Subgráfico 1: Distribución muestral de las clases ---
    sns.countplot(
        data=df,
        x='estado_final',
        order=classes,
        hue='estado_final',
        palette='Set2',
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title('Distribución de Estudiantes (SPADIES 3.0)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Estado Académico Final', fontsize=11)
    axes[0].set_ylabel('Número de Estudiantes', fontsize=11)

    # Ciclo para rotular cada barra con su conteo exacto y porcentaje relativo
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(
            f'{int(height)} ({height/len(df)*100:.1f}%)',
            (p.get_x() + p.get_width() / 2.0, height / 2.0),
            ha='center', va='center', fontsize=10, color='white', fontweight='bold'
        )

    # Crear indicador binario focal para correlaciones (1: Desertor, 0: Otro)
    des_binary = (df['estado_final'] == 'Desertor').astype(int)

    # --- Subgráfico 2: Correlación de Pearson ---
    corr_p = df[num_cols].apply(lambda col: col.corr(des_binary, method='pearson')).sort_values()
    colors_p = ['#d95f02' if c > 0 else '#7570b3' for c in corr_p.values]
    axes[1].barh(corr_p.index, corr_p.values, color=colors_p)
    axes[1].set_title('Correlación de Pearson con Deserción', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Coeficiente Pearson', fontsize=11)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)

    # --- Subgráfico 3: Correlación de Spearman ---
    corr_s = df[num_cols].apply(lambda col: col.corr(des_binary, method='spearman')).sort_values()
    colors_s = ['#1b9e77' if c > 0 else '#e7298a' for c in corr_s.values]
    axes[2].barh(corr_s.index, corr_s.values, color=colors_s)
    axes[2].set_title('Correlación de Spearman (Monótona)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Coeficiente Spearman', fontsize=11)
    axes[2].axvline(0, color='black', linestyle='--', linewidth=0.8)

    plt.tight_layout()

    # Guardar en disco si corresponde
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "eda_distribucion_correlaciones.png"

    plt.savefig(output_path, dpi=300)
    return fig, axes


def plot_key_variables_distribution(
    df: pd.DataFrame,
    classes: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, Any]:
    """
    Figura 2: Diagramas de caja (Boxplots) que comparan la dispersión de las dos variables
    predictoras de mayor peso académico (Promedio Acumulado y Asignaturas Reprobadas) entre estados.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos preprocesados.
    classes : list of str, opcional
        Lista ordenada de clases. Por defecto CLASSES.
    output_path : str o Path, opcional
        Ruta para persistir la imagen en disco.

    Retorna:
    --------
    tuple:
        (fig, axes) objetos de Matplotlib.
    """
    if classes is None:
        classes = CLASSES

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Boxplot para Promedio Académico
    sns.boxplot(
        data=df,
        x='estado_final',
        y='promedio_academico',
        order=classes,
        hue='estado_final',
        palette='Set1',
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title('Promedio Académico por Estado Académico', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Estado del Estudiante', fontsize=12)
    axes[0].set_ylabel('Promedio Académico (0.0 - 5.0)', fontsize=12)

    # Boxplot para Materias Reprobadas
    sns.boxplot(
        data=df,
        x='estado_final',
        y='materias_reprobadas',
        order=classes,
        hue='estado_final',
        palette='Set1',
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title('Materias Reprobadas por Estado Académico', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Estado del Estudiante', fontsize=12)
    axes[1].set_ylabel('Materias Reprobadas', fontsize=12)

    plt.tight_layout()

    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "eda_variables_clave.png"

    plt.savefig(output_path, dpi=300)
    return fig, axes


def plot_scaler_impact(
    df: pd.DataFrame,
    feature: str = 'distancia_hogar_ies_km',
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, Any]:
    """
    Figura 3: Evidencia visual empírica del efecto de 6 técnicas de escalamiento sobre
    la forma de la densidad (KDE) y distribución de una variable con marcada asimetría positiva.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con la variable a analizar.
    feature : str, por defecto 'distancia_hogar_ies_km'
        Nombre de la columna numérica continua.
    output_path : str o Path, opcional
        Ruta de destino para la imagen.

    Retorna:
    --------
    tuple:
        (fig, axes) objetos de Matplotlib.
    """
    sample_feature = df[[feature]].values
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # 1. Escala Original sin transformar
    sns.histplot(sample_feature, kde=True, ax=axes[0, 0], color='gray')
    axes[0, 0].set_title('1. Original (Raw)', fontweight='bold')

    # 2. StandardScaler (Centrado en media 0, desviación unitaria)
    sns.histplot(StandardScaler().fit_transform(sample_feature), kde=True, ax=axes[0, 1], color='blue')
    axes[0, 1].set_title('2. Standard Scaler (Z-score)', fontweight='bold')

    # 3. MinMaxScaler (Acotamiento estricto a [0, 1])
    sns.histplot(MinMaxScaler().fit_transform(sample_feature), kde=True, ax=axes[0, 2], color='green')
    axes[0, 2].set_title('3. Min Max Scaler (0 a 1)', fontweight='bold')

    # 4. RobustScaler (Basado en mediana y rango intercuartílico IQR)
    sns.histplot(RobustScaler().fit_transform(sample_feature), kde=True, ax=axes[1, 0], color='purple')
    axes[1, 0].set_title('4. Robust Scaler (Mediana e IQR)', fontweight='bold')

    # 5. QuantileTransformer (Mapeo a distribución uniforme)
    sns.histplot(
        QuantileTransformer(n_quantiles=50, random_state=42).fit_transform(sample_feature),
        kde=True, ax=axes[1, 1], color='orange'
    )
    axes[1, 1].set_title('5. Quantile Transformer (Uniforme)', fontweight='bold')

    # 6. PowerTransformer (Transformación de potencia Yeo-Johnson estabilizadora)
    sns.histplot(PowerTransformer().fit_transform(sample_feature), kde=True, ax=axes[1, 2], color='red')
    axes[1, 2].set_title('6. Power Transformer (Yeo-Johnson)', fontweight='bold')

    plt.tight_layout()

    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "eda_efecto_escaladores.png"

    plt.savefig(output_path, dpi=300)
    return fig, axes


def plot_normalization_comparison(
    summary_df: pd.DataFrame,
    n_trials: int = 50,
    metric_col: str = 'Average F1 (Weighted)',
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, Any]:
    """
    Figura 4: Diagrama de barras comparativo de rendimiento multiclase que contrasta
    los escaladores evaluados frente al modelo Baseline sin normalizar.

    Parámetros:
    -----------
    summary_df : pd.DataFrame
        Tabla resumen con las métricas consolidadas de las repeticiones.
    n_trials : int, por defecto 50
        Número de iteraciones representadas para indicar en el título.
    metric_col : str, por defecto 'Average F1 (Weighted)'
        Columna numérica del DataFrame utilizada como eje X.
    output_path : str o Path, opcional
        Ruta para guardar la figura.

    Retorna:
    --------
    tuple:
        (fig, ax) objetos de Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Colores semánticos: naranja para el baseline y verde azulado para los métodos evaluados
    colors_bar = ['#d95f02' if 'Baseline' in name else '#1b9e77' for name in summary_df['Normalization Method']]

    barplot = sns.barplot(
        data=summary_df,
        x=metric_col,
        y='Normalization Method',
        hue='Normalization Method',
        palette=colors_bar,
        legend=False,
        ax=ax
    )

    ax.set_title(
        f'Evaluación de Pipelines SVM Multiclase ({metric_col}, {n_trials} Trials)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('F1-Score Ponderado Promedio (3 Clases)', fontsize=12)
    ax.set_ylabel('Método de Normalización en Pipeline', fontsize=12)
    ax.set_xlim(0.40, 1.0)

    # Ciclo para etiquetar el valor numérico en el extremo de cada barra
    for p in barplot.patches:
        width = p.get_width()
        barplot.annotate(
            f'{width:.4f}',
            (width + 0.005, p.get_y() + p.get_height() / 2.0),
            ha='left', va='center', fontsize=10, fontweight='bold'
        )

    plt.tight_layout()

    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "svm_normalizacion_vs_baseline.png"

    plt.savefig(output_path, dpi=300)
    return fig, ax


def plot_permutation_importance(
    imp_df: pd.DataFrame,
    top_n: int = 15,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, Any]:
    """
    Figura 5: Gráfico horizontal de barras de error con la Importancia de Variables por Permutación.

    Parámetros:
    -----------
    imp_df : pd.DataFrame
        DataFrame con las columnas 'Feature', 'Importance_Mean' e 'Importance_Std'.
    top_n : int, por defecto 15
        Cantidad de variables más influyentes a graficar.
    output_path : str o Path, opcional
        Ruta destino de la figura.

    Retorna:
    --------
    tuple:
        (fig, ax) objetos de Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    top_df = imp_df.head(top_n)

    # Graficar en orden inverso para que la variable más importante aparezca en la parte superior
    ax.barh(
        top_df['Feature'][::-1],
        top_df['Importance_Mean'][::-1],
        color='#2b5c8f',
        xerr=top_df['Importance_Std'][::-1]
    )

    ax.set_title(
        'Importancia de Variables en Pipeline SVM (Permutation Importance Multiclase)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Disminución Promedio en F1-Score Ponderado al Permutar Variable', fontsize=12)
    plt.tight_layout()

    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "svm_importancia_variables.png"

    plt.savefig(output_path, dpi=300)
    return fig, ax


def plot_pca_hyperplane(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    classes: Optional[List[str]] = None,
    sample_size: int = 1000,
    random_state: int = 42,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, Any]:
    """
    Figura 6: Visualización de las fronteras de decisión y los vectores de soporte
    del modelo SVM proyectados sobre las dos componentes principales (PCA 2D).

    Parámetros:
    -----------
    pipeline : Pipeline
        Pipeline SVM ajustado que contiene el paso 'preprocessor'.
    X_train : pd.DataFrame
        Características de entrenamiento para ajustar el proyector PCA.
    y_train : np.ndarray
        Etiquetas de clase de entrenamiento.
    X_test : pd.DataFrame
        Características de prueba para visualización de dispersión.
    y_test : np.ndarray
        Etiquetas de clase de prueba.
    classes : list of str, opcional
        Lista ordenada de clases. Por defecto CLASSES.
    sample_size : int, por defecto 1000
        Número de puntos de prueba a graficar para evitar sobrecarga visual.
    random_state : int, por defecto 42
        Semilla para el muestreo y PCA.
    output_path : str o Path, opcional
        Ruta para guardar la imagen.

    Retorna:
    --------
    tuple:
        (fig, ax) objetos de Matplotlib.
    """
    if classes is None:
        classes = CLASSES

    # 1. Transformar datos mediante el preprocesador del pipeline ajustado
    X_train_trans = pipeline.named_steps['preprocessor'].transform(X_train)
    X_test_trans = pipeline.named_steps['preprocessor'].transform(X_test)

    # 2. Reducir dimensionalidad a 2 dimensiones con Análisis de Componentes Principales
    pca = PCA(n_components=2, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_trans)
    X_test_pca = pca.transform(X_test_trans)

    # 3. Ajustar un clasificador SVM bidimensional para mapear las regiones de decisión continuas
    svm_pca = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=random_state)
    svm_pca.fit(X_train_pca, y_train)

    # 4. Crear rejilla de evaluación (Meshgrid) sobre el espacio 2D
    x_min, x_max = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
    y_min, y_max = X_train_pca[:, 1].min() - 0.5, X_train_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))

    # Predecir sobre cada punto de la malla para delimitar las regiones de las clases
    Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(11, 8))

    # Colorear regiones de decisión con contornos suaves
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='Set2')

    # Submuestrear puntos de prueba para representación limpia
    np.random.seed(random_state)
    sample_idx = np.random.choice(len(y_test), size=min(sample_size, len(y_test)), replace=False)
    X_sample_pca = X_test_pca[sample_idx]
    y_sample = y_test[sample_idx]

    # Ciclo para dispersar los estudiantes de cada clase con su respectivo color
    for idx_c, label_c in enumerate(classes):
        ax.scatter(
            X_sample_pca[y_sample == idx_c, 0],
            X_sample_pca[y_sample == idx_c, 1],
            c=SCATTER_PALETTE[idx_c],
            label=label_c,
            alpha=0.6,
            edgecolors='k',
            s=30
        )

    # Resaltar los vectores de soporte (Support Vectors) seleccionados por el algoritmo
    sv = svm_pca.support_vectors_
    ax.scatter(
        sv[:, 0],
        sv[:, 1],
        s=80,
        facecolors='none',
        edgecolors='black',
        linewidths=1.0,
        label=f'Vectores de Soporte ({len(sv)} puntos)'
    )

    ax.set_title('Fronteras de Decisión SVM Multiclase (Kernel RBF en PCA 2D)', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Varianza)', fontsize=12)
    ax.set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Varianza)', fontsize=12)
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()

    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "svm_hiperplano_pca_2d.png"

    plt.savefig(output_path, dpi=300)
    return fig, ax


def plot_confusion_matrices(
    cm: np.ndarray,
    classes: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, Any]:
    """
    Figura 7: Genera un panel dual con la matriz de confusión multiclase 3x3:
    - Izquierda: Conteos absolutos de clasificaciones correctas e incorrectas.
    - Derecha: Matriz normalizada por fila (Sensibilidad o Recall por clase real).

    Parámetros:
    -----------
    cm : np.ndarray
        Matriz de confusión cruda calculada con sklearn.metrics.confusion_matrix.
    classes : list of str, opcional
        Nombres cualitativos de las clases. Por defecto CLASSES.
    output_path : str o Path, opcional
        Ruta para guardar la imagen generada.

    Retorna:
    --------
    tuple:
        (fig, axes) objetos de Matplotlib.
    """
    if classes is None:
        classes = CLASSES

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Subgráfico 1: Conteos Absolutos ---
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
        xticklabels=classes, yticklabels=classes, annot_kws={'size': 13, 'weight': 'bold'}
    )
    axes[0].set_title('Matriz de Confusión (Conteos Absolutos)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[0].set_ylabel('Estado Real', fontsize=12)

    # --- Subgráfico 2: Normalizada por fila (Recall por clase) ---
    # Dividir cada fila entre la suma de sus elementos para obtener probabilidades condicionales
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=axes[1],
        xticklabels=classes, yticklabels=classes, annot_kws={'size': 13, 'weight': 'bold'}
    )
    axes[1].set_title('Matriz de Confusión Normalizada por Clase Real (Recall)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicción del Modelo', fontsize=12)
    axes[1].set_ylabel('Estado Real', fontsize=12)

    plt.tight_layout()

    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "svm_matriz_confusion_roc.png"

    plt.savefig(output_path, dpi=300)
    return fig, axes


def plot_decision_boundary_top2(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    sample_size: int = 300,
    random_state: int = 42
) -> Tuple[plt.Figure, Any]:
    """
    Genera la visualización cristalina del hiperplano de decisión de SVM y márgenes
    sobre las dos variables predictoras de mayor relevancia (Promedio Académico y Materias Reprobadas).
    Permite interpretar intuitivamente la separación de clases sin la abstracción del PCA.
    """
    if features is None:
        features = ['promedio_academico', 'materias_reprobadas']

    X = df[features].values
    y_bin = (df['estado_final'] == 'Desertor').astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = SVC(kernel='rbf', C=1.0, random_state=random_state)
    svm.fit(X_scaled, y_bin)

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = svm.decision_function(grid_scaled).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Regiones sombreadas suaves
    ax.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=['#e0f3f8', '#fee0d2'], alpha=0.55)

    # Contornos de decisión y márgenes
    contours = ax.contour(
        xx, yy, Z, levels=[-1.0, 0.0, 1.0],
        linestyles=['--', '-', '--'],
        colors=['#2b83ba', '#d7191c', '#2b83ba'],
        linewidths=[1.8, 2.8, 1.8]
    )
    ax.clabel(
        contours, inline=True, fontsize=10,
        fmt={-1.0: 'Margen (-1)', 0.0: 'Hiperplano Separador f(x)=0', 1.0: 'Margen (+1)'}
    )

    # Submuestra limpia de estudiantes
    np.random.seed(random_state)
    sample_idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sub = X[sample_idx]
    y_sub = y_bin[sample_idx]

    ax.scatter(
        X_sub[y_sub == 0, 0], X_sub[y_sub == 0, 1],
        c='#2b83ba', label='Persistente (Graduado / En curso)',
        alpha=0.7, edgecolors='k', s=45
    )
    ax.scatter(
        X_sub[y_sub == 1, 0], X_sub[y_sub == 1, 1],
        c='#d7191c', label='Desertor',
        alpha=0.7, edgecolors='k', s=45
    )

    # Vectores de soporte en el espacio original de variables
    sv_orig = scaler.inverse_transform(svm.support_vectors_)
    sv_sample = sv_orig[np.random.choice(len(sv_orig), size=min(60, len(sv_orig)), replace=False)]
    ax.scatter(
        sv_sample[:, 0], sv_sample[:, 1],
        s=130, facecolors='none', edgecolors='black', linewidths=2.0,
        label=f'Vectores de Soporte (Muestra de {len(svm.support_vectors_)} puntos límite)'
    )

    ax.set_title('Frontera de Decisión e Hiperplano SVM en Variables Críticas', fontsize=14, fontweight='bold')
    ax.set_xlabel('Promedio Académico Acumulado (0.0 - 5.0)', fontsize=12)
    ax.set_ylabel('Materias Reprobadas (0 - 8)', fontsize=12)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92)

    plt.tight_layout()

    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "svm_hiperplano_top2_variables.png"

    plt.savefig(output_path, dpi=300)
    return fig, ax


def plot_decision_function_distribution(
    df: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    random_state: int = 42
) -> Tuple[plt.Figure, Any]:
    """
    Figura de interpretabilidad 1D: Distribución de distancias funcionales al hiperplano f(x)
    y visualización de los márgenes y vectores de soporte.
    """
    if num_cols is None:
        num_cols = ['promedio_academico', 'materias_reprobadas', 'semestre_cursado', 'distancia_hogar_ies_km', 'estrato_socioeconomico']

    X = df[num_cols].values
    y_bin = (df['estado_final'] == 'Desertor').astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = SVC(kernel='rbf', C=1.0, random_state=random_state)
    svm.fit(X_scaled, y_bin)

    dfunc = svm.decision_function(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(dfunc[y_bin == 0], color='#2b83ba', kde=True, label='Estudiantes Persistentes', stat='density', alpha=0.5, ax=ax)
    sns.histplot(dfunc[y_bin == 1], color='#d7191c', kde=True, label='Estudiantes Desertores', stat='density', alpha=0.5, ax=ax)

    ax.axvline(0, color='black', linestyle='-', linewidth=2.5, label='Hiperplano Separador f(x) = 0')
    ax.axvline(1, color='#2b83ba', linestyle='--', linewidth=1.8, label='Margen (+1)')
    ax.axvline(-1, color='#d7191c', linestyle='--', linewidth=1.8, label='Margen (-1)')
    ax.axvspan(-1, 1, color='gray', alpha=0.15, label='Zona del Margen y Vectores de Soporte [-1, +1]')

    ax.set_title('Distribución de Distancias Funcionales de Estudiantes al Hiperplano SVM', fontsize=14, fontweight='bold')
    ax.set_xlabel('Distancia Funcional f(x) = wᵀφ(x) + b (Margen Geométrico)', fontsize=12)
    ax.set_ylabel('Densidad de Probabilidad de Estudiantes', fontsize=12)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95)

    plt.tight_layout()

    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "svm_distancia_hiperplano_estudiantes.png"

    plt.savefig(output_path, dpi=300)
    return fig, ax
