
"""
Módulo de Visualización Científica del Experimento SVM.
======================================================

Genera figuras exploratorias y de evaluación para el experimento
de clasificación multiclase del estado académico estudiantil.

Proyecto:
Machine Learning II - 2026-II

Clases:
- Graduado
- En curso
- Desertor

Figuras:
1.  plot_eda_distributions
2.  plot_key_variables_distribution
3.  plot_scaler_impact
4.  plot_normalization_comparison
5.  plot_permutation_importance
6.  plot_pca_hyperplane
7.  plot_confusion_matrices
8.  plot_decision_boundary_top2
9.  plot_decision_function_distribution
10. plot_classification_metrics
11. plot_trial_score_distributions
12. plot_desertor_metrics

Consideraciones:
- El conjunto de datos es sintético e inspirado en el contexto SPADIES.
- Las visualizaciones descriptivas no demuestran causalidad.
- Los gráficos binarios son experimentos exploratorios independientes.
- PCA es una proyección visual, no una frontera real del SVM.
- Las puntuaciones de decisión no son distancias geométricas.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler,
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


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

sns.set_theme(
    style="whitegrid",
    font_scale=1.0
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def _prepare_output_path(
    output_path: Optional[Union[str, Path]],
    filename: str
) -> Path:
    """
    Prepara la ruta de salida y crea los directorios necesarios.

    Si output_path es None, utiliza FIGURES_DIR.
    """

    if output_path is None:
        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )
        path = FIGURES_DIR / filename

    else:
        path = Path(output_path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    return path


def _save_figure(
    fig: plt.Figure,
    output_path: Optional[Union[str, Path]],
    filename: str
) -> Path:
    """
    Guarda una figura en alta resolución.

    Devuelve la ruta donde fue almacenada.
    No cierra la figura para permitir modificaciones posteriores.
    """

    path = _prepare_output_path(
        output_path,
        filename
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none"
    )

    return path


def _validate_dataframe(
    df: pd.DataFrame,
    required_columns: Sequence[str]
) -> None:
    """
    Valida el tipo de objeto y la presencia de columnas.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Se esperaba un objeto pandas.DataFrame."
        )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}"
        )


def _validate_target(
    df: pd.DataFrame,
    classes: Optional[List[str]] = None
) -> List[str]:
    """
    Valida la variable objetivo textual estado_final.
    """

    _validate_dataframe(
        df,
        ["estado_final"]
    )

    if classes is None:
        classes = list(CLASSES)

    observed = set(
        df["estado_final"].dropna().unique()
    )

    unknown = observed - set(classes)

    if unknown:
        raise ValueError(
            f"Se encontraron clases no reconocidas: {unknown}"
        )

    if len(observed) < 2:
        raise ValueError(
            "Se necesitan al menos dos clases observadas."
        )

    return list(classes)


def _get_class_color(
    label: str,
    index: int = 0
) -> str:
    """
    Obtiene el color asociado a una clase.
    """

    if isinstance(CLASS_PALETTE, dict):
        if label in CLASS_PALETTE:
            return CLASS_PALETTE[label]

    if SCATTER_PALETTE:
        return SCATTER_PALETTE[
            index % len(SCATTER_PALETTE)
        ]

    return "steelblue"


def _to_dense(X: Any) -> np.ndarray:
    """
    Convierte matrices dispersas en matrices densas.
    """

    if hasattr(X, "toarray"):
        X = X.toarray()

    return np.asarray(X)


def _validate_numeric_columns(
    df: pd.DataFrame,
    columns: Sequence[str]
) -> List[str]:
    """
    Conserva columnas numéricas disponibles y no constantes.
    """

    valid = []

    for col in columns:

        if col not in df.columns:
            continue

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if series.notna().sum() == 0:
            continue

        if series.nunique(dropna=True) <= 1:
            continue

        valid.append(col)

    if not valid:
        raise ValueError(
            "No existen variables numéricas válidas."
        )

    return valid


def _validate_sample_size(
    sample_size: int,
    minimum: int = 2
) -> None:
    """
    Valida el tamaño de muestra.
    """

    if not isinstance(sample_size, int) or sample_size < minimum:
        raise ValueError(
            f"sample_size debe ser un entero >= {minimum}."
        )


def _get_svm_from_pipeline(pipeline) -> SVC:
    """
    Localiza el estimador SVC dentro del pipeline entrenado.
    """

    if not hasattr(pipeline, "named_steps"):
        raise TypeError(
            "Se esperaba un sklearn Pipeline entrenado."
        )

    svm_models = [
        step
        for step in pipeline.named_steps.values()
        if isinstance(step, SVC)
    ]

    if not svm_models:
        raise ValueError(
            "No se encontró un estimador SVC dentro del pipeline."
        )

    svm_model = svm_models[-1]

    if not hasattr(svm_model, "support_vectors_"):
        raise ValueError(
            "El SVM todavía no está entrenado."
        )

    return svm_model


# ============================================================
# 1. DISTRIBUCIÓN DE CLASES Y CORRELACIONES
# ============================================================

def plot_eda_distributions(
    df: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
):
    """
    Distribución de clases y correlaciones Pearson/Spearman.

    Las correlaciones se calculan respecto a la variable binaria:
    Desertor frente al resto.

    Son asociaciones descriptivas y no implican causalidad.
    """

    classes = _validate_target(
        df,
        classes
    )

    if num_cols is None:
        num_cols = NUM_COLS

    num_cols = _validate_numeric_columns(
        df,
        num_cols
    )

    data = df.dropna(
        subset=["estado_final"]
    ).copy()

    if data.empty:
        raise ValueError(
            "No existen registros válidos."
        )

    desertor_binary = (
        data["estado_final"] == "Desertor"
    ).astype(int)

    pearson_values = {}
    spearman_values = {}

    for col in num_cols:

        values = pd.to_numeric(
            data[col],
            errors="coerce"
        )

        pearson_values[col] = values.corr(
            desertor_binary,
            method="pearson"
        )

        spearman_values[col] = values.corr(
            desertor_binary,
            method="spearman"
        )

    pearson = pd.Series(
        pearson_values,
        dtype=float
    ).dropna()

    spearman = pd.Series(
        spearman_values,
        dtype=float
    ).dropna()

    pearson = pearson.loc[
        pearson.abs()
        .sort_values(ascending=False)
        .index
    ].head(10).sort_values()

    spearman = spearman.loc[
        spearman.abs()
        .sort_values(ascending=False)
        .index
    ].head(10).sort_values()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(21, 7),
        gridspec_kw={
            "width_ratios": [1, 1.2, 1.2]
        }
    )

    # Distribución de clases.
    counts = (
        data["estado_final"]
        .value_counts()
        .reindex(classes, fill_value=0)
    )

    colors = [
        _get_class_color(label, i)
        for i, label in enumerate(classes)
    ]

    bars = axes[0].bar(
        counts.index,
        counts.values,
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )

    total = counts.sum()

    for bar, count in zip(
        bars,
        counts.values
    ):

        percentage = (
            100 * count / total
            if total > 0
            else 0
        )

        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9
        )

    axes[0].set_title(
        "Distribución de clases",
        fontweight="bold"
    )

    axes[0].set_xlabel(
        "Estado académico final"
    )

    axes[0].set_ylabel(
        "Número de estudiantes"
    )

    axes[0].tick_params(
        axis="x",
        rotation=20
    )

    # Pearson.
    if not pearson.empty:

        colors_p = [
            "#d95f02" if value > 0 else "#7570b3"
            for value in pearson.values
        ]

        axes[1].barh(
            pearson.index,
            pearson.values,
            color=colors_p
        )

    axes[1].axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.9
    )

    axes[1].set_title(
        "Correlación de Pearson",
        fontweight="bold"
    )

    axes[1].set_xlabel(
        "Coeficiente de correlación"
    )

    # Spearman.
    if not spearman.empty:

        colors_s = [
            "#1b9e77" if value > 0 else "#e7298a"
            for value in spearman.values
        ]

        axes[2].barh(
            spearman.index,
            spearman.values,
            color=colors_s
        )

    axes[2].axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.9
    )

    axes[2].set_title(
        "Correlación de Spearman",
        fontweight="bold"
    )

    axes[2].set_xlabel(
        "Coeficiente de correlación"
    )

    fig.suptitle(
        "Análisis exploratorio del conjunto de datos sintético",
        fontsize=15,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.005,
        "Las correlaciones son descriptivas; no implican causalidad. "
        "La variable objetivo es sintética.",
        ha="center",
        fontsize=9,
        color="dimgray"
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 0.94]
    )

    _save_figure(
        fig,
        output_path,
        "eda_distribucion_correlaciones.png"
    )

    return fig, axes


# ============================================================
# 2. DISTRIBUCIÓN DE VARIABLES CLAVE
# ============================================================

def plot_key_variables_distribution(
    df: pd.DataFrame,
    classes: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
):
    """
    Boxplots de promedio académico y materias reprobadas.
    """

    classes = _validate_target(
        df,
        classes
    )

    required = [
        "estado_final",
        "promedio_academico",
        "materias_reprobadas"
    ]

    _validate_dataframe(
        df,
        required
    )

    data = df[required].copy()

    for col in [
        "promedio_academico",
        "materias_reprobadas"
    ]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna()

    if data.empty:
        raise ValueError(
            "No hay datos válidos para generar los boxplots."
        )

    palette = {
        label: _get_class_color(label, i)
        for i, label in enumerate(classes)
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6)
    )

    sns.boxplot(
        data=data,
        x="estado_final",
        y="promedio_academico",
        order=classes,
        hue="estado_final",
        hue_order=classes,
        palette=palette,
        dodge=False,
        ax=axes[0]
    )

    if axes[0].legend_ is not None:
        axes[0].legend_.remove()

    axes[0].set_title(
        "Distribución del promedio académico",
        fontweight="bold"
    )

    axes[0].set_xlabel(
        "Estado académico"
    )

    axes[0].set_ylabel(
        "Promedio académico"
    )

    axes[0].tick_params(
        axis="x",
        rotation=15
    )

    sns.boxplot(
        data=data,
        x="estado_final",
        y="materias_reprobadas",
        order=classes,
        hue="estado_final",
        hue_order=classes,
        palette=palette,
        dodge=False,
        ax=axes[1]
    )

    if axes[1].legend_ is not None:
        axes[1].legend_.remove()

    axes[1].set_title(
        "Distribución de materias reprobadas",
        fontweight="bold"
    )

    axes[1].set_xlabel(
        "Estado académico"
    )

    axes[1].set_ylabel(
        "Número de materias reprobadas"
    )

    axes[1].tick_params(
        axis="x",
        rotation=15
    )

    fig.suptitle(
        "Distribuciones descriptivas por estado académico",
        fontsize=15,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.005,
        "Las diferencias describen el conjunto sintético y no demuestran "
        "relaciones causales.",
        ha="center",
        fontsize=9,
        color="dimgray"
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 0.93]
    )

    _save_figure(
        fig,
        output_path,
        "eda_variables_clave.png"
    )

    return fig, axes


# ============================================================
# 3. IMPACTO DE LOS ESCALADORES
# ============================================================

def plot_scaler_impact(
    df: pd.DataFrame,
    feature: str = "distancia_hogar_ies_km",
    output_path: Optional[Union[str, Path]] = None
):
    """
    Compara la variable original con seis transformaciones:

    - StandardScaler
    - MinMaxScaler
    - RobustScaler
    - QuantileTransformer
    - MaxAbsScaler
    - PowerTransformer

    Es una comparación ilustrativa, no una evaluación de modelos.
    """

    _validate_dataframe(
        df,
        [feature]
    )

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).dropna().values.reshape(-1, 1)

    if len(values) < 2:
        raise ValueError(
            f"La variable {feature} no tiene suficientes observaciones."
        )

    if np.unique(values).size < 2:
        raise ValueError(
            f"La variable {feature} es constante."
        )

    transformers = [
        (
            "StandardScaler",
            StandardScaler()
        ),
        (
            "MinMaxScaler",
            MinMaxScaler()
        ),
        (
            "RobustScaler",
            RobustScaler()
        ),
        (
            "QuantileTransformer",
            QuantileTransformer(
                n_quantiles=min(50, len(values)),
                output_distribution="uniform",
                random_state=42
            )
        ),
        (
            "MaxAbsScaler",
            MaxAbsScaler()
        ),
        (
            "PowerTransformer",
            PowerTransformer(
                method="yeo-johnson"
            )
        )
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(19, 9)
    )

    axes = axes.flatten()

    sns.histplot(
        values.ravel(),
        bins=35,
        stat="density",
        color="gray",
        ax=axes[0]
    )

    axes[0].set_title(
        "Variable original",
        fontweight="bold"
    )

    for index, (name, transformer) in enumerate(
        transformers,
        start=1
    ):

        transformed = transformer.fit_transform(
            values
        )

        sns.histplot(
            transformed.ravel(),
            bins=35,
            stat="density",
            color=sns.color_palette("Set2")[index % 8],
            ax=axes[index]
        )

        axes[index].set_title(
            name,
            fontweight="bold"
        )

    axes[-1].axis("off")

    fig.suptitle(
        f"Impacto de las transformaciones sobre {feature}",
        fontsize=15,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.005,
        "Transformadores ajustados sobre los datos recibidos únicamente "
        "con fines ilustrativos. No representa el rendimiento de los pipelines.",
        ha="center",
        fontsize=9,
        color="dimgray"
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 0.93]
    )

    _save_figure(
        fig,
        output_path,
        "eda_efecto_escaladores.png"
    )

    return fig, axes


# ============================================================
# 4. COMPARACIÓN DE NORMALIZACIONES
# ============================================================

def plot_normalization_comparison(
    summary_df: pd.DataFrame,
    n_trials: int = 50,
    metric_col: str = "Average F1 (Macro)",
    output_path: Optional[Union[str, Path]] = None
):
    """
    Compara el rendimiento promedio de las configuraciones.

    Admite:
    - Average F1 (Macro)
    - Average F1 (Weighted)
    - Average F1 (Desertor)

    Utiliza la desviación estándar correspondiente cuando existe.
    """

    _validate_dataframe(
        summary_df,
        [
            "Normalization Method",
            metric_col
        ]
    )

    data = summary_df.copy()

    data[metric_col] = pd.to_numeric(
        data[metric_col],
        errors="coerce"
    )

    data = data.dropna(
        subset=[metric_col]
    )

    if data.empty:
        raise ValueError(
            "No existen resultados válidos para comparar."
        )

    std_col = metric_col.replace(
        "Average",
        "Std"
    )

    data["_std"] = 0.0

    if std_col in data.columns:

        data["_std"] = pd.to_numeric(
            data[std_col],
            errors="coerce"
        ).fillna(0)

    data["_std"] = data["_std"].clip(
        lower=0
    )

    data = data.sort_values(
        metric_col,
        ascending=True
    ).reset_index(drop=True)

    methods = data[
        "Normalization Method"
    ].astype(str).values

    values = data[
        metric_col
    ].values

    errors = data[
        "_std"
    ].values

    colors = [
        "#d95f02"
        if (
            "baseline" in name.lower()
            or "no normalization" in name.lower()
        )
        else "#1b9e77"
        for name in methods
    ]

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    positions = np.arange(
        len(data)
    )

    ax.barh(
        positions,
        values,
        xerr=errors,
        color=colors,
        capsize=4,
        edgecolor="black",
        linewidth=0.4
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(methods)

    ax.set_xlabel(metric_col)
    ax.set_ylabel("Configuración")

    ax.set_title(
        f"Comparación de configuraciones SVM "
        f"({metric_col}, {n_trials} trials)",
        fontsize=14,
        fontweight="bold"
    )

    lower = max(
        0,
        float(np.nanmin(values - errors)) - 0.05
    )

    upper = min(
        1.05,
        float(np.nanmax(values + errors)) + 0.08
    )

    if upper <= lower:
        upper = min(1.05, lower + 0.1)

    ax.set_xlim(
        lower,
        upper
    )

    for position, value in zip(
        positions,
        values
    ):

        ax.text(
            value + 0.005,
            position,
            f"{value:.4f}",
            va="center",
            fontsize=9
        )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.4
    )

    fig.tight_layout()

    _save_figure(
        fig,
        output_path,
        "svm_normalizacion_vs_baseline.png"
    )

    return fig, ax


# ============================================================
# 5. IMPORTANCIA DE VARIABLES POR PERMUTACIÓN
# ============================================================


# ============================================================
# 5. IMPORTANCIA DE VARIABLES POR PERMUTACIÓN
# ============================================================

def plot_permutation_importance(
    imp_df: pd.DataFrame,
    top_n: int = 15,
    output_path: Optional[Union[str, Path]] = None
):
    """
    Visualiza la importancia de variables por permutación.

    Admite los dos formatos de columnas:

    Formato 1:
    - Feature
    - Importance Mean
    - Importance Std

    Formato 2:
    - Feature
    - Importance_Mean
    - Importance_Std

    La métrica utilizada por evaluate.py es F1 macro.

    Las importancias negativas son válidas: indican que la
    permutación de una variable no mejoró el rendimiento y,
    en ese ensayo, incluso pudo aumentarlo.
    """

    if not isinstance(imp_df, pd.DataFrame):
        raise TypeError(
            "imp_df debe ser un pandas.DataFrame."
        )

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError(
            "top_n debe ser un entero mayor que cero."
        )

    data = imp_df.copy()

    # --------------------------------------------------------
    # Compatibilidad de nombres de columnas
    # --------------------------------------------------------

    column_aliases = {
        "Importance Mean": "Importance_Mean",
        "Importance Std": "Importance_Std"
    }

    data = data.rename(
        columns={
            old: new
            for old, new in column_aliases.items()
            if old in data.columns
            and new not in data.columns
        }
    )

    required = [
        "Feature",
        "Importance_Mean",
        "Importance_Std"
    ]

    _validate_dataframe(
        data,
        required
    )

    # --------------------------------------------------------
    # Conversión y limpieza de datos
    # --------------------------------------------------------

    data["Feature"] = (
        data["Feature"]
        .astype(str)
        .str.strip()
    )

    data["Importance_Mean"] = pd.to_numeric(
        data["Importance_Mean"],
        errors="coerce"
    )

    data["Importance_Std"] = pd.to_numeric(
        data["Importance_Std"],
        errors="coerce"
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna(
        subset=[
            "Feature",
            "Importance_Mean"
        ]
    )

    data["Importance_Std"] = (
        data["Importance_Std"]
        .fillna(0)
        .clip(lower=0)
    )

    data = data[
        data["Feature"] != ""
    ]

    if data.empty:
        raise ValueError(
            "No hay importancias numéricas válidas para graficar."
        )

    # --------------------------------------------------------
    # Selección de las variables más importantes
    # --------------------------------------------------------

    data = data.sort_values(
        "Importance_Mean",
        ascending=False
    ).head(top_n)

    data = data.sort_values(
        "Importance_Mean",
        ascending=True
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Construcción de la figura
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            11,
            max(6, len(data) * 0.4)
        )
    )

    ax.barh(
        data["Feature"],
        data["Importance_Mean"],
        xerr=data["Importance_Std"],
        color="#2b5c8f",
        capsize=3,
        edgecolor="black",
        linewidth=0.4,
        error_kw={
            "elinewidth": 1,
            "capthick": 1
        }
    )

    # Referencia de importancia cero.
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8
    )

    ax.set_title(
        "Importancia de variables por permutación",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Disminución promedio del F1 macro al permutar la variable"
    )

    ax.set_ylabel(
        "Variable"
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.35
    )

    # --------------------------------------------------------
    # Anotación metodológica
    # --------------------------------------------------------

    ax.text(
        0.01,
        -0.13,
        "Valores positivos indican pérdida de rendimiento al permutar "
        "la variable; valores negativos indican una mejora observada.",
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray"
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Exportación
    # --------------------------------------------------------

    _save_figure(
        fig,
        output_path,
        "svm_importancia_variables.png"
    )

    return fig, ax



# ============================================================
# 6. PROYECCIÓN PCA Y VECTORES DE SOPORTE
# ============================================================

def plot_pca_hyperplane(
    pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    classes: Optional[List[str]] = None,
    sample_size: int = 1000,
    random_state: int = 42,
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "SVM"
):
    """
    Proyecta mediante PCA las observaciones de prueba y los vectores
    de soporte de un SVM entrenado.

    PCA se ajusta exclusivamente con las características transformadas
    del conjunto de entrenamiento.

    No representa las fronteras reales del SVM.
    """

    if classes is None:
        classes = list(CLASSES)

    _validate_sample_size(
        sample_size,
        minimum=2
    )

    if not hasattr(pipeline, "named_steps"):
        raise TypeError(
            "pipeline debe ser un sklearn Pipeline."
        )

    if "preprocessor" not in pipeline.named_steps:
        raise ValueError(
            "El pipeline debe contener el paso 'preprocessor'."
        )

    svm_model = _get_svm_from_pipeline(
        pipeline
    )

    if len(X_train) < 2:
        raise ValueError(
            "Se necesitan al menos dos observaciones de entrenamiento."
        )

    if len(X_test) == 0:
        raise ValueError(
            "El conjunto de prueba está vacío."
        )

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    X_train_trans = _to_dense(
        preprocessor.transform(X_train)
    )

    X_test_trans = _to_dense(
        preprocessor.transform(X_test)
    )

    if X_train_trans.shape[1] < 2:
        raise ValueError(
            "PCA necesita al menos dos características transformadas."
        )

    if svm_model.support_vectors_.shape[1] != X_train_trans.shape[1]:
        raise ValueError(
            "Las dimensiones de los vectores de soporte no coinciden "
            "con las características transformadas."
        )

    pca = PCA(
        n_components=2,
        random_state=random_state
    )

    pca.fit(
        X_train_trans
    )

    X_test_pca = pca.transform(
        X_test_trans
    )

    support_vectors_pca = pca.transform(
        svm_model.support_vectors_
    )

    y_test_array = np.asarray(y_test)

    if len(y_test_array) != len(X_test):
        raise ValueError(
            "y_test y X_test deben tener la misma cantidad de registros."
        )

    rng = np.random.default_rng(
        random_state
    )

    sample_idx = rng.choice(
        len(y_test_array),
        size=min(sample_size, len(y_test_array)),
        replace=False
    )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    sampled_labels = y_test_array[sample_idx]

    for index, label in enumerate(classes):

        mask = sampled_labels == label

        if not np.any(mask):
            continue

        ax.scatter(
            X_test_pca[sample_idx][mask, 0],
            X_test_pca[sample_idx][mask, 1],
            color=_get_class_color(label, index),
            label=label,
            alpha=0.62,
            edgecolors="white",
            linewidths=0.25,
            s=25
        )

    ax.scatter(
        support_vectors_pca[:, 0],
        support_vectors_pca[:, 1],
        s=40,
        facecolors="none",
        edgecolors="black",
        linewidths=0.7,
        alpha=0.7,
        label=(
            f"Vectores de soporte "
            f"(n={len(support_vectors_pca)})"
        )
    )

    variance_1 = (
        pca.explained_variance_ratio_[0] * 100
    )

    variance_2 = (
        pca.explained_variance_ratio_[1] * 100
    )

    ax.set_title(
        f"Proyección PCA de observaciones y vectores de soporte "
        f"({model_name})",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        f"Componente principal 1 ({variance_1:.1f}%)"
    )

    ax.set_ylabel(
        f"Componente principal 2 ({variance_2:.1f}%)"
    )

    ax.text(
        0.01,
        0.01,
        "PCA es una proyección visual y no representa "
        "las fronteras reales del SVM.",
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray"
    )

    ax.legend(
        loc="best",
        fontsize=9
    )

    fig.tight_layout()

    filename = (
        "svm_proyeccion_pca_"
        f"{str(model_name).lower().replace(' ', '_')}.png"
    )

    _save_figure(
        fig,
        output_path,
        filename
    )

    return fig, ax


# ============================================================
# 7. MATRICES DE CONFUSIÓN
# ============================================================

def plot_confusion_matrices(
    cm: np.ndarray,
    classes: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
):
    """
    Genera matriz de confusión absoluta y normalizada por filas.

    La normalización por filas representa el recall de cada clase.
    """

    if classes is None:
        classes = list(CLASSES)

    cm = np.asarray(cm)

    expected_shape = (
        len(classes),
        len(classes)
    )

    if cm.shape != expected_shape:
        raise ValueError(
            f"La matriz debe tener forma {expected_shape}; "
            f"se recibió {cm.shape}."
        )

    if not np.issubdtype(cm.dtype, np.number):
        raise TypeError(
            "La matriz de confusión debe contener valores numéricos."
        )

    if not np.all(np.isfinite(cm)):
        raise ValueError(
            "La matriz contiene valores no finitos."
        )

    if np.any(cm < 0):
        raise ValueError(
            "La matriz no puede contener valores negativos."
        )

    row_totals = cm.sum(
        axis=1,
        keepdims=True
    )

    cm_norm = np.divide(
        cm.astype(float),
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={
            "size": 12,
            "weight": "bold"
        }
    )

    axes[0].set_title(
        "Matriz de confusión: conteos absolutos",
        fontweight="bold"
    )

    axes[0].set_xlabel("Clase predicha")
    axes[0].set_ylabel("Clase real")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=False,
        ax=axes[1],
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={
            "size": 12,
            "weight": "bold"
        }
    )

    axes[1].set_title(
        "Matriz normalizada por clase real (Recall)",
        fontweight="bold"
    )

    axes[1].set_xlabel("Clase predicha")
    axes[1].set_ylabel("Clase real")

    fig.tight_layout()

    _save_figure(
        fig,
        output_path,
        "svm_matriz_confusion.png"
    )

    return fig, axes


# ============================================================
# 8. FRONTERA BINARIA EXPLORATORIA
# ============================================================

def plot_decision_boundary_top2(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    sample_size: int = 300,
    random_state: int = 42
):
    """
    Entrena un SVM RBF binario independiente para visualizar
    Desertor frente al resto usando dos variables.

    Debe recibir únicamente datos de entrenamiento/desarrollo,
    nunca el test final.

    No representa el pipeline multiclase definitivo.
    """

    if features is None:
        features = [
            "promedio_academico",
            "materias_reprobadas"
        ]

    if len(features) != 2:
        raise ValueError(
            "Se deben proporcionar exactamente dos variables."
        )

    _validate_sample_size(
        sample_size,
        minimum=10
    )

    _validate_dataframe(
        df,
        features + ["estado_final"]
    )

    data = df[
        features + ["estado_final"]
    ].copy()

    for col in features:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(
        subset=features + ["estado_final"]
    )

    data["target_binario"] = (
        data["estado_final"] == "Desertor"
    ).astype(int)

    if data["target_binario"].nunique() != 2:
        raise ValueError(
            "Se necesitan registros Desertor y no Desertor."
        )

    if len(data) < 10:
        raise ValueError(
            "No hay suficientes registros para el análisis."
        )

    if data["target_binario"].value_counts().min() < 2:
        raise ValueError(
            "Cada clase necesita al menos dos observaciones."
        )

    # Muestreo estratificado.
    if len(data) > sample_size:

        sampled_parts = []

        proportions = (
            data["target_binario"]
            .value_counts(normalize=True)
        )

        for label in [0, 1]:

            group = data[
                data["target_binario"] == label
            ]

            n_group = max(
                1,
                int(round(sample_size * proportions[label]))
            )

            n_group = min(
                n_group,
                len(group)
            )

            sampled_parts.append(
                group.sample(
                    n=n_group,
                    random_state=random_state
                )
            )

        data = pd.concat(
            sampled_parts
        ).sample(
            frac=1,
            random_state=random_state
        )

    X = data[features].values
    y = data["target_binario"].values

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    svm = SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced"
    )

    svm.fit(
        X_scaled,
        y
    )

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    x_padding = max(
        (x_max - x_min) * 0.08,
        0.1
    )

    y_padding = max(
        (y_max - y_min) * 0.08,
        0.1
    )

    xx, yy = np.meshgrid(
        np.linspace(
            x_min - x_padding,
            x_max + x_padding,
            250
        ),
        np.linspace(
            y_min - y_padding,
            y_max + y_padding,
            250
        )
    )

    grid = np.c_[
        xx.ravel(),
        yy.ravel()
    ]

    grid_scaled = scaler.transform(
        grid
    )

    decision = svm.decision_function(
        grid_scaled
    ).reshape(xx.shape)

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.contourf(
        xx,
        yy,
        decision,
        levels=[
            -np.inf,
            0,
            np.inf
        ],
        colors=[
            "#d9edf7",
            "#f8d7da"
        ],
        alpha=0.6
    )

    contours = ax.contour(
        xx,
        yy,
        decision,
        levels=[-1, 0, 1],
        colors=[
            "#2b83ba",
            "#d7191c",
            "#2b83ba"
        ],
        linestyles=[
            "--",
            "-",
            "--"
        ],
        linewidths=[
            1.5,
            2.5,
            1.5
        ]
    )

    ax.clabel(
        contours,
        inline=True,
        fontsize=9,
        fmt={
            -1: "Puntuación -1",
            0: "Decisión 0",
            1: "Puntuación +1"
        }
    )

    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color="#2b83ba",
        label="No Desertor",
        alpha=0.65,
        edgecolors="black",
        linewidths=0.3,
        s=35
    )

    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color="#d7191c",
        label="Desertor",
        alpha=0.7,
        edgecolors="black",
        linewidths=0.3,
        s=35
    )

    support_original = scaler.inverse_transform(
        svm.support_vectors_
    )

    ax.scatter(
        support_original[:, 0],
        support_original[:, 1],
        s=85,
        facecolors="none",
        edgecolors="black",
        linewidths=1,
        label=(
            f"Vectores de soporte "
            f"(n={len(support_original)})"
        )
    )

    ax.set_title(
        "Separación binaria exploratoria con SVM RBF",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])

    ax.text(
        0.01,
        0.01,
        "Modelo binario independiente. No corresponde al SVM "
        "multiclase final.",
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray"
    )

    ax.legend(
        loc="best",
        fontsize=9
    )

    fig.tight_layout()

    _save_figure(
        fig,
        output_path,
        "svm_frontera_binaria_dos_variables.png"
    )

    return fig, ax


# ============================================================
# 9. DISTRIBUCIÓN DE PUNTUACIONES DE DECISIÓN
# ============================================================

def plot_decision_function_distribution(
    df: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    random_state: int = 42
):
    """
    Entrena un SVM RBF binario independiente.

    El modelo se ajusta sobre entrenamiento y las puntuaciones
    se calculan exclusivamente sobre observaciones de prueba.

    No representa las puntuaciones del SVM multiclase definitivo.
    """

    _validate_target(df)

    if num_cols is None:
        num_cols = [
            "promedio_academico",
            "materias_reprobadas",
            "semestre_cursado",
            "distancia_hogar_ies_km",
            "estrato_socioeconomico"
        ]

    num_cols = _validate_numeric_columns(
        df,
        num_cols
    )

    data = df[
        num_cols + ["estado_final"]
    ].copy()

    for col in num_cols:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna()

    X = data[num_cols].values

    y = (
        data["estado_final"] == "Desertor"
    ).astype(int).values

    if np.unique(y).size != 2:
        raise ValueError(
            "Se necesitan ambas clases para entrenar el SVM binario."
        )

    if np.bincount(y).min() < 2:
        raise ValueError(
            "Cada clase necesita al menos dos observaciones."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    svm = SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced"
    )

    svm.fit(
        X_train_scaled,
        y_train
    )

    scores = svm.decision_function(
        X_test_scaled
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    sns.histplot(
        scores[y_test == 0],
        color="#2b83ba",
        label="No Desertor",
        stat="density",
        element="step",
        fill=True,
        alpha=0.35,
        bins=30,
        ax=ax
    )

    sns.histplot(
        scores[y_test == 1],
        color="#d7191c",
        label="Desertor",
        stat="density",
        element="step",
        fill=True,
        alpha=0.35,
        bins=30,
        ax=ax
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Umbral de decisión = 0"
    )

    ax.set_title(
        "Distribución de puntuaciones de decisión del SVM RBF binario",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Puntuación de decisión SVM RBF (binario)"
    )

    ax.set_ylabel(
        "Densidad"
    )

    ax.text(
        0.01,
        0.01,
        "Puntuaciones calculadas sobre el conjunto de prueba "
        "del modelo binario. No son distancias geométricas.",
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray"
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    _save_figure(
        fig,
        output_path,
        "svm_puntuaciones_decision_binaria.png"
    )

    return fig, ax


# ============================================================
# 10. MÉTRICAS DE CLASIFICACIÓN POR CLASE
# ============================================================

def plot_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
):
    """
    Compara precisión, recall y F1-score de cada clase.
    Utiliza etiquetas textuales.
    """

    if classes is None:
        classes = list(CLASSES)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true y y_pred deben tener la misma longitud."
        )

    if len(y_true) == 0:
        raise ValueError(
            "No hay observaciones para evaluar."
        )

    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0
    )

    metrics = []

    for label in classes:

        if label not in report:
            continue

        metrics.append({
            "Clase": label,
            "Precisión": report[label]["precision"],
            "Recall": report[label]["recall"],
            "F1-score": report[label]["f1-score"]
        })

    metrics_df = pd.DataFrame(
        metrics
    )

    if metrics_df.empty:
        raise ValueError(
            "No se encontraron métricas por clase."
        )

    plot_df = metrics_df.melt(
        id_vars="Clase",
        var_name="Métrica",
        value_name="Valor"
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    sns.barplot(
        data=plot_df,
        x="Clase",
        y="Valor",
        hue="Métrica",
        palette="Set2",
        ax=ax
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_title(
        "Desempeño del SVM por clase",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Clase")
    ax.set_ylabel("Valor de la métrica")

    ax.legend(
        title="Métrica",
        loc="best"
    )

    fig.tight_layout()

    _save_figure(
        fig,
        output_path,
        "svm_metricas_por_clase.png"
    )

    return fig, ax


# ============================================================
# 11. DISTRIBUCIÓN DE RESULTADOS ENTRE TRIALS
# ============================================================

def plot_trial_score_distributions(
    results_macro: dict,
    output_path: Optional[Union[str, Path]] = None,
    metric_name: str = "F1 macro"
):
    """
    Representa la distribución de resultados entre trials.

    results_macro debe ser el diccionario devuelto por
    run_multiclass_trials para la métrica F1 macro.
    """

    if not isinstance(results_macro, dict) or not results_macro:
        raise ValueError(
            "results_macro debe ser un diccionario no vacío."
        )

    records = []

    for method, values in results_macro.items():

        values = np.asarray(
            values,
            dtype=float
        )

        values = values[
            np.isfinite(values)
        ]

        for value in values:

            records.append({
                "Configuración": str(method),
                "Resultado": float(value)
            })

    data = pd.DataFrame(
        records
    )

    if data.empty:
        raise ValueError(
            "No hay resultados numéricos válidos."
        )

    order = (
        data.groupby("Configuración")["Resultado"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    sns.boxplot(
        data=data,
        x="Resultado",
        y="Configuración",
        order=order,
        color="#a6cee3",
        showfliers=False,
        ax=ax
    )

    sns.stripplot(
        data=data,
        x="Resultado",
        y="Configuración",
        order=order,
        color="#1f78b4",
        alpha=0.35,
        size=3,
        ax=ax
    )

    ax.set_title(
        f"Variabilidad del rendimiento entre trials ({metric_name})",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(metric_name)
    ax.set_ylabel("Configuración")

    fig.tight_layout()

    _save_figure(
        fig,
        output_path,
        "svm_distribucion_f1_trials.png"
    )

    return fig, ax


# ============================================================
# 12. MÉTRICAS ESPECÍFICAS DE LA CLASE DESERTOR
# ============================================================

def plot_desertor_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Union[str, Path]] = None
):
    """
    Muestra precisión, recall y F1-score de la clase Desertor.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true y y_pred deben tener la misma longitud."
        )

    if len(y_true) == 0:
        raise ValueError(
            "No hay observaciones para evaluar."
        )

    report = classification_report(
        y_true,
        y_pred,
        labels=["Desertor"],
        output_dict=True,
        zero_division=0
    )

    if "Desertor" not in report:
        raise ValueError(
            "No se encontraron resultados para la clase Desertor."
        )

    values = [
        report["Desertor"]["precision"],
        report["Desertor"]["recall"],
        report["Desertor"]["f1-score"]
    ]

    labels = [
        "Precisión",
        "Recall",
        "F1-score"
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        labels,
        values,
        color=[
            "#80b1d3",
            "#fb8072",
            "#8dd3c7"
        ],
        edgecolor="black",
        linewidth=0.5
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    ax.set_ylim(
        0,
        1.12
    )

    ax.set_title(
        "Desempeño del modelo en la clase Desertor",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Valor de la métrica"
    )

    ax.text(
        0.5,
        -0.16,
        "El recall indica qué proporción de los desertores reales "
        "fue identificada correctamente.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="dimgray"
    )

    fig.tight_layout()

    _save_figure(
        fig,
        output_path,
        "svm_metricas_clase_desertor.png"
    )

    return fig, ax
