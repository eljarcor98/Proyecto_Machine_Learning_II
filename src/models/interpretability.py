
"""
Explicabilidad de modelos SVM mediante SHAP, LIME y PDP
=======================================================

Explica los modelos finales RBF y Poly-3 utilizando las variables originales.

SHAP:
    - Importancia global basada en valores SHAP.
    - Explicación local de un caso de la clase Desertor.
    - Se explica el score funcional de Desertor, no una probabilidad.
    - Las variables categóricas se codifican temporalmente para SHAP y
      se reconstruyen antes de consultar el modelo.

LIME:
    - Explicación local de un caso Desertor.
    - Softmax sobre decision_function para obtener puntuaciones relativas.
    - Estas puntuaciones NO son probabilidades calibradas.

PDP:
    - Dependencia parcial del score funcional de Desertor.
    - Se analizan variables numéricas seleccionadas.
    - No se interpretan los resultados como relaciones causales.

Los modelos no se reentrenan ni se modifican.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from lime.lime_tabular import LimeTabularExplainer
from scipy.special import softmax


# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

RANDOM_STATE = 42

SHAP_BACKGROUND_SIZE = 30
SHAP_EXPLAIN_SIZE = 15

LIME_NUM_SAMPLES = 1000
LIME_NUM_FEATURES = 10

PDP_SAMPLE_SIZE = 150
PDP_GRID_SIZE = 20

TARGET_CLASS = "Desertor"

PDP_PREFERRED_FEATURES = [
    "semestre_cursado",
    "promedio_academico",
    "materias_reprobadas"
]


# =============================================================================
# 2. VALIDACIÓN Y FUNCIONES AUXILIARES
# =============================================================================

def _validate_inputs(model, X_background, X_explain):
    """Comprueba la compatibilidad de los datos y del modelo."""

    if not isinstance(X_background, pd.DataFrame):
        raise TypeError("X_background debe ser un DataFrame.")

    if not isinstance(X_explain, pd.DataFrame):
        raise TypeError("X_explain debe ser un DataFrame.")

    if X_background.empty or X_explain.empty:
        raise ValueError(
            "Los datos para explicabilidad no pueden estar vacíos."
        )

    if list(X_background.columns) != list(X_explain.columns):
        raise ValueError(
            "X_background y X_explain deben tener las mismas columnas "
            "y en el mismo orden."
        )

    if X_background.isnull().any().any():
        raise ValueError(
            "X_background contiene valores faltantes. "
            "Revise el preprocesamiento antes de calcular explicabilidad."
        )

    if X_explain.isnull().any().any():
        raise ValueError(
            "X_explain contiene valores faltantes. "
            "Revise el preprocesamiento antes de calcular explicabilidad."
        )

    if not hasattr(model, "classes_"):
        raise TypeError(
            "El modelo debe exponer el atributo classes_."
        )

    if TARGET_CLASS not in model.classes_:
        raise ValueError(
            f"El modelo no contiene la clase {TARGET_CLASS}. "
            f"Clases disponibles: {list(model.classes_)}"
        )

    if not hasattr(model, "decision_function"):
        raise TypeError(
            "El modelo debe implementar decision_function()."
        )

    return True


def _get_target_class_index(model):
    """Obtiene la posición de la clase Desertor en el modelo."""

    return list(model.classes_).index(TARGET_CLASS)


def _get_target_decision_score(model, X):
    """
    Devuelve el score funcional correspondiente a Desertor.

    No transforma los scores en probabilidades.
    """

    scores = np.asarray(model.decision_function(X))

    if scores.ndim == 1:
        if len(model.classes_) != 2:
            raise ValueError(
                "Se esperaba una salida multiclase de decision_function."
            )
        return scores

    target_index = _get_target_class_index(model)

    if scores.shape[1] != len(model.classes_):
        raise ValueError(
            "La dimensión de decision_function no coincide "
            "con el número de clases."
        )

    return scores[:, target_index]


def _select_desertor_case(X_explain, y_explain=None):
    """
    Selecciona un caso de la clase Desertor cuando y_explain
    está disponible. En caso contrario, utiliza el primer registro.
    """

    if y_explain is not None:

        y_array = np.asarray(y_explain)

        if len(y_array) != len(X_explain):
            raise ValueError(
                "y_explain y X_explain tienen longitudes diferentes."
            )

        positions = np.where(y_array == TARGET_CLASS)[0]

        if len(positions) > 0:
            return int(positions[0])

    return 0


def _save_figure(fig, path):
    """Guarda y cierra una figura."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"-> Figura guardada: {path}")


def _save_csv(df, path):
    """Guarda un DataFrame en CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"-> CSV guardado: {path}")


# =============================================================================
# 3. PREPARACIÓN NUMÉRICA TEMPORAL PARA SHAP
# =============================================================================

def _prepare_shap_data(X_background, X_explain):
    """
    Codifica temporalmente las variables categóricas para SHAP.

    Las variables numéricas conservan sus valores.
    Las variables categóricas se representan mediante códigos enteros.

    La codificación NO modifica los datos originales ni el modelo.
    Antes de llamar al SVM, los códigos se reconstruyen a sus categorías
    originales.
    """

    feature_names = list(X_background.columns)

    encoded_background = np.zeros(
        (len(X_background), len(feature_names)),
        dtype=float
    )

    encoded_explain = np.zeros(
        (len(X_explain), len(feature_names)),
        dtype=float
    )

    categorical_features = []
    category_maps = {}

    combined = pd.concat(
        [X_background, X_explain],
        axis=0
    )

    for position, feature in enumerate(feature_names):

        is_numeric = (
            pd.api.types.is_numeric_dtype(X_background[feature])
            and not pd.api.types.is_bool_dtype(X_background[feature])
        )

        if is_numeric:

            background_values = pd.to_numeric(
                X_background[feature],
                errors="raise"
            ).to_numpy(dtype=float)

            explain_values = pd.to_numeric(
                X_explain[feature],
                errors="raise"
            ).to_numpy(dtype=float)

            if (
                not np.isfinite(background_values).all()
                or not np.isfinite(explain_values).all()
            ):
                raise ValueError(
                    f"La variable numérica {feature} contiene "
                    "valores no finitos."
                )

            encoded_background[:, position] = background_values
            encoded_explain[:, position] = explain_values

        else:

            categorical_features.append(position)

            categories = pd.Index(
                combined[feature].unique()
            ).tolist()

            if not categories:
                raise ValueError(
                    f"No se encontraron categorías para {feature}."
                )

            category_to_code = {
                category: index
                for index, category in enumerate(categories)
            }

            code_to_category = {
                index: category
                for index, category in enumerate(categories)
            }

            category_maps[feature] = code_to_category

            background_codes = (
                X_background[feature]
                .map(category_to_code)
            )

            explain_codes = (
                X_explain[feature]
                .map(category_to_code)
            )

            if background_codes.isnull().any():
                raise ValueError(
                    f"No fue posible codificar todas las categorías "
                    f"de {feature} en X_background."
                )

            if explain_codes.isnull().any():
                raise ValueError(
                    f"No fue posible codificar todas las categorías "
                    f"de {feature} en X_explain."
                )

            encoded_background[:, position] = (
                background_codes.to_numpy(dtype=float)
            )

            encoded_explain[:, position] = (
                explain_codes.to_numpy(dtype=float)
            )

    return (
        encoded_background,
        encoded_explain,
        feature_names,
        categorical_features,
        category_maps
    )


def _decode_shap_data(
    encoded_data,
    feature_names,
    categorical_features,
    category_maps
):
    """
    Reconstruye un DataFrame con los nombres y categorías originales
    a partir de la matriz numérica temporal utilizada por SHAP.
    """

    encoded_data = np.asarray(encoded_data, dtype=float)

    if encoded_data.ndim == 1:
        encoded_data = encoded_data.reshape(1, -1)

    if encoded_data.shape[1] != len(feature_names):
        raise ValueError(
            "La cantidad de columnas recibida por SHAP no coincide "
            "con las variables originales."
        )

    categorical_set = set(categorical_features)

    reconstructed = {}

    for position, feature in enumerate(feature_names):

        column_values = encoded_data[:, position]

        if position in categorical_set:

            mapping = category_maps[feature]

            codes = np.rint(column_values).astype(int)

            invalid_codes = [
                code
                for code in codes
                if code not in mapping
            ]

            if invalid_codes:
                raise ValueError(
                    f"SHAP generó códigos categóricos no válidos "
                    f"para {feature}: {invalid_codes[:5]}"
                )

            reconstructed[feature] = [
                mapping[code]
                for code in codes
            ]

        else:

            reconstructed[feature] = column_values.astype(float)

    return pd.DataFrame(
        reconstructed,
        columns=feature_names
    )


# =============================================================================
# 4. SHAP
# =============================================================================

def calculate_shap_explanations(
    model,
    X_background,
    X_explain,
    y_explain,
    model_key,
    figures_dir,
    metrics_dir
):
    """
    Calcula explicaciones SHAP globales y locales.

    SHAP trabaja con una matriz numérica temporal. La función predictora
    reconstruye las variables originales antes de consultar el pipeline SVM.
    """

    print("\n" + "-" * 70)
    print(f"SHAP: MODELO {model_key.upper()}")
    print("-" * 70)

    (
        encoded_background,
        encoded_explain,
        feature_names,
        categorical_features,
        category_maps
    ) = _prepare_shap_data(
        X_background,
        X_explain
    )

    background_size = min(
        SHAP_BACKGROUND_SIZE,
        len(encoded_background)
    )

    background_indices = np.random.default_rng(
        RANDOM_STATE
    ).choice(
        len(encoded_background),
        size=background_size,
        replace=False
    )

    background = encoded_background[background_indices]

    explain_size = min(
        SHAP_EXPLAIN_SIZE,
        len(encoded_explain)
    )

    explain_data = encoded_explain[:explain_size]

    case_position = _select_desertor_case(
        X_explain,
        y_explain
    )

    local_case = encoded_explain[
        case_position:case_position + 1
    ]

    def predict_desertor_score(encoded_data):
        """Calcula el puntaje de decisión funcional para la clase Desertor a partir de datos codificados."""
        frame = _decode_shap_data(
            encoded_data=encoded_data,
            feature_names=feature_names,
            categorical_features=categorical_features,
            category_maps=category_maps
        )

        return _get_target_decision_score(
            model,
            frame
        )

    masker = shap.maskers.Independent(
        background,
        max_samples=len(background)
    )

    explainer = shap.Explainer(
        predict_desertor_score,
        masker,
        algorithm="permutation",
        feature_names=feature_names
    )

    max_evals = 2 * len(feature_names) + 1

    print(
        f"-> Calculando SHAP para {len(explain_data)} registros "
        f"y {len(feature_names)} variables..."
    )

    print(
        f"-> Variables categóricas codificadas temporalmente: "
        f"{len(categorical_features)}."
    )

    shap_explanation = explainer(
        explain_data,
        max_evals=max_evals
    )

    shap_values = np.asarray(
        shap_explanation.values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    if shap_values.ndim != 2:
        raise ValueError(
            f"Dimensión SHAP inesperada: {shap_values.shape}"
        )

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "La cantidad de valores SHAP no coincide con "
            "las variables originales."
        )

    # -------------------------------------------------------------------------
    # IMPORTANCIA GLOBAL
    # -------------------------------------------------------------------------

    global_importance = np.mean(
        np.abs(shap_values),
        axis=0
    )

    global_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Absolute SHAP": global_importance
    }).sort_values(
        "Mean Absolute SHAP",
        ascending=False
    )

    _save_csv(
        global_df,
        metrics_dir / f"shap_importance_{model_key}.csv"
    )

    top_global = global_df.head(15).sort_values(
        "Mean Absolute SHAP",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        top_global["Feature"],
        top_global["Mean Absolute SHAP"]
    )

    ax.set_title(
        f"Importancia global SHAP - {model_key.upper()}",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Media del valor absoluto SHAP"
    )

    ax.set_ylabel("Variable original")

    ax.grid(
        axis="x",
        alpha=0.25
    )

    _save_figure(
        fig,
        figures_dir / f"shap_summary_{model_key}.png"
    )

    # -------------------------------------------------------------------------
    # EXPLICACIÓN LOCAL
    # -------------------------------------------------------------------------

    print(
        "-> Calculando explicación SHAP local para un caso Desertor..."
    )

    local_explanation = explainer(
        local_case,
        max_evals=max_evals
    )

    local_values = np.asarray(
        local_explanation.values
    ).reshape(-1)

    if len(local_values) != len(feature_names):
        raise ValueError(
            "La cantidad de valores SHAP locales no coincide "
            "con las variables originales."
        )

    local_original = X_explain.iloc[
        [case_position]
    ].copy()

    local_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": local_values,
        "Feature Value": (
            local_original.iloc[0]
            .astype(str)
            .reindex(feature_names)
            .values
        )
    })

    local_df["Absolute SHAP"] = np.abs(
        local_df["SHAP Value"]
    )

    local_df = local_df.sort_values(
        "Absolute SHAP",
        ascending=False
    )

    _save_csv(
        local_df,
        metrics_dir / f"shap_local_{model_key}.csv"
    )

    top_local = local_df.head(12).sort_values(
        "SHAP Value",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        top_local["Feature"],
        top_local["SHAP Value"]
    )

    ax.axvline(
        0,
        linewidth=1
    )

    ax.set_title(
        f"Explicación local SHAP - {model_key.upper()}\n"
        "Caso perteneciente a la clase Desertor",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Contribución SHAP al score funcional de Desertor"
    )

    ax.set_ylabel("Variable original")

    ax.grid(
        axis="x",
        alpha=0.25
    )

    _save_figure(
        fig,
        figures_dir / f"shap_local_{model_key}.png"
    )

    print(
        "-> SHAP finalizado. Los valores explican el score funcional "
        "de Desertor, no probabilidades."
    )

    return global_df, local_df


# =============================================================================
# 5. LIME
# =============================================================================

def _prepare_lime_data(X_background, X_explain):
    """
    Codifica las variables categóricas para LimeTabularExplainer.

    Antes de consultar el modelo, los códigos se convierten nuevamente
    a las categorías originales.
    """

    feature_names = list(X_background.columns)

    combined = pd.concat(
        [X_background, X_explain],
        axis=0
    )

    categorical_features = []
    categorical_names = {}
    category_maps = {}

    encoded_background = np.zeros(
        (len(X_background), len(feature_names)),
        dtype=float
    )

    encoded_explain = np.zeros(
        (len(X_explain), len(feature_names)),
        dtype=float
    )

    for position, feature in enumerate(feature_names):

        is_numeric = (
            pd.api.types.is_numeric_dtype(X_background[feature])
            and not pd.api.types.is_bool_dtype(X_background[feature])
        )

        if is_numeric:

            encoded_background[:, position] = (
                pd.to_numeric(
                    X_background[feature],
                    errors="raise"
                ).to_numpy(dtype=float)
            )

            encoded_explain[:, position] = (
                pd.to_numeric(
                    X_explain[feature],
                    errors="raise"
                ).to_numpy(dtype=float)
            )

        else:

            categorical_features.append(position)

            categories = pd.Index(
                combined[feature].unique()
            ).tolist()

            if not categories:
                raise ValueError(
                    f"La variable categórica {feature} no tiene categorías."
                )

            category_to_code = {
                category: index
                for index, category in enumerate(categories)
            }

            code_to_category = {
                index: category
                for index, category in enumerate(categories)
            }

            category_maps[feature] = code_to_category

            categorical_names[position] = [
                str(category)
                for category in categories
            ]

            encoded_background[:, position] = (
                X_background[feature]
                .map(category_to_code)
                .to_numpy(dtype=float)
            )

            encoded_explain[:, position] = (
                X_explain[feature]
                .map(category_to_code)
                .to_numpy(dtype=float)
            )

    return (
        encoded_background,
        encoded_explain,
        categorical_features,
        categorical_names,
        category_maps,
        feature_names
    )


def calculate_lime_explanation(
    model,
    X_background,
    X_explain,
    y_explain,
    model_key,
    figures_dir,
    metrics_dir
):
    """
    Calcula una explicación local LIME para un caso Desertor.

    Como SVC tiene probability=False, se aplica softmax a los scores
    funcionales para producir puntuaciones relativas normalizadas.

    Estas puntuaciones no son probabilidades calibradas.
    """

    print("\n" + "-" * 70)
    print(f"LIME: MODELO {model_key.upper()}")
    print("-" * 70)

    (
        encoded_background,
        encoded_explain,
        categorical_features,
        categorical_names,
        category_maps,
        feature_names
    ) = _prepare_lime_data(
        X_background,
        X_explain
    )

    case_position = _select_desertor_case(
        X_explain,
        y_explain
    )

    instance = encoded_explain[case_position]

    class_names = [
        str(value)
        for value in model.classes_
    ]

    def lime_predict(encoded_data):
        """
        Reconstruye las variables originales y obtiene scores SVM.
        """

        encoded_data = np.asarray(encoded_data)

        if encoded_data.ndim == 1:
            encoded_data = encoded_data.reshape(1, -1)

        reconstructed = {}

        for position, feature in enumerate(feature_names):

            if position in categorical_features:

                mapping = category_maps[feature]

                codes = np.rint(
                    encoded_data[:, position]
                ).astype(int)

                values = []

                for code in codes:

                    if code not in mapping:
                        code = min(
                            max(code, 0),
                            len(mapping) - 1
                        )

                    values.append(mapping[code])

                reconstructed[feature] = values

            else:

                reconstructed[feature] = (
                    encoded_data[:, position].astype(float)
                )

        frame = pd.DataFrame(
            reconstructed,
            columns=feature_names
        )

        scores = np.asarray(
            model.decision_function(frame)
        )

        if scores.ndim == 1:
            scores = np.column_stack([
                -scores,
                scores
            ])

        return softmax(
            scores,
            axis=1
        )

    explainer = LimeTabularExplainer(
        training_data=encoded_background,
        feature_names=feature_names,
        class_names=class_names,
        categorical_features=categorical_features,
        categorical_names=categorical_names,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE
    )

    print(
        f"-> Generando explicación LIME con "
        f"{LIME_NUM_SAMPLES} perturbaciones..."
    )

    target_index = _get_target_class_index(model)

    # IMPORTANTE:
    # La API de LimeTabularExplainer utiliza predict_fn.
    # classifier_fn no es un argumento válido de explain_instance().
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=lime_predict,
        labels=(target_index,),
        num_features=min(
            LIME_NUM_FEATURES,
            len(feature_names)
        ),
        num_samples=LIME_NUM_SAMPLES
    )

    explanation_list = explanation.as_list(
        label=target_index
    )

    lime_df = pd.DataFrame(
        explanation_list,
        columns=[
            "Feature Condition",
            "LIME Weight"
        ]
    )

    lime_df["Absolute Weight"] = np.abs(
        lime_df["LIME Weight"]
    )

    lime_df = lime_df.sort_values(
        "Absolute Weight",
        ascending=False
    )

    _save_csv(
        lime_df,
        metrics_dir / f"lime_explanation_{model_key}.csv"
    )

    plot_df = lime_df.sort_values(
        "LIME Weight",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        plot_df["Feature Condition"],
        plot_df["LIME Weight"]
    )

    ax.axvline(
        0,
        linewidth=1
    )

    ax.set_title(
        f"Explicación local LIME - {model_key.upper()}\n"
        "Caso perteneciente a la clase Desertor",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Peso local del modelo sustituto"
    )

    ax.set_ylabel("Condición de la variable")

    ax.grid(
        axis="x",
        alpha=0.25
    )

    _save_figure(
        fig,
        figures_dir / f"lime_explanation_{model_key}.png"
    )

    print(
        "-> LIME finalizado. Las puntuaciones normalizadas son "
        "relativas y no probabilidades calibradas."
    )

    return lime_df


# =============================================================================
# 6. PARTIAL DEPENDENCE PLOT (PDP)
# =============================================================================

def calculate_pdp(
    model,
    X_explain,
    model_key,
    numeric_features,
    figures_dir,
    metrics_dir
):
    """
    Calcula PDP manualmente sobre el score funcional de Desertor.

    Para cada valor de una variable:
        1. Se reemplaza esa variable por el valor de la cuadrícula.
        2. Se conservan las demás variables del conjunto de referencia.
        3. Se calcula el score de Desertor.
        4. Se promedian los scores.

    No transforma los scores en probabilidades.
    """

    print("\n" + "-" * 70)
    print(f"PDP: MODELO {model_key.upper()}")
    print("-" * 70)

    selected_features = [
        feature
        for feature in PDP_PREFERRED_FEATURES
        if feature in numeric_features
        and feature in X_explain.columns
        and pd.api.types.is_numeric_dtype(X_explain[feature])
    ]

    if not selected_features:
        print(
            "-> No se encontraron variables numéricas disponibles para PDP."
        )
        return pd.DataFrame()

    reference_data = X_explain.sample(
        n=min(PDP_SAMPLE_SIZE, len(X_explain)),
        random_state=RANDOM_STATE
    ).copy()

    results = []

    fig, axes = plt.subplots(
        1,
        len(selected_features),
        figsize=(6 * len(selected_features), 4.8),
        squeeze=False
    )

    plotted_features = 0

    for position, feature in enumerate(selected_features):

        values = pd.to_numeric(
            X_explain[feature],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        grid = np.unique(
            np.quantile(
                values,
                np.linspace(
                    0.05,
                    0.95,
                    PDP_GRID_SIZE
                )
            )
        )

        mean_scores = []

        for value in grid:

            modified_data = reference_data.copy()
            modified_data[feature] = value

            scores = _get_target_decision_score(
                model,
                modified_data
            )

            mean_score = float(
                np.mean(scores)
            )

            mean_scores.append(mean_score)

            results.append({
                "Model": model_key,
                "Feature": feature,
                "Feature Value": value,
                "Mean Desertor Decision Score": mean_score,
                "Reference Sample Size": len(reference_data)
            })

        ax = axes[0, plotted_features]
        plotted_features += 1

        ax.plot(
            grid,
            mean_scores,
            marker="o",
            markersize=3
        )

        ax.set_title(
            feature,
            fontsize=11,
            fontweight="bold"
        )

        ax.set_xlabel(feature)

        ax.set_ylabel(
            "Score funcional medio de Desertor"
        )

        ax.grid(alpha=0.25)

    if plotted_features == 0:
        plt.close(fig)
        print("-> No fue posible construir los gráficos PDP.")
        return pd.DataFrame()

    for unused_position in range(
        plotted_features,
        len(selected_features)
    ):
        axes[0, unused_position].set_visible(False)

    fig.suptitle(
        f"Partial Dependence Plot - {model_key.upper()}",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()

    _save_figure(
        fig,
        figures_dir / f"pdp_{model_key}.png"
    )

    pdp_df = pd.DataFrame(results)

    _save_csv(
        pdp_df,
        metrics_dir / f"pdp_values_{model_key}.csv"
    )

    print(
        "-> PDP finalizado. Los gráficos describen cambios promedio "
        "en el score funcional, no efectos causales."
    )

    return pdp_df


# =============================================================================
# 7. FUNCIÓN INTEGRADORA
# =============================================================================

def run_model_interpretability(
    model,
    X_background,
    X_explain,
    y_explain,
    model_name,
    numeric_features,
    figures_dir,
    metrics_dir
):
    """
    Ejecuta SHAP, LIME y PDP para un modelo final.

    No modifica ni reentrena el estimador.
    """

    model_key = (
        model_name.lower()
        .replace("-", "")
        .replace(" ", "_")
    )

    _validate_inputs(
        model,
        X_background,
        X_explain
    )

    figures_dir = Path(figures_dir)
    metrics_dir = Path(metrics_dir)

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 80)
    print(f"EXPLICABILIDAD DEL MODELO {model_name.upper()}")
    print("=" * 80)

    shap_global, shap_local = calculate_shap_explanations(
        model=model,
        X_background=X_background,
        X_explain=X_explain,
        y_explain=y_explain,
        model_key=model_key,
        figures_dir=figures_dir,
        metrics_dir=metrics_dir
    )

    lime_results = calculate_lime_explanation(
        model=model,
        X_background=X_background,
        X_explain=X_explain,
        y_explain=y_explain,
        model_key=model_key,
        figures_dir=figures_dir,
        metrics_dir=metrics_dir
    )

    pdp_results = calculate_pdp(
        model=model,
        X_explain=X_explain,
        model_key=model_key,
        numeric_features=numeric_features,
        figures_dir=figures_dir,
        metrics_dir=metrics_dir
    )

    print("\nRESUMEN DE EXPLICABILIDAD:")
    print(f"-> SHAP global: {len(shap_global)} variables.")
    print(f"-> SHAP local: {len(shap_local)} variables.")
    print(f"-> LIME: {len(lime_results)} condiciones.")

    pdp_count = (
        pdp_results["Feature"].nunique()
        if not pdp_results.empty
        else 0
    )

    print(f"-> PDP: {pdp_count} variables.")

    print(
        "-> Los resultados son explicaciones del comportamiento "
        "de los modelos sobre el dataset sintético."
    )

    return {
        "shap_global": shap_global,
        "shap_local": shap_local,
        "lime": lime_results,
        "pdp": pdp_results
    }
