
"""
Módulo de Construcción de Pipelines y Preprocesadores SVM Multiclase.
=====================================================================
Implementa ocho configuraciones experimentales de preprocesamiento
para evaluar modelos SVM multiclase con kernels RBF y polinomial.

Las transformaciones se integran mediante ColumnTransformer y Pipeline
para evitar fuga de información durante el entrenamiento y evaluación.

Configuraciones:
----------------
1. StandardScaler + OneHotEncoder.
2. MinMaxScaler + OneHotEncoder.
3. RobustScaler + OneHotEncoder.
4. QuantileTransformer + OneHotEncoder.
5. MaxAbsScaler + OneHotEncoder.
6. PowerTransformer + OneHotEncoder.
7. Sin escalamiento numérico + OneHotEncoder (Baseline).
8. StandardScaler + OrdinalEncoder.

Nota:
-----
OrdinalEncoder asigna códigos enteros a las categorías y puede introducir
relaciones de orden artificiales. Se incluye como alternativa experimental.

Todas las transformaciones se ajustan exclusivamente con los datos
que recibe el método fit() del pipeline.

Machine Learning II - 2026-II
"""

import inspect
from numbers import Integral, Real
from typing import Any, Dict, List, Optional

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler
)

from sklearn.svm import SVC

from src.utils.config import CAT_COLS, NUM_COLS


# ============================================================
# 1. CONSTRUCCIÓN COMPATIBLE DE ONE HOT ENCODER
# ============================================================

def _make_onehot_encoder() -> OneHotEncoder:
    """
    Construye OneHotEncoder con compatibilidad entre versiones
    de Scikit-Learn.

    En versiones recientes se utiliza sparse_output=False.
    En versiones anteriores se utiliza sparse=False.

    handle_unknown="ignore" permite procesar categorías que no
    aparecieron durante el entrenamiento.

    Returns
    -------
    OneHotEncoder
        Codificador categórico que genera matrices densas.
    """

    parameters = inspect.signature(
        OneHotEncoder
    ).parameters

    if "sparse_output" in parameters:

        return OneHotEncoder(
            drop=None,
            handle_unknown="ignore",
            sparse_output=False
        )

    return OneHotEncoder(
        drop=None,
        handle_unknown="ignore",
        sparse=False
    )


# ============================================================
# 2. CATÁLOGO DE CONFIGURACIONES EXPERIMENTALES
# ============================================================

def get_normalization_scalers() -> Dict[str, Any]:
    """
    Retorna las ocho configuraciones experimentales.

    Las primeras siete utilizan OneHotEncoder para las variables
    categóricas y el transformador correspondiente para las numéricas.

    La octava utiliza StandardScaler para variables numéricas
    y OrdinalEncoder para variables categóricas.

    Returns
    -------
    Dict[str, Any]
        Diccionario con los nombres y transformadores de cada método.
    """

    return {

        # 1. Estandarización: media 0 y desviación estándar 1.
        "Standard Scaler": StandardScaler(),

        # 2. Escalamiento al intervalo [0, 1].
        "Min Max Scaler": MinMaxScaler(),

        # 3. Escalamiento robusto basado en mediana e IQR.
        "Robust Scaler": RobustScaler(),

        # 4. Transformación por cuantiles.
        "Quantile Transformer": QuantileTransformer(
            n_quantiles=50,
            random_state=42,
            output_distribution="uniform"
        ),

        # 5. Escalamiento por valor absoluto máximo.
        "Max Absolute Scaler": MaxAbsScaler(),

        # 6. Transformación de potencia Yeo-Johnson.
        "Power Transformer": PowerTransformer(
            method="yeo-johnson"
        ),

        # 7. Baseline: no aplica escalamiento numérico.
        "No Normalization (Baseline)": FunctionTransformer(
            validate=False
        ),

        # 8. StandardScaler numérico + OrdinalEncoder categórico.
        "Ordinal Encoder": OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    }


# ============================================================
# 3. VALIDACIÓN DE COLUMNAS
# ============================================================

def _validate_columns(
    num_cols: List[str],
    cat_cols: List[str]
) -> None:
    """
    Valida las listas de variables numéricas y categóricas.

    Comprueba que:
    - Ambas listas contengan columnas.
    - No existan nombres duplicados.
    - No existan columnas compartidas.
    - Los nombres sean cadenas no vacías.
    """

    if not num_cols:
        raise ValueError(
            "La lista de variables numéricas está vacía."
        )

    if not cat_cols:
        raise ValueError(
            "La lista de variables categóricas está vacía."
        )

    for col in num_cols + cat_cols:

        if not isinstance(col, str) or not col.strip():
            raise TypeError(
                "Todos los nombres de columnas deben ser "
                "cadenas no vacías."
            )

    if len(num_cols) != len(set(num_cols)):
        raise ValueError(
            "Existen columnas numéricas duplicadas."
        )

    if len(cat_cols) != len(set(cat_cols)):
        raise ValueError(
            "Existen columnas categóricas duplicadas."
        )

    overlapping_cols = set(num_cols).intersection(cat_cols)

    if overlapping_cols:
        raise ValueError(
            "Una misma columna no puede pertenecer simultáneamente "
            f"a las variables numéricas y categóricas: {overlapping_cols}"
        )


# ============================================================
# 4. CONSTRUCCIÓN DEL PREPROCESADOR
# ============================================================

def build_preprocessor(
    scaler: Any,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    is_ordinal: bool = False
) -> ColumnTransformer:
    """
    Construye un preprocesador independiente para variables numéricas
    y categóricas.

    Parameters
    ----------
    scaler : Any
        Transformador de la configuración experimental.

    num_cols : list of str, optional
        Variables numéricas. Por defecto utiliza NUM_COLS.

    cat_cols : list of str, optional
        Variables categóricas. Por defecto utiliza CAT_COLS.

    is_ordinal : bool, default=False
        Si es True, utiliza StandardScaler para variables numéricas
        y el OrdinalEncoder recibido para variables categóricas.

    Returns
    -------
    ColumnTransformer
        Preprocesador listo para integrarse en un Pipeline.

    Raises
    ------
    ValueError
        Si las columnas están vacías, duplicadas o se superponen.

    TypeError
        Si scaler no implementa fit y transform.
    """

    # --------------------------------------------------------
    # Valores predeterminados
    # --------------------------------------------------------

    if num_cols is None:
        num_cols = list(NUM_COLS)
    else:
        num_cols = list(num_cols)

    if cat_cols is None:
        cat_cols = list(CAT_COLS)
    else:
        cat_cols = list(cat_cols)

    # --------------------------------------------------------
    # Validación de columnas
    # --------------------------------------------------------

    _validate_columns(
        num_cols=num_cols,
        cat_cols=cat_cols
    )

    # --------------------------------------------------------
    # Validación del transformador
    # --------------------------------------------------------

    if not hasattr(scaler, "fit") or not hasattr(scaler, "transform"):
        raise TypeError(
            "scaler debe implementar los métodos fit() y transform()."
        )

    try:

        scaler_inst = clone(scaler)

    except Exception as exc:

        raise TypeError(
            "No fue posible clonar el transformador recibido. "
            "Verifica que sea compatible con Scikit-Learn."
        ) from exc

    # --------------------------------------------------------
    # Configuración ordinal
    # --------------------------------------------------------

    if is_ordinal:

        if not isinstance(scaler_inst, OrdinalEncoder):
            raise TypeError(
                "La configuración ordinal requiere recibir "
                "una instancia de OrdinalEncoder."
            )

        return ColumnTransformer(
            transformers=[

                (
                    "num",
                    StandardScaler(),
                    num_cols
                ),

                (
                    "cat",
                    scaler_inst,
                    cat_cols
                )

            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

    # --------------------------------------------------------
    # Configuraciones One-Hot
    # --------------------------------------------------------

    return ColumnTransformer(
        transformers=[

            (
                "num",
                scaler_inst,
                num_cols
            ),

            (
                "cat",
                _make_onehot_encoder(),
                cat_cols
            )

        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


# ============================================================
# 5. CONSTRUCCIÓN DEL PIPELINE SVM
# ============================================================

def build_svm_pipeline(
    scaler: Any,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    kernel: str = "rbf",
    degree: int = 3,
    coef0: float = 1.0,
    C: float = 1.0,
    class_weight: Optional[Any] = "balanced",
    probability: bool = False,
    cache_size: float = 1000,
    random_state: int = 42,
    is_ordinal: bool = False
) -> Pipeline:
    """
    Construye un pipeline integral de preprocesamiento y clasificación SVM.

    Parameters
    ----------
    scaler : Any
        Transformador correspondiente a la configuración experimental.

    num_cols : list of str, optional
        Variables numéricas.

    cat_cols : list of str, optional
        Variables categóricas.

    kernel : str, default="rbf"
        Kernel SVM. Valores permitidos: "rbf" y "poly".

    degree : int, default=3
        Grado del kernel polinomial.

    coef0 : float, default=1.0
        Término independiente del kernel polinomial.

    C : float, default=1.0
        Parámetro de regularización del clasificador.

    class_weight : str, dict o None, default="balanced"
        Ponderación de las clases.

    probability : bool, default=False
        Activa la estimación de probabilidades.

    cache_size : float, default=1000
        Tamaño de caché del kernel en MB.

    random_state : int, default=42
        Semilla de reproducibilidad.

    is_ordinal : bool, default=False
        Activa la configuración StandardScaler + OrdinalEncoder.

    Returns
    -------
    Pipeline
        Pipeline compuesto por preprocesador y clasificador SVC.

    Raises
    ------
    ValueError
        Si los parámetros del clasificador no son válidos.

    TypeError
        Si los parámetros tienen tipos incorrectos.
    """

    # --------------------------------------------------------
    # Validar kernel
    # --------------------------------------------------------

    if not isinstance(kernel, str):
        raise TypeError(
            "kernel debe ser una cadena de texto."
        )

    kernel = kernel.lower()

    if kernel not in ("rbf", "poly"):
        raise ValueError(
            "Kernel no válido. Utiliza 'rbf' o 'poly'."
        )

    # --------------------------------------------------------
    # Validar degree
    # --------------------------------------------------------

    if isinstance(degree, bool) or not isinstance(degree, Integral):
        raise TypeError(
            "degree debe ser un número entero."
        )

    if degree < 1:
        raise ValueError(
            "degree debe ser un entero positivo."
        )

    # --------------------------------------------------------
    # Validar C
    # --------------------------------------------------------

    if isinstance(C, bool) or not isinstance(C, Real):
        raise TypeError(
            "C debe ser un número real."
        )

    if not float(C) > 0:
        raise ValueError(
            "El parámetro C debe ser mayor que cero."
        )

    # --------------------------------------------------------
    # Validar coef0
    # --------------------------------------------------------

    if isinstance(coef0, bool) or not isinstance(coef0, Real):
        raise TypeError(
            "coef0 debe ser un número real."
        )

    # --------------------------------------------------------
    # Validar cache_size
    # --------------------------------------------------------

    if (
        isinstance(cache_size, bool)
        or not isinstance(cache_size, Real)
    ):
        raise TypeError(
            "cache_size debe ser un número real."
        )

    if cache_size <= 0:
        raise ValueError(
            "cache_size debe ser mayor que cero."
        )

    # --------------------------------------------------------
    # Validar probability
    # --------------------------------------------------------

    if not isinstance(probability, bool):
        raise TypeError(
            "probability debe ser True o False."
        )

    # --------------------------------------------------------
    # Validar random_state
    # --------------------------------------------------------

    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, Integral)
    ):
        raise TypeError(
            "random_state debe ser un entero o None."
        )

    # --------------------------------------------------------
    # Construir preprocesador
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        scaler=scaler,
        num_cols=num_cols,
        cat_cols=cat_cols,
        is_ordinal=is_ordinal
    )

    # --------------------------------------------------------
    # Construir clasificador SVM
    # --------------------------------------------------------

    classifier = SVC(
        kernel=kernel,
        degree=int(degree),
        coef0=float(coef0),
        C=float(C),
        class_weight=class_weight,
        probability=probability,
        cache_size=float(cache_size),
        random_state=random_state
    )

    # --------------------------------------------------------
    # Ensamblar pipeline
    # --------------------------------------------------------

    svm_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    return svm_pipeline
