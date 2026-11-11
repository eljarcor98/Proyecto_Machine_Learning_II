"""
Módulo de Construcción de Pipelines y Preprocesadores SVM Multiclase.
=====================================================================
Implementa la arquitectura de preprocesamiento desacoplado y ensamblaje de pipelines
en Scikit-Learn para garantizar la ausencia total de fuga de información (data leakage).

Características:
----------------
1. Catálogo exhaustivo de 8 técnicas de normalización y transformación empírica:
   - StandardScaler: Estandarización a media cero y varianza unitaria (z = (x - mu)/sigma).
   - MinMaxScaler: Re-escalamiento lineal acotado en el intervalo [0, 1].
   - RobustScaler: Escalamiento robusto ante outliers basado en la mediana y el IQR.
   - QuantileTransformer: Mapeo no lineal hacia distribución uniforme basado en cuantiles empíricos.
   - PowerTransformer: Transformación estabilizadora de varianza y corrección de asimetría (Yeo-Johnson).
   - MaxAbsScaler: Escalamiento proporcional al valor absoluto máximo sin desplazar la media.
   - OrdinalEncoder: Asignación secuencial de rangos discretos enteros.
   - No Normalization (Baseline): Identidad funcional para cuantificar el impacto del no escalamiento.
2. Construcción de ColumnTransformer diferenciando flujo numérico y categórico.
3. Ensamblaje con clasificador SVC de margen blando (Soft-Margin SVM) con kernel RBF.

Autor: Arnold Santiago Torres
Machine Learning II - 2026-II
"""

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


def get_normalization_scalers() -> Dict[str, Any]:
    """
    Retorna el catálogo oficial de los 8 estimadores de transformación y escalamiento
    para su contraste empírico frente a la condición sin normalizar (Baseline).

    Retorna:
    --------
    dict de {str: transformer_instance}:
        Diccionario con nombres legibles y las instancias de transformadores preconfiguradas.
    """
    return {
        'Standard Scaler': StandardScaler(),
        'Min Max Scaler': MinMaxScaler(),
        'Robust Scaler': RobustScaler(),
        'Quantile Transformer': QuantileTransformer(n_quantiles=50, random_state=42, output_distribution='uniform'),
        'Max Absolute Scaler': MaxAbsScaler(),
        'Power Transformer': PowerTransformer(method='yeo-johnson'),
        'Ordinal Encoder': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        'No Normalization (Baseline)': FunctionTransformer(validate=False)
    }


def build_preprocessor(
    scaler: Any,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    is_ordinal: bool = False
) -> ColumnTransformer:
    """
    Construye un ColumnTransformer de Scikit-Learn que aísla las transformaciones
    numéricas y categóricas para evitar fuga de información durante el ajuste.

    Parámetros:
    -----------
    scaler : Any
        Instancia de transformador de Scikit-Learn a aplicar sobre las columnas numéricas.
    num_cols : list of str, opcional
        Lista de nombres de variables numéricas. Por defecto NUM_COLS.
    cat_cols : list of str, opcional
        Lista de nombres de variables categóricas. Por defecto CAT_COLS.
    is_ordinal : bool, por defecto False
        Indica si se evalúa la variante de OrdinalEncoder sobre las características categóricas.

    Retorna:
    --------
    ColumnTransformer:
        Preprocesador configurado listo para integrarse en un Pipeline.
    """
    if num_cols is None:
        num_cols = NUM_COLS
    if cat_cols is None:
        cat_cols = CAT_COLS

    # Clonar el transformador para asegurar que cada pipeline opere con una instancia independiente
    try:
        scaler_inst = clone(scaler)
    except Exception:
        scaler_inst = scaler

    # Configuración especial si se evalúa OrdinalEncoder como método de transformación
    if is_ordinal:
        return ColumnTransformer(
            transformers=[
                ('num', scaler_inst, cat_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
            ]
        )

    # Flujo estándar: escalador sobre variables continuas y One-Hot Encoding sobre categóricas
    return ColumnTransformer(
        transformers=[
            ('num', scaler_inst, num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )


def build_svm_pipeline(
    scaler: Any,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    kernel: str = 'rbf',
    C: float = 1.0,
    class_weight: str = 'balanced',
    probability: bool = False,
    cache_size: int = 1000,
    random_state: int = 42,
    is_ordinal: bool = False
) -> Pipeline:
    """
    Ensambla un Pipeline integral de Scikit-Learn acoplando el preprocesador de columnas
    con el estimador clasificador de Máquinas de Soporte Vectorial (SVC).

    Parámetros:
    -----------
    scaler : Any
        Instancia del método de normalización a evaluar.
    num_cols : list of str, opcional
        Variables numéricas. Por defecto NUM_COLS.
    cat_cols : list of str, opcional
        Variables categóricas. Por defecto CAT_COLS.
    kernel : str, por defecto 'rbf'
        Función de núcleo para mapeo no lineal al espacio de Hilbert (Radial Basis Function).
    C : float, por defecto 1.0
        Parámetro de regularización y penalización por violaciones del margen blando.
    class_weight : str, por defecto 'balanced'
        Ponderación automática inversamente proporcional a las frecuencias de las clases
        para mitigar el desbalance muestral multiclase.
    probability : bool, por defecto False
        Determina si se computa la estimación de probabilidades mediante escalamiento de Platt.
        Se mantiene en False por rendimiento y compatibilidad con versiones recientes de Scikit-Learn.
    cache_size : int, por defecto 1000
        Tamaño de la memoria caché del kernel en megabytes (RAM) para acelerar el cálculo cuadrático.
    random_state : int, por defecto 42
        Semilla para asegurar reproducibilidad.
    is_ordinal : bool, por defecto False
        Bandera para configuración de codificación ordinal.

    Retorna:
    --------
    Pipeline:
        Flujo de procesamiento reproducible acoplado: steps=[('preprocessor', ...), ('classifier', ...)]
    """
    # 1. Construir el transformador de columnas
    preprocessor = build_preprocessor(
        scaler=scaler,
        num_cols=num_cols,
        cat_cols=cat_cols,
        is_ordinal=is_ordinal
    )

    # 2. Configurar el clasificador SVM
    classifier = SVC(
        kernel=kernel,
        C=C,
        class_weight=class_weight,
        probability=probability,
        cache_size=cache_size,
        random_state=random_state
    )

    # 3. Ensamblar el pipeline secuencial
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
