"""
Submódulo de utilidades del proyecto.
"""

from src.utils.config import (
    PROJECT_ROOT,
    DATA_RAW_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    DEFAULT_DATA_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_METRICS_FILE,
    NUM_COLS,
    CAT_COLS,
    ALL_FEATURE_COLS,
    CLASSES,
    CLASS_MAP,
    INVERSE_CLASS_MAP,
    CLASS_PALETTE,
    SCATTER_PALETTE
)

from src.utils.metrics import (
    compute_classification_metrics,
    build_summary_comparison_table,
    export_summary_to_csv,
    compute_subgroup_recall
)

__all__ = [
    'PROJECT_ROOT',
    'DATA_RAW_DIR',
    'FIGURES_DIR',
    'METRICS_DIR',
    'MODELS_DIR',
    'DEFAULT_DATA_FILE',
    'DEFAULT_MODEL_FILE',
    'DEFAULT_METRICS_FILE',
    'NUM_COLS',
    'CAT_COLS',
    'ALL_FEATURE_COLS',
    'CLASSES',
    'CLASS_MAP',
    'INVERSE_CLASS_MAP',
    'CLASS_PALETTE',
    'SCATTER_PALETTE',
    'compute_classification_metrics',
    'build_summary_comparison_table',
    'export_summary_to_csv',
    'compute_subgroup_recall'
]
