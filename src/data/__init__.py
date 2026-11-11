"""
Submódulo de carga y procesamiento de datos.
"""

from src.data.preprocessing import (
    load_data,
    create_spadies_target,
    compute_outlier_diagnostics,
    get_train_test_split
)

__all__ = [
    'load_data',
    'create_spadies_target',
    'compute_outlier_diagnostics',
    'get_train_test_split'
]
