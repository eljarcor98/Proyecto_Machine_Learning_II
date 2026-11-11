"""
Submódulo de generación de gráficos y visualizaciones.
"""

from src.visualization.plots import (
    plot_eda_distributions,
    plot_key_variables_distribution,
    plot_scaler_impact,
    plot_normalization_comparison,
    plot_permutation_importance,
    plot_pca_hyperplane,
    plot_confusion_matrices,
    plot_decision_boundary_top2,
    plot_decision_function_distribution
)

__all__ = [
    'plot_eda_distributions',
    'plot_key_variables_distribution',
    'plot_scaler_impact',
    'plot_normalization_comparison',
    'plot_permutation_importance',
    'plot_pca_hyperplane',
    'plot_confusion_matrices',
    'plot_decision_boundary_top2',
    'plot_decision_function_distribution'
]
