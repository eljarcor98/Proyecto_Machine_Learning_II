"""
Experimento SVM: Predicción de Deserción Estudiantil sobre Dataset Sintético (11,500 Estudiantes)
- Clasificación Multiclase SPADIES 3.0 (Graduado, En curso, Desertor)
- EDA Completo de Variables Socioeconómicas y Académicas (Outliers IQR, Spearman vs Pearson)
- Flujo Reproducible Sin Fuga de Datos (Scikit-Learn Pipeline + ColumnTransformer)
- Evaluación de 8 Normalizaciones vs Baseline sobre 50 particiones independientes
- Importancia de Variables en la Clasificación (Permutation Importance sobre SVM)
- Visualización del Hiperplano 2D (PCA) y Análisis de Desempeño por Subgrupo

Dataset: estudiantes_sinteticos_11500.csv
Autor: Arnold Santiago Torres
Machine Learning II - 2026-II
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed, dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, ConfusionMatrixDisplay
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
    QuantileTransformer, PowerTransformer, OrdinalEncoder, OneHotEncoder, FunctionTransformer
)
from scipy.stats import ttest_ind, skew, kurtosis

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'Arial'

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Archivos", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. CARGA DE DATOS Y FORMULACIÓN SPADIES 3.0 (3 CLASES)
# ==========================================
print("=========================================================")
print("1. ANÁLISIS EXPLORATORIO DE DATOS (EDA) - DATASET SINTÉTICO (11,500 ESTUDIANTES)")
print("=========================================================", flush=True)

data_path = os.path.join(os.path.dirname(__file__), "Archivos", "estudiantes_sinteticos_11500.csv")
if not os.path.exists(data_path):
    data_path = "estudiantes_sinteticos_11500.csv"

df = pd.read_csv(data_path)

print(f"Dimensiones del dataset: {df.shape[0]} estudiantes y {df.shape[1]} variables.", flush=True)
print(f"Valores nulos totales en el dataset: {df.isnull().sum().sum()}", flush=True)

# ==============================================================================
# FORMULACIÓN DE LA VARIABLE OBJETIVO MULTICLASE ('estado_final') - SPADIES 3.0
# 3 Clases: 'Graduado' (0), 'En curso' (1), 'Desertor' (2)
# ==============================================================================
duracion_map = {'Tecnico profesional': 4, 'Tecnologico': 6, 'Universitario': 10}
duracion = df['nivel_formacion'].map(duracion_map).fillna(8)

riesgo = (
    (df['promedio_academico'] < 3.0).astype(int) * 2.5 +
    (df['materias_reprobadas'] >= 2).astype(int) * 2.5 +
    (df['puntaje_icfes_percentil'] < 40).astype(int) * 1.0 +
    (df['trabaja_mientras_estudia'] == 1).astype(int) * 1.0 +
    (df['beneficiario_beca'] == 0).astype(int) * 0.5 +
    (df['estrato_socioeconomico'] <= 2).astype(int) * 0.5
)

is_desertor = (riesgo >= 3.5)
is_graduado = (~is_desertor) & (df['semestre_cursado'] >= duracion) & (df['promedio_academico'] >= 3.2)

df['estado_final'] = np.where(is_desertor, 'Desertor', np.where(is_graduado, 'Graduado', 'En curso'))

cat_cols = ['sexo', 'estado_civil', 'zona_residencia', 'departamento',
            'nivel_educativo_padre', 'nivel_educativo_madre', 'sector_ies',
            'nivel_formacion', 'metodologia', 'area_conocimiento']

num_cols = ['edad_ingreso', 'promedio_academico', 'materias_reprobadas',
            'distancia_hogar_ies_km', 'puntaje_icfes_percentil', 'semestre_cursado',
            'estrato_socioeconomico', 'anio_registro', 'trabaja_mientras_estudia',
            'beneficiario_icetex', 'beneficiario_beca']

X_raw = df[num_cols + cat_cols]
clases = ['Graduado', 'En curso', 'Desertor']
class_map = {c: i for i, c in enumerate(clases)}
y = df['estado_final'].map(class_map).values

print("\nDistribución de la Variable Objetivo 'estado_final' (3 Clases SPADIES):", flush=True)
print(pd.Series(df['estado_final']).value_counts())
print(pd.Series(df['estado_final']).value_counts(normalize=True) * 100)

# DIAGNÓSTICO ESTADÍSTICO DE OUTLIERS Y ASIMETRÍA EN VARIABLES CONTINUAS
print("\n" + "="*80)
print("DIAGNÓSTICO ESTADÍSTICO DE VARIABLES CONTINUAS (OUTLIERS E IQR)")
print("="*80, flush=True)
diag_rows = []
for col in num_cols:
    data_col = df[col].dropna()
    q25, q75 = np.percentile(data_col, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    n_outliers = np.sum((data_col < lower_bound) | (data_col > upper_bound))
    pct_outliers = (n_outliers / len(data_col)) * 100
    sk = skew(data_col)
    diag_rows.append({
        'Variable': col,
        'Media': data_col.mean(),
        'Mediana': data_col.median(),
        'IQR': iqr,
        'Outliers Count': n_outliers,
        'Outliers %': pct_outliers,
        'Skewness': sk
    })

diag_df = pd.DataFrame(diag_rows)
print(diag_df.to_string(index=False, formatters={
    'Media': '{:.2f}'.format,
    'Mediana': '{:.2f}'.format,
    'IQR': '{:.2f}'.format,
    'Outliers %': '{:.2f}%'.format,
    'Skewness': '{:.2f}'.format
}), flush=True)

# FIGURA 1: EDA - Distribución de 3 Clases y Correlaciones de Pearson vs Spearman
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.countplot(data=df, x='estado_final', order=clases, palette='Set2', ax=axes[0])
axes[0].set_title('Distribución de Estudiantes (SPADIES 3.0)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Estado Académico Final', fontsize=11)
axes[0].set_ylabel('Número de Estudiantes', fontsize=11)
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)',
                     (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', fontsize=10, color='white', fontweight='bold')

des_binary = (df['estado_final'] == 'Desertor').astype(int)
corr_p = df[num_cols].apply(lambda col: col.corr(des_binary, method='pearson')).sort_values()
colors_p = ['#d95f02' if c > 0 else '#7570b3' for c in corr_p.values]
axes[1].barh(corr_p.index, corr_p.values, color=colors_p)
axes[1].set_title('Correlación de Pearson con Deserción', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Coeficiente Pearson', fontsize=11)
axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)

corr_s = df[num_cols].apply(lambda col: col.corr(des_binary, method='spearman')).sort_values()
colors_s = ['#1b9e77' if c > 0 else '#e7298a' for c in corr_s.values]
axes[2].barh(corr_s.index, corr_s.values, color=colors_s)
axes[2].set_title('Correlación de Spearman (Monótona)', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Coeficiente Spearman', fontsize=11)
axes[2].axvline(0, color='black', linestyle='--', linewidth=0.8)

plt.tight_layout()
fig_eda_path = os.path.join(OUTPUT_DIR, "eda_distribucion_correlaciones.png")
plt.savefig(fig_eda_path, dpi=300)
plt.close()
print(f"[OK] Gráfico EDA (Distribución 3 Clases + Pearson/Spearman) guardado en: {fig_eda_path}", flush=True)

# FIGURA 2: Distribución de variables clave por estado final
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(data=df, x='estado_final', y='promedio_academico', order=clases, palette='Set1', ax=axes[0])
axes[0].set_title('Promedio Académico por Estado Académico', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Estado del Estudiante', fontsize=12)
axes[0].set_ylabel('Promedio Académico (0.0 - 5.0)', fontsize=12)

sns.boxplot(data=df, x='estado_final', y='materias_reprobadas', order=clases, palette='Set1', ax=axes[1])
axes[1].set_title('Materias Reprobadas por Estado Académico', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Estado del Estudiante', fontsize=12)
axes[1].set_ylabel('Materias Reprobadas', fontsize=12)

plt.tight_layout()
fig_box_path = os.path.join(OUTPUT_DIR, "eda_variables_clave.png")
plt.savefig(fig_box_path, dpi=300)
plt.close()
print(f"[OK] Gráfico de variables clave guardado en: {fig_box_path}", flush=True)

# FIGURA ADICIONAL EDA: Demostración del impacto de escaladores en variable continua
sample_feature = df[['distancia_hogar_ies_km']].values
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

sns.histplot(sample_feature, kde=True, ax=axes[0, 0], color='gray')
axes[0, 0].set_title('1. Original (Raw)', fontweight='bold')

sns.histplot(StandardScaler().fit_transform(sample_feature), kde=True, ax=axes[0, 1], color='blue')
axes[0, 1].set_title('2. Standard Scaler (Z-score)', fontweight='bold')

sns.histplot(MinMaxScaler().fit_transform(sample_feature), kde=True, ax=axes[0, 2], color='green')
axes[0, 2].set_title('3. Min Max Scaler (0 a 1)', fontweight='bold')

sns.histplot(RobustScaler().fit_transform(sample_feature), kde=True, ax=axes[1, 0], color='purple')
axes[1, 0].set_title('4. Robust Scaler (Mediana e IQR)', fontweight='bold')

sns.histplot(QuantileTransformer(n_quantiles=50, random_state=42).fit_transform(sample_feature), kde=True, ax=axes[1, 1], color='orange')
axes[1, 1].set_title('5. Quantile Transformer (Uniforme)', fontweight='bold')

sns.histplot(PowerTransformer().fit_transform(sample_feature), kde=True, ax=axes[1, 2], color='red')
axes[1, 2].set_title('6. Power Transformer (Yeo-Johnson)', fontweight='bold')

plt.tight_layout()
fig_scalers_demo = os.path.join(OUTPUT_DIR, "eda_efecto_escaladores.png")
plt.savefig(fig_scalers_demo, dpi=300)
plt.close()
print(f"[OK] Gráfico de demostración del efecto de escaladores guardado en: {fig_scalers_demo}", flush=True)


# ==========================================
# 2. FLUJO REPRODUCIBLE SIN FUGA DE DATOS (PIPELINE MULTICLASE)
# ==========================================
print("\n=========================================================")
print("2. FLUJO REPRODUCIBLE SIN FUGA DE DATOS (PIPELINES + COLUMNTRANSFORMER)")
print("   (Evaluación exhaustiva multiclase sobre 50 particiones independientes)")
print("=========================================================", flush=True)

normalization_methods = {
    'Standard Scaler': StandardScaler(),
    'Min Max Scaler': MinMaxScaler(),
    'Robust Scaler': RobustScaler(),
    'Quantile Transformer': QuantileTransformer(n_quantiles=50, random_state=42, output_distribution='uniform'),
    'Max Absolute Scaler': MaxAbsScaler(),
    'Power Transformer': PowerTransformer(method='yeo-johnson'),
    'Ordinal Encoder': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    'No Normalization (Baseline)': FunctionTransformer(validate=False)
}

N_TRIALS = 50
N_JOBS = -1

def evaluate_single_trial(trial_idx):
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42 + trial_idx, stratify=y
    )
    
    trial_scores_weighted = {}
    trial_scores_macro = {}
    trial_scores_desertor = {}
    
    idx_des = class_map['Desertor']
    
    for method_name, num_scaler in normalization_methods.items():
        try:
            num_scaler_inst = type(num_scaler)(**num_scaler.get_params()) if hasattr(num_scaler, 'get_params') else num_scaler

            if method_name == 'Ordinal Encoder':
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', num_scaler_inst, cat_cols),
                        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
                    ]
                )
            else:
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', num_scaler_inst, num_cols),
                        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
                    ]
                )

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', SVC(kernel='rbf', C=1.0, class_weight='balanced', cache_size=1000, random_state=42))
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            
            trial_scores_weighted[method_name] = f1_score(y_test, preds, average='weighted')
            trial_scores_macro[method_name] = f1_score(y_test, preds, average='macro')
            trial_scores_desertor[method_name] = f1_score(y_test, preds, labels=[idx_des], average='macro')
        except Exception as e:
            trial_scores_weighted[method_name] = np.nan
            trial_scores_macro[method_name] = np.nan
            trial_scores_desertor[method_name] = np.nan
            
    return trial_scores_weighted, trial_scores_macro, trial_scores_desertor

print(f"Iniciando {N_TRIALS} iteraciones de Pipelines independientes (n_jobs={N_JOBS})...", flush=True)
parallel_results = Parallel(n_jobs=N_JOBS)(
    delayed(evaluate_single_trial)(t) for t in range(N_TRIALS)
)
print("Iteraciones completadas exitosamente.", flush=True)

results_weighted = {method: [] for method in normalization_methods.keys()}
results_macro = {method: [] for method in normalization_methods.keys()}
results_desertor = {method: [] for method in normalization_methods.keys()}

for res_wei, res_mac, res_des in parallel_results:
    for method_name in normalization_methods.keys():
        results_weighted[method_name].append(res_wei[method_name])
        results_macro[method_name].append(res_mac[method_name])
        results_desertor[method_name].append(res_des[method_name])

baseline_scores_wei = results_weighted['No Normalization (Baseline)']
summary_rows = []

for method_name in normalization_methods.keys():
    scores_wei = results_weighted[method_name]
    scores_mac = results_macro[method_name]
    scores_des = results_desertor[method_name]
    
    mean_f1_wei = np.mean(scores_wei)
    std_f1_wei = np.std(scores_wei)
    mean_f1_mac = np.mean(scores_mac)
    mean_f1_des = np.mean(scores_des)
    
    if method_name != 'No Normalization (Baseline)':
        t_stat, p_val = ttest_ind(scores_wei, baseline_scores_wei, equal_var=False)
        delta = mean_f1_wei - np.mean(baseline_scores_wei)
    else:
        t_stat, p_val, delta = 0.0, 1.0, 0.0
        
    summary_rows.append({
        'Normalization Method': method_name,
        'Average F1 (Weighted)': mean_f1_wei,
        'Average F1 (Macro)': mean_f1_mac,
        'Average F1 (Desertor)': mean_f1_des,
        'Std Dev': std_f1_wei,
        'F1 Delta vs Baseline': delta,
        'T-Statistic': t_stat,
        'P-value vs Baseline': p_val
    })

summary_df = pd.DataFrame(summary_rows).sort_values(by='Average F1 (Weighted)', ascending=False)

csv_output_path = os.path.join(os.path.dirname(__file__), "Archivos", "resultados_svm_normalizacion.csv")
summary_df.to_csv(csv_output_path, index=False)
print(f"[OK] Resultados multiclase de 50 trials guardados en CSV: {csv_output_path}", flush=True)

print("\n" + "="*100)
print("TABLA COMPARATIVA: PROMEDIO DE F1-SCORE MULTICLASE (50 TRIALS SIN FUGA DE DATOS)")
print("="*100)
print(summary_df.to_string(index=False, formatters={
    'Average F1 (Weighted)': '{:.4f}'.format,
    'Average F1 (Macro)': '{:.4f}'.format,
    'Average F1 (Desertor)': '{:.4f}'.format,
    'Std Dev': '{:.4f}'.format,
    'F1 Delta vs Baseline': '{:+.4f}'.format,
    'T-Statistic': '{:.3f}'.format,
    'P-value vs Baseline': '{:.6f}'.format
}), flush=True)

# FIGURA 3: Comparación de F1-Score Ponderado (Normalizadores vs Baseline)
plt.figure(figsize=(12, 6))
colors_bar = ['#d95f02' if 'Baseline' in name else '#1b9e77' for name in summary_df['Normalization Method']]
barplot = sns.barplot(
    data=summary_df,
    x='Average F1 (Weighted)',
    y='Normalization Method',
    palette=colors_bar
)

plt.title(f'Evaluación de Pipelines SVM Multiclase (F1-Score Ponderado, {N_TRIALS} Trials)', fontsize=14, fontweight='bold')
plt.xlabel('F1-Score Ponderado Promedio (3 Clases)', fontsize=12)
plt.ylabel('Método de Normalización en Pipeline', fontsize=12)
plt.xlim(0.40, 1.0)

for p in barplot.patches:
    width = p.get_width()
    barplot.annotate(f'{width:.4f}',
                     (width + 0.005, p.get_y() + p.get_height() / 2.),
                     ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
fig_baseline_path = os.path.join(OUTPUT_DIR, "svm_normalizacion_vs_baseline.png")
plt.savefig(fig_baseline_path, dpi=300)
plt.close()
print(f"[OK] Gráfico comparativo de baseline guardado en: {fig_baseline_path}", flush=True)


# ==========================================
# 3. IMPORTANCIA DE VARIABLES EN PIPELINE
# ==========================================
print("\n=========================================================")
print("3. IMPORTANCIA DE VARIABLES TOMADAS POR EL MODELO SVM MULTICLASE")
print("   (Permutation Importance con n_jobs=2)")
print("=========================================================", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

best_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

best_pipeline = Pipeline(steps=[
    ('preprocessor', best_preprocessor),
    ('classifier', SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, cache_size=1000, random_state=42))
])

best_pipeline.fit(X_train, y_train)

# Exportar artefacto de modelo reproducible
model_artifact_path = os.path.join(os.path.dirname(__file__), "Archivos", "svm_pipeline_reproducible.joblib")
dump(best_pipeline, model_artifact_path)
print(f"[OK] Pipeline completo serializado guardado en: {model_artifact_path}", flush=True)

perm_importance = permutation_importance(
    best_pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
)

sorted_importances_idx = perm_importance.importances_mean.argsort()[::-1]
feature_names = X_raw.columns

imp_df = pd.DataFrame({
    'Feature': feature_names[sorted_importances_idx],
    'Importance_Mean': perm_importance.importances_mean[sorted_importances_idx],
    'Importance_Std': perm_importance.importances_std[sorted_importances_idx]
})

print("\nTop 10 Variables más influyentes para el Pipeline SVM Multiclase:", flush=True)
print(imp_df.head(10).to_string(index=False), flush=True)

# FIGURA 4: Importancia de Variables
plt.figure(figsize=(10, 8))
top_15_imp = imp_df.head(15)
plt.barh(top_15_imp['Feature'][::-1], top_15_imp['Importance_Mean'][::-1], color='#2b5c8f', xerr=top_15_imp['Importance_Std'][::-1])
plt.title('Importancia de Variables en Pipeline SVM (Permutation Importance Multiclase)', fontsize=14, fontweight='bold')
plt.xlabel('Disminución Promedio en F1-Score Ponderado al Permutar Variable', fontsize=12)
plt.tight_layout()
fig_imp_path = os.path.join(OUTPUT_DIR, "svm_importancia_variables.png")
plt.savefig(fig_imp_path, dpi=300)
plt.close()
print(f"[OK] Gráfico de importancia de variables guardado en: {fig_imp_path}", flush=True)


# ==========================================
# 4. VISUALIZACIÓN DEL HIPERPLANO MULTICLASE (PCA 2D)
# ==========================================
print("\n=========================================================")
print("4. VISUALIZACIÓN DEL HIPERPLANO MULTICLASE SVM (PCA 2D)")
print("=========================================================", flush=True)

X_train_trans = best_pipeline.named_steps['preprocessor'].transform(X_train)
X_test_trans = best_pipeline.named_steps['preprocessor'].transform(X_test)

pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_trans)
X_test_pca = pca.transform(X_test_trans)

svm_pca = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)
svm_pca.fit(X_train_pca, y_train)

x_min, x_max = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
y_min, y_max = X_train_pca[:, 1].min() - 0.5, X_train_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))

Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(11, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set2')

sample_idx = np.random.choice(len(y_test), size=1000, replace=False)
X_sample_pca = X_test_pca[sample_idx]
y_sample = y_test[sample_idx]

scatter_colors = ['#1b9e77', '#d95f02', '#7570b3']
for idx_c, label_c in enumerate(clases):
    plt.scatter(X_sample_pca[y_sample == idx_c, 0], X_sample_pca[y_sample == idx_c, 1],
                c=scatter_colors[idx_c], label=label_c, alpha=0.6, edgecolors='k', s=30)

sv = svm_pca.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], s=80, facecolors='none', edgecolors='black', linewidths=1.0,
            label=f'Vectores de Soporte ({len(sv)} puntos)')

plt.title('Fronteras de Decisión SVM Multiclase (Kernel RBF en PCA 2D)', fontsize=14, fontweight='bold')
plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Varianza)', fontsize=12)
plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Varianza)', fontsize=12)
plt.legend(loc='upper right', frameon=True)

plt.tight_layout()
fig_hyperplane_path = os.path.join(OUTPUT_DIR, "svm_hiperplano_pca_2d.png")
plt.savefig(fig_hyperplane_path, dpi=300)
plt.close()
print(f"[OK] Gráfico del Hiperplano PCA Multiclase guardado en: {fig_hyperplane_path}", flush=True)


# ==========================================
# 5. CLASIFICACIÓN DETALLADA Y DESEMPEÑO POR SUBGRUPO
# ==========================================
print("\n=========================================================")
print("5. ANÁLISIS DE CLASIFICACIÓN DE ESTUDIANTES (TEST SET = 2,300 ESTUDIANTES)")
print("=========================================================", flush=True)

y_pred = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("\nREPORTE DE CLASIFICACIÓN DETALLADO (3 CLASES):", flush=True)
print(classification_report(y_test, y_pred, target_names=clases, digits=4), flush=True)

acc = accuracy_score(y_test, y_pred)
f1_wei = f1_score(y_test, y_pred, average='weighted')
f1_mac = f1_score(y_test, y_pred, average='macro')
rec_des = recall_score(y_test, y_pred, labels=[class_map['Desertor']], average='macro')

print(f"Accuracy Total:       {acc:.4f}", flush=True)
print(f"F1-Score Ponderado:  {f1_wei:.4f}", flush=True)
print(f"F1-Score Macro:      {f1_mac:.4f}", flush=True)
print(f"Recall (Desertor):   {rec_des:.4f}", flush=True)

# FIGURA 5: Matriz de Confusión 3x3
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=clases, yticklabels=clases, annot_kws={'size': 13, 'weight': 'bold'})
axes[0].set_title('Matriz de Confusión (Conteos Absolutos)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicción del Modelo', fontsize=12)
axes[0].set_ylabel('Estado Real', fontsize=12)

cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=axes[1],
            xticklabels=clases, yticklabels=clases, annot_kws={'size': 13, 'weight': 'bold'})
axes[1].set_title('Matriz de Confusión Normalizada por Clase Real (Recall)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicción del Modelo', fontsize=12)
axes[1].set_ylabel('Estado Real', fontsize=12)

plt.tight_layout()
fig_eval_path = os.path.join(OUTPUT_DIR, "svm_matriz_confusion_roc.png")
plt.savefig(fig_eval_path, dpi=300)
plt.close()
print(f"[OK] Gráfico de Matriz de Confusión Multiclase guardado en: {fig_eval_path}", flush=True)

# DESGLOSE DE RECALL DE DESERTOR POR SUBGRUPO (Auditoría de Sesgo)
print("\n" + "="*80)
print("DESGLOSE DE RECALL (DESERTOR) POR SUBGRUPO DE ESTUDIANTES")
print("="*80, flush=True)

E = X_test.copy()
E['real'] = y_test
E['pred'] = y_pred
idx_des = class_map['Desertor']

subgroup_rows = []
for col in ['sexo', 'zona_residencia', 'sector_ies', 'metodologia']:
    for grupo, sub in E[E['real'] == idx_des].groupby(col):
        if len(sub) >= 10:
            rec_g = (sub['pred'] == idx_des).mean()
            subgroup_rows.append({'Variable': col, 'Subgrupo': grupo, 'N Desertores': len(sub), 'Recall (Desertor)': rec_g})

subgroup_df = pd.DataFrame(subgroup_rows)
print(subgroup_df.to_string(index=False, formatters={'Recall (Desertor)': '{:.4f}'.format}), flush=True)

print("\n=========================================================")
print("PROCESO COMPLETADO CON ÉXITO: PIPELINE MULTICLASE SIN FUGA DE DATOS.")
print(f"Imágenes generadas en: {OUTPUT_DIR}")
print("=========================================================", flush=True)
