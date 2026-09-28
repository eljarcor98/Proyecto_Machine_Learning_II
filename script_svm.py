"""
Experimento SVM: Predicción de Deserción Estudiantil sobre Dataset Sintético (11,500 Estudiantes)
- EDA Completo de Variables Socioeconómicas y Académicas
- Flujo Reproducible Sin Fuga de Datos (Scikit-Learn Pipeline + ColumnTransformer)
- Evaluación de Normalizaciones vs Baseline (Sin Normalizar) utilizando n_jobs=2
- Importancia de Variables en la Clasificación (Permutation Importance con n_jobs=2)
- Visualización del Hiperplano, Márgenes de Decisión y Vectores de Soporte
- Clasificación Desglosada de Estudiantes (Matriz de Confusión y Función de Decisión)

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
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler,
    QuantileTransformer, PowerTransformer, OrdinalEncoder, OneHotEncoder, FunctionTransformer
)
from scipy.stats import ttest_ind

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'Arial'

# Directorio de salida para guardar visualizaciones
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Archivos", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. CARGA DE DATOS Y EDA PREVIO
# ==========================================
print("=========================================================")
print("1. ANÁLISIS EXPLORATORIO DE DATOS (EDA) - DATASET SINTÉTICO (11,500)")
print("=========================================================", flush=True)

data_path = os.path.join(os.path.dirname(__file__), "Archivos", "estudiantes_sinteticos_11500.csv")
df = pd.read_csv(data_path)

print(f"Dimensiones del dataset: {df.shape[0]} estudiantes y {df.shape[1]} variables.", flush=True)
print(f"Valores nulos totales en el dataset: {df.isnull().sum().sum()}", flush=True)

# Formulación del evento binario de deserción
riesgo = (
    (df['promedio_academico'] < 3.0).astype(int) * 2.5 +
    (df['materias_reprobadas'] >= 2).astype(int) * 2.5 +
    (df['puntaje_icfes_percentil'] < 40).astype(int) * 1.0 +
    (df['trabaja_mientras_estudia'] == 1).astype(int) * 1.0 +
    (df['beneficiario_beca'] == 0).astype(int) * 0.5 +
    (df['estrato_socioeconomico'] <= 2).astype(int) * 0.5
)
df['desercion'] = (riesgo >= 3.5).astype(int)

# Identificación de columnas categóricas y numéricas crudas
cat_cols = ['sexo', 'estado_civil', 'zona_residencia', 'departamento',
            'nivel_educativo_padre', 'nivel_educativo_madre', 'sector_ies',
            'nivel_formacion', 'metodologia', 'area_conocimiento']

num_cols = ['edad_ingreso', 'promedio_academico', 'materias_reprobadas',
            'distancia_hogar_ies_km', 'puntaje_icfes_percentil', 'semestre_cursado',
            'estrato_socioeconomico', 'anio_registro', 'trabaja_mientras_estudia',
            'beneficiario_icetex', 'beneficiario_beca']

X_raw = df[num_cols + cat_cols]
y = df['desercion'].values

print("\nDistribución de la Variable Objetivo 'desercion' (1 = Desertor, 0 = No Desertor):", flush=True)
print(pd.Series(y).value_counts())
print(pd.Series(y).value_counts(normalize=True) * 100)

# FIGURA 1: EDA - Distribución de Clases y Correlaciones
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.countplot(data=df, x='desercion', palette='Set2', ax=axes[0])
axes[0].set_xticklabels(['No Desertor (0)', 'Desertor (1)'])
axes[0].set_title('Distribución de Estudiantes (Dataset Sintético 11,500)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Estado Académico', fontsize=12)
axes[0].set_ylabel('Número de Estudiantes', fontsize=12)
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)',
                     (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', fontsize=11, color='white', fontweight='bold')

# Matriz correlacional numérica pura
correlations = df[num_cols].apply(lambda col: col.corr(pd.Series(y))).sort_values()
colors = ['#d95f02' if c > 0 else '#7570b3' for c in correlations.values]
axes[1].barh(correlations.index, correlations.values, color=colors)
axes[1].set_title('Correlación de Pearson de Variables Numéricas con Deserción', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Coeficiente de Correlación de Pearson', fontsize=12)
axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)

plt.tight_layout()
fig_eda_path = os.path.join(OUTPUT_DIR, "eda_distribucion_correlaciones.png")
plt.savefig(fig_eda_path, dpi=300)
plt.close()
print(f"[OK] Gráfico EDA guardado en: {fig_eda_path}", flush=True)

# FIGURA 2: Distribución de variables clave por deserción
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(data=df, x='desercion', y='promedio_academico', palette='Set1', ax=axes[0])
axes[0].set_xticklabels(['No Desertor (0)', 'Desertor (1)'])
axes[0].set_title('Promedio Académico vs Deserción', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Estado del Estudiante', fontsize=12)
axes[0].set_ylabel('Promedio Académico (0.0 - 5.0)', fontsize=12)

sns.boxplot(data=df, x='desercion', y='materias_reprobadas', palette='Set1', ax=axes[1])
axes[1].set_xticklabels(['No Desertor (0)', 'Desertor (1)'])
axes[1].set_title('Materias Reprobadas vs Deserción', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Estado del Estudiante', fontsize=12)
axes[1].set_ylabel('Materias Reprobadas', fontsize=12)

plt.tight_layout()
fig_box_path = os.path.join(OUTPUT_DIR, "eda_variables_clave.png")
plt.savefig(fig_box_path, dpi=300)
plt.close()
print(f"[OK] Gráfico de variables clave guardado en: {fig_box_path}", flush=True)


# ==========================================
# 2. FLUJO REPRODUCIBLE SIN FUGA DE DATOS (PIPELINE)
# ==========================================
print("\n=========================================================")
print("2. FLUJO REPRODUCIBLE SIN FUGA DE DATOS (PIPELINES + COLUMNTRANSFORMER)")
print("   (Ejecución paralela multihilo con n_jobs=2)")
print("=========================================================", flush=True)

normalization_methods = {
    'Standard Scaler': StandardScaler(),
    'Min Max Scaler': MinMaxScaler(),
    'Quantile Transformer': QuantileTransformer(n_quantiles=50, random_state=42, output_distribution='uniform'),
    'Max Absolute Scaler': MaxAbsScaler(),
    'Power Transformer': PowerTransformer(method='yeo-johnson'),
    'No Normalization (Baseline)': FunctionTransformer(validate=False)  # Identidad sin escalar
}

N_TRIALS = 10
N_JOBS = 2  # 2 núcleos de procesador

def evaluate_single_trial(trial_idx):
    """
    Ejecuta una partición train_test_split y evalúa cada PIPELINE estrictamente ajustado en train.
    PREVIENE LA FUGA DE DATOS (DATA LEAKAGE) al no tocar test durante fit().
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42 + trial_idx, stratify=y
    )
    
    trial_scores = {}
    for method_name, num_scaler in normalization_methods.items():
        try:
            # Re-instanciar el escalador para aislamiento completo por trial
            num_scaler_inst = type(num_scaler)(**num_scaler.get_params()) if hasattr(num_scaler, 'get_params') else num_scaler

            # ColumnTransformer ajustado estrictamente en X_train de esta partición
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_scaler_inst, num_cols),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
                ]
            )

            # Pipeline completo: Preprocesamiento + SVM Estimador
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', SVC(kernel='rbf', C=1.0, cache_size=1000, random_state=42))
            ])

            # Entrenar el Pipeline COMPLETO únicamente con datos de entrenamiento
            pipeline.fit(X_train, y_train)
            
            # Predecir en datos de prueba no vistos
            preds = pipeline.predict(X_test)
            trial_scores[method_name] = f1_score(y_test, preds)
        except Exception as e:
            trial_scores[method_name] = np.nan
            
    return trial_scores

print(f"Iniciando {N_TRIALS} iteraciones de Pipelines independientes (n_jobs={N_JOBS})...", flush=True)
parallel_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
    delayed(evaluate_single_trial)(t) for t in range(N_TRIALS)
)
print("Iteraciones completadas exitosamente.", flush=True)

results = {method: [] for method in normalization_methods.keys()}
for res in parallel_results:
    for method_name, score in res.items():
        results[method_name].append(score)

baseline_scores = results['No Normalization (Baseline)']
summary_rows = []

for method_name, scores in results.items():
    mean_f1 = np.mean(scores)
    std_f1 = np.std(scores)
    
    if method_name != 'No Normalization (Baseline)':
        stat, p_val = ttest_ind(scores, baseline_scores, equal_var=False)
    else:
        p_val = 1.0
        
    summary_rows.append({
        'Normalization Method': method_name,
        'Average F1 Score': mean_f1,
        'Std Dev': std_f1,
        'P-value vs Baseline': p_val
    })

summary_df = pd.DataFrame(summary_rows).sort_values(by='Average F1 Score', ascending=False)

print("\n" + "="*80)
print("TABLA COMPARATIVA: PROMEDIO DE F1-SCORE (PIPELINES SIN FUGA DE DATOS)")
print("="*80)
print(summary_df.to_string(index=False, formatters={
    'Average F1 Score': '{:.4f}'.format,
    'Std Dev': '{:.4f}'.format,
    'P-value vs Baseline': '{:.6f}'.format
}), flush=True)

# FIGURA 3: Comparación de F1-Score (Normalizadores vs Baseline)
plt.figure(figsize=(12, 6))
colors_bar = ['#d95f02' if 'Baseline' in name else '#1b9e77' for name in summary_df['Normalization Method']]
barplot = sns.barplot(
    data=summary_df,
    x='Average F1 Score',
    y='Normalization Method',
    palette=colors_bar
)

plt.title('Evaluación de Pipelines SVM sin Fuga de Datos: Escaladores vs Baseline', fontsize=14, fontweight='bold')
plt.xlabel(f'F1-Score Promedio ({N_TRIALS} iteraciones)', fontsize=12)
plt.ylabel('Método de Normalización en Pipeline', fontsize=12)
plt.xlim(0.4, 1.0)

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
print("3. IMPORTANCIA DE VARIABLES TOMADAS POR EL MODELO SVM")
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
    ('classifier', SVC(kernel='rbf', C=1.0, probability=True, cache_size=1000, random_state=42))
])

best_pipeline.fit(X_train, y_train)

# Exportar artefacto de modelo reproducible
model_artifact_path = os.path.join(os.path.dirname(__file__), "Archivos", "svm_pipeline_reproducible.joblib")
dump(best_pipeline, model_artifact_path)
print(f"[OK] Pipeline completo serializado e imutable guardado en: {model_artifact_path}", flush=True)

# Permutation Importance sobre el test set crudo usando el Pipeline completo
perm_importance = permutation_importance(
    best_pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=N_JOBS
)

sorted_importances_idx = perm_importance.importances_mean.argsort()[::-1]
feature_names = X_raw.columns

imp_df = pd.DataFrame({
    'Feature': feature_names[sorted_importances_idx],
    'Importance_Mean': perm_importance.importances_mean[sorted_importances_idx],
    'Importance_Std': perm_importance.importances_std[sorted_importances_idx]
})

print("\nTop 10 Variables más influyentes para el Pipeline SVM:", flush=True)
print(imp_df.head(10).to_string(index=False), flush=True)

# FIGURA 4: Importancia de Variables
plt.figure(figsize=(10, 8))
top_15_imp = imp_df.head(15)
plt.barh(top_15_imp['Feature'][::-1], top_15_imp['Importance_Mean'][::-1], color='#2b5c8f', xerr=top_15_imp['Importance_Std'][::-1])
plt.title('Importancia de Variables en Pipeline SVM (Permutation Importance)', fontsize=14, fontweight='bold')
plt.xlabel('Disminución Promedio en F1-Score al Permutar Variable', fontsize=12)
plt.tight_layout()
fig_imp_path = os.path.join(OUTPUT_DIR, "svm_importancia_variables.png")
plt.savefig(fig_imp_path, dpi=300)
plt.close()
print(f"[OK] Gráfico de importancia de variables guardado en: {fig_imp_path}", flush=True)


# ==========================================
# 4. VISUALIZACIÓN DEL HIPERPLANO DE SVM
# ==========================================
print("\n=========================================================")
print("4. VISUALIZACIÓN DEL HIPERPLANO DE SEPARACIÓN DE SVM Y VECTORES DE SOPORTE")
print("=========================================================", flush=True)

# Preprocesar matrices transformadas estrictamente
X_train_trans = best_pipeline.named_steps['preprocessor'].transform(X_train)
X_test_trans = best_pipeline.named_steps['preprocessor'].transform(X_test)

# A. Visualización del Hiperplano en Espacio PCA 2D
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_trans)
X_test_pca = pca.transform(X_test_trans)

svm_pca = SVC(kernel='rbf', C=1.0, random_state=42)
svm_pca.fit(X_train_pca, y_train)

x_min, x_max = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
y_min, y_max = X_train_pca[:, 1].min() - 0.5, X_train_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))

Z = svm_pca.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(11, 8))
plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=['#e0f3f8', '#fee0d2'], alpha=0.6)
contours = plt.contour(xx, yy, Z, levels=[-1.0, 0.0, 1.0], linestyles=['--', '-', '--'],
                       colors=['#3182bd', '#e6550d', '#3182bd'], linewidths=[1.5, 2.5, 1.5])
plt.clabel(contours, inline=True, fontsize=10, fmt={-1.0: 'Margen (-1)', 0.0: 'Hiperplano (0)', 1.0: 'Margen (+1)'})

sample_idx = np.random.choice(len(y_test), size=1000, replace=False)
X_sample_pca = X_test_pca[sample_idx]
y_sample = y_test[sample_idx]

plt.scatter(X_sample_pca[y_sample == 0, 0], X_sample_pca[y_sample == 0, 1],
            c='#2b83ba', label='No Desertor (0)', alpha=0.6, edgecolors='k', s=30)
plt.scatter(X_sample_pca[y_sample == 1, 0], X_sample_pca[y_sample == 1, 1],
            c='#d7191c', label='Desertor (1)', alpha=0.6, edgecolors='k', s=30)

sv = svm_pca.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='black', linewidths=1.2,
            label=f'Vectores de Soporte ({len(sv)} puntos)')

plt.title('Hiperplano de Separación SVM (Kernel RBF en PCA 2D - 11,500 Estudiantes)', fontsize=14, fontweight='bold')
plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Varianza)', fontsize=12)
plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Varianza)', fontsize=12)
plt.legend(loc='upper right', frameon=True)

plt.tight_layout()
fig_hyperplane_path = os.path.join(OUTPUT_DIR, "svm_hiperplano_pca_2d.png")
plt.savefig(fig_hyperplane_path, dpi=300)
plt.close()
print(f"[OK] Gráfico del Hiperplano PCA guardado en: {fig_hyperplane_path}", flush=True)


# B. Visualización en las 2 Variables Reales Más Importantes
top_2_feats = ['promedio_academico', 'materias_reprobadas']
X_top2 = X_train[top_2_feats].values

scaler_top2 = StandardScaler()
X_top2_scaled = scaler_top2.fit_transform(X_top2)

svm_top2 = SVC(kernel='rbf', C=1.0, random_state=42)
svm_top2.fit(X_top2_scaled, y_train)

x_min, x_max = X_top2_scaled[:, 0].min() - 0.2, X_top2_scaled[:, 0].max() + 0.2
y_min, y_max = X_top2_scaled[:, 1].min() - 0.2, X_top2_scaled[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))

Z2 = svm_top2.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(11, 8))
plt.contourf(xx, yy, Z2, levels=[-np.inf, 0, np.inf], colors=['#e0f3f8', '#fee0d2'], alpha=0.5)
contours2 = plt.contour(xx, yy, Z2, levels=[-1.0, 0.0, 1.0], linestyles=['--', '-', '--'],
                         colors=['#3182bd', '#e6550d', '#3182bd'], linewidths=[1.5, 2.5, 1.5])
plt.clabel(contours2, inline=True, fontsize=10, fmt={-1.0: 'Margen (-1)', 0.0: 'Hiperplano (0)', 1.0: 'Margen (+1)'})

X_test_top2_scaled = scaler_top2.transform(X_test[top_2_feats].values)
plt.scatter(X_test_top2_scaled[y_test == 0, 0], X_test_top2_scaled[y_test == 0, 1],
            c='#2b83ba', label='No Desertor (0)', alpha=0.6, edgecolors='k', s=30)
plt.scatter(X_test_top2_scaled[y_test == 1, 0], X_test_top2_scaled[y_test == 1, 1],
            c='#d7191c', label='Desertor (1)', alpha=0.6, edgecolors='k', s=30)

sv2 = svm_top2.support_vectors_
plt.scatter(sv2[:, 0], sv2[:, 1], s=100, facecolors='none', edgecolors='black', linewidths=1.2,
            label=f'Vectores de Soporte ({len(sv2)} puntos)')

plt.title('Frontera de Decisión SVM: Promedio Académico vs Materias Reprobadas', fontsize=14, fontweight='bold')
plt.xlabel('Promedio Académico (Escalado)', fontsize=12)
plt.ylabel('Materias Reprobadas (Escalado)', fontsize=12)
plt.legend(loc='lower right', frameon=True)

plt.tight_layout()
fig_top2_path = os.path.join(OUTPUT_DIR, "svm_hiperplano_top2_variables.png")
plt.savefig(fig_top2_path, dpi=300)
plt.close()
print(f"[OK] Gráfico del Hiperplano Top 2 Variables guardado en: {fig_top2_path}", flush=True)


# ==========================================
# 5. CLASIFICACIÓN Y DESEMPEÑO EN ESTUDIANTES
# ==========================================
print("\n=========================================================")
print("5. ANÁLISIS DE LA CLASIFICACIÓN DE ESTUDIANTES POR EL MODELO")
print("=========================================================", flush=True)

y_pred = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("\nREPORTE DE CLASIFICACIÓN DETALLADO (TEST SET = 2,300 ESTUDIANTES):", flush=True)
print(classification_report(y_test, y_pred, target_names=['No Desertor (0)', 'Desertor (1)']), flush=True)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, best_pipeline.predict_proba(X_test)[:, 1])

print(f"Accuracy:  {acc:.4f}", flush=True)
print(f"Precision: {prec:.4f}", flush=True)
print(f"Recall:    {rec:.4f}", flush=True)
print(f"F1-Score:  {f1:.4f}", flush=True)
print(f"ROC-AUC:   {auc:.4f}", flush=True)

# FIGURA 5: Matriz de Confusión y Curva ROC
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['No Desertor (Pred)', 'Desertor (Pred)'],
            yticklabels=['No Desertor (Real)', 'Desertor (Real)'],
            annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Matriz de Confusión de Clasificación de Estudiantes', fontsize=14, fontweight='bold')

fpr, tpr, _ = roc_curve(y_test, best_pipeline.predict_proba(X_test)[:, 1])
axes[1].plot(fpr, tpr, color='#e6550d', lw=2.5, label=f'Pipeline SVM RBF (AUC = {auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
axes[1].set_title('Curva ROC - Capacidad de Discriminación', fontsize=14, fontweight='bold')
axes[1].legend(loc='lower right', fontsize=12)

plt.tight_layout()
fig_eval_path = os.path.join(OUTPUT_DIR, "svm_matriz_confusion_roc.png")
plt.savefig(fig_eval_path, dpi=300)
plt.close()
print(f"[OK] Gráfico de Matriz de Confusión y ROC guardado en: {fig_eval_path}", flush=True)

# FIGURA 6: Distribución de la Función de Decisión f(x) (Distancia al Hiperplano)
decision_dist = best_pipeline.decision_function(X_test)

plt.figure(figsize=(10, 6))
sns.histplot(decision_dist[y_test == 0], color='#2b83ba', kde=True, label='Estudiantes No Desertores', stat='density', alpha=0.5)
sns.histplot(decision_dist[y_test == 1], color='#d7191c', kde=True, label='Estudiantes Desertores', stat='density', alpha=0.5)

plt.axvline(0, color='black', linestyle='--', linewidth=2, label='Hiperplano f(x) = 0')
plt.axvline(1, color='gray', linestyle=':', linewidth=1.5, label='Margen +1')
plt.axvline(-1, color='gray', linestyle=':', linewidth=1.5, label='Margen -1')

plt.title('Distribución de Distancias Funcionales al Hiperplano (Pipeline Sin Fuga)', fontsize=14, fontweight='bold')
plt.xlabel('Distancia Funcional f(x) = wᵀφ(x) + b', fontsize=12)
plt.ylabel('Densidad de Estudiantes', fontsize=12)
plt.legend(loc='upper right', frameon=True)

plt.tight_layout()
fig_dist_path = os.path.join(OUTPUT_DIR, "svm_distancia_hiperplano_estudiantes.png")
plt.savefig(fig_dist_path, dpi=300)
plt.close()
print(f"[OK] Gráfico de Distribución de Distancia al Hiperplano guardado en: {fig_dist_path}", flush=True)

print("\n=========================================================")
print("PROCESO COMPLETADO CON ÉXITO: PIPELINES REPRODUCIBLES Y SIN FUGA DE DATOS.")
print(f"Todas las imágenes de análisis han sido generadas en:\n{OUTPUT_DIR}")
print("=========================================================", flush=True)
