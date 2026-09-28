"""
Experimento SVM: Predicción de Deserción Estudiantil evaluando diferentes métodos de normalización
Basado en las condiciones del artículo IEEE Xplore:
'Support Vector Machine for Predicting Student Dropout Under Different Normalization Methods'
"""

import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    QuantileTransformer,
    PowerTransformer,
    OrdinalEncoder,
    OneHotEncoder
)
from scipy.stats import ttest_ind

warnings.filterwarnings('ignore')

# 1. Cargar el dataset (Dataset UCI del artículo)
data_path = r"c:\Users\Santiago T\Documents\Ubiversidad 2026\2026 - II\Machine Learning 2\Proyecto\Archivos\UCI_Dataset\data.csv"
print(f"Cargando dataset desde: {data_path}")
df = pd.read_csv(data_path, sep=';')

# Filtrar para clasificación binaria de deserción (Dropout vs Graduate / Enrolled)
# Target: 'Dropout' -> 1, 'Graduate'/'Enrolled' -> 0
df['Target_Binary'] = df['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)

# Separar características (X) y variable objetivo (y)
X_raw = df.drop(columns=['Target', 'Target_Binary'])
y = df['Target_Binary'].values

print(f"Dataset cargado: {X_raw.shape[0]} registros, {X_raw.shape[1]} características.")
print("Distribución de la variable objetivo (Dropout = 1, No Dropout = 0):")
print(pd.Series(y).value_counts(normalize=True).to_dict())

# 2. Métodos de normalización (Tabla I del artículo)
normalization_methods = {
    'One Hot Encoder': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
    'Max Absolute Scaler': MaxAbsScaler(),
    'Min Max Scaler': MinMaxScaler(),
    'Quantile Transformer': QuantileTransformer(n_quantiles=100, random_state=42, output_distribution='uniform'),
    'Power Transformer': PowerTransformer(method='yeo-johnson'),
    'Standard Scaler': StandardScaler(),
    'Ordinal Encoder': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    'No Normalization': None
}

N_TRIALS = 15
results = {method: [] for method in normalization_methods.keys()}

print(f"\nEjecutando {N_TRIALS} iteraciones de SVM para cada método de normalización...")

for trial in range(N_TRIALS):
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42 + trial, stratify=y
    )

    for method_name, scaler in normalization_methods.items():
        try:
            if scaler is None:
                X_tr = X_train.values
                X_te = X_test.values
            else:
                X_tr = scaler.fit_transform(X_train)
                X_te = scaler.transform(X_test)

            clf = SVC(kernel='rbf', C=1.0, cache_size=1000, random_state=42)
            clf.fit(X_tr, y_train)
            
            preds = clf.predict(X_te)
            score = f1_score(y_test, preds)
            results[method_name].append(score)
        except Exception as e:
            print(f"Error en {method_name}: {e}")

# 3. Resumen de resultados (Equivalente a Tabla II del artículo)
summary_rows = []
baseline_scores = results['No Normalization']

for method_name, scores in results.items():
    if len(scores) > 0:
        mean_f1 = np.mean(scores)
        std_f1 = np.std(scores)
        if method_name != 'No Normalization' and len(baseline_scores) > 0:
            stat, p_val = ttest_ind(scores, baseline_scores, equal_var=False)
        else:
            p_val = 1.0

        summary_rows.append({
            'Normalization Method': method_name,
            'Average F1 Score': round(float(mean_f1), 6),
            'Std Dev': round(float(std_f1), 6),
            'P value': round(float(p_val), 6) if p_val >= 0.000001 else 0.00000
        })

summary_df = pd.DataFrame(summary_rows).sort_values(by='Average F1 Score', ascending=False)

print("\n" + "="*75)
print("TABLA II: PROMEDIO DE F1-SCORE TRAS ITERACIONES CON SVM")
print("="*75)
print(summary_df.to_string(index=False))

# Guardar resultados en CSV
output_csv = r"c:\Users\Santiago T\Documents\Ubiversidad 2026\2026 - II\Machine Learning 2\Proyecto\Archivos\resultados_svm_normalizacion.csv"
summary_df.to_csv(output_csv, index=False)
print(f"\nResultados guardados exitosamente en: {output_csv}")
