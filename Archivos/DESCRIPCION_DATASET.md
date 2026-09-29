# Descripcion del Dataset y Variables

## 1. Informacion General del Dataset

- **Nombre del archivo**: `estudiantes_sinteticos_11500.csv`
- **Total de registros**: 11,500 estudiantes
- **Total de variables**: 22 (21 características predictoras y 1 variable objetivo)
- **Valores nulos**: 0 (0.0%)
- **Variable Objetivo (`desercion`)**:
  - `0` (No deserción / Permanece): 7,637 estudiantes (66.41%)
  - `1` (Deserción): 3,863 estudiantes (33.59%)

---

## 2. Diccionario de Variables

### 2.1 Variables Socio-Demograficas y Geograficas

| Variable | Tipo de Dato | Tipo de Rango | Rango Original / Valores | Descripcion |
| :--- | :--- | :--- | :--- | :--- |
| `edad_ingreso` | Numérica (Entero) | Continuo discreto | `[15 - 32]` años | Edad del estudiante al ingresar a la institución educativa. |
| `sexo` | Categórica | Nominal (2 cat.) | `'Hombre'`, `'Mujer'` | Género reportado por el estudiante. |
| `estado_civil` | Categórica | Nominal (4 cat.) | `'Soltero'`, `'Union libre'`, `'Casado'`, `'Otro'` | Estado civil registrado en la matrícula. |
| `zona_residencia` | Categórica | Nominal (2 cat.) | `'Urbana'`, `'Rural'` | Tipo de zona de la vivienda de origen. |
| `departamento` | Categórica | Nominal (27 cat.) | 27 departamentos de Colombia | Departamento de procedencia del estudiante. |
| `distancia_hogar_ies_km` | Numérica (Continuo) | Continuo | `[0.22 - 250.00]` km | Distancia estimada desde la residencia a la sede de la IES. |

### 2.2 Variables Academicas

| Variable | Tipo de Dato | Tipo de Rango | Rango Original / Valores | Descripcion |
| :--- | :--- | :--- | :--- | :--- |
| `promedio_academico` | Numérica (Continuo) | Continuo | `[0.87 - 5.00]` | Promedio acumulado de calificaciones (escala 0.0 - 5.0). |
| `materias_reprobadas` | Numérica (Entero) | Discreto | `[0 - 12]` asignaturas | Total acumulado de materias reprobadas. |
| `puntaje_icfes_percentil` | Numérica (Entero) | Discreto | `[1 - 100]` percentil | Percentil nacional obtenido en el examen de Estado Saber 11. |
| `semestre_cursado` | Numérica (Entero) | Discreto | `[1 - 12]` semestres | Semestre académico actual del estudiante. |
| `nivel_formacion` | Categórica | Ordinal/Nominal (3 cat.) | `'Universitario'`, `'Tecnologico'`, `'Tecnico profesional'` | Nivel de formación del programa matriculado. |
| `metodologia` | Categórica | Nominal (3 cat.) | `'Presencial'`, `'Virtual'`, `'Distancia tradicional'` | Modalidad de impartición de las clases. |
| `area_conocimiento` | Categórica | Nominal (8 cat.) | 8 áreas del conocimiento | Área académica o campo disciplinar. |

### 2.3 Variables Socioeconomicas, Familiares y Financieras

| Variable | Tipo de Dato | Tipo de Rango | Rango Original / Valores | Descripcion |
| :--- | :--- | :--- | :--- | :--- |
| `estrato_socioeconomico` | Numérica (Entero) | Ordinal | `[1 - 6]` | Estrato socioeconómico de la residencia. |
| `nivel_educativo_padre` | Categórica | Ordinal (6 cat.) | `'Ninguno'`, `'Primaria'`, `'Bachillerato'`, `'Tecnico'`, `'Universitario'`, `'Posgrado'` | Máximo nivel académico del padre. |
| `nivel_educativo_madre` | Categórica | Ordinal (6 cat.) | `'Ninguno'`, `'Primaria'`, `'Bachillerato'`, `'Tecnico'`, `'Universitario'`, `'Posgrado'` | Máximo nivel académico de la madre. |
| `sector_ies` | Categórica | Nominal (2 cat.) | `'Publico'`, `'Privado'` | Carácter institucional de la universidad/IES. |
| `trabaja_mientras_estudia` | Numérica (Binario) | Binario | `0` (No), `1` (Sí) | Indica si el estudiante trabaja simultáneamente. |
| `beneficiario_icetex` | Numérica (Binario) | Binario | `0` (No), `1` (Sí) | Indica si cuenta con crédito o apoyo ICETEX. |
| `beneficiario_beca` | Numérica (Binario) | Binario | `0` (No), `1` (Sí) | Indica si dispone de beca académica o institucional. |
| `anio_registro` | Numérica (Entero) | Discreto | `[2021 - 2025]` | Año de registro o ingreso a la cohorte. |

### 2.4 Variable Objetivo (Target)

| Variable | Tipo de Dato | Tipo de Rango | Rango Original / Valores | Descripcion |
| :--- | :--- | :--- | :--- | :--- |
| `desercion` | Numérica (Binario) | Binario | `0` (Permanece), `1` (Deserta) | Condición final del estudiante respecto al abandono de estudios. |

---

## 2.5 Criterio y Formula de Construccion de la Variable Objetivo (`desercion`)

Dado que el archivo original no contaba con una etiqueta explícita de deserción, la variable objetivo `desercion` se construyó mediante un **índice de riesgo ponderado sintético**, fundamentado en la literatura de deserción universitaria (factores académicos, laborales y socioeconómicos).

### Formula de Puntaje de Riesgo
Para cada estudiante $i$, se calcula un puntaje de riesgo continuo $R_i \in [0, 7.5]$ sumando las siguientes condiciones:

$$R_i = 2.5 \cdot \mathbb{I}(\text{promedio} < 3.0) + 2.5 \cdot \mathbb{I}(\text{materias\_reprobadas} \ge 2) + 1.0 \cdot \mathbb{I}(\text{icfes\_percentil} < 40) + 1.0 \cdot \mathbb{I}(\text{trabaja} = 1) + 0.5 \cdot \mathbb{I}(\text{beca} = 0) + 0.5 \cdot \mathbb{I}(\text{estrato} \le 2)$$

Donde $\mathbb{I}(\cdot)$ es la función indicadora que toma el valor 1 si la condición se cumple y 0 en caso contrario.

### Desglose de Factores y Ponderaciones

1. **Desempeño Académico Actual (Peso Alto: 5.0 puntos máximos)**:
   - `promedio_academico < 3.0` (+2.5 puntos): Representa bajo rendimiento crítico (notas por debajo del umbral de aprobación de 3.0 en escala 0.0 - 5.0).
   - `materias_reprobadas >= 2` (+2.5 puntos): Refleja pérdida recurrente de asignaturas y alto riesgo de pérdida de cupo.

2. **Competencias Previas y Carga Laboral (Peso Medio: 2.0 puntos máximos)**:
   - `puntaje_icfes_percentil < 40` (+1.0 punto): Vacíos en competencias básicas de ingreso (percentil inferior al 40%).
   - `trabaja_mientras_estudia == 1` (+1.0 punto): Doble jornada académica-laboral que limita el tiempo de estudio.

3. **Vulnerabilidad Socioeconómica y Financiera (Peso de Apoyo: 1.0 punto máximo)**:
   - `beneficiario_beca == 0` (+0.5 puntos): Ausencia de beca institucional o estatal.
   - `estrato_socioeconomico <= 2` (+0.5 puntos): Limitaciones económicas del entorno familiar (estratos 1 y 2).

### Regla de Decisión Binaria y Umbral
Se establece un **umbral crítico de 3.5 puntos** para clasificar la condición final del estudiante:

$$\text{desercion}_i = \begin{cases} 1 & \text{si } R_i \ge 3.5 \quad (\text{Estudiante en Deserción}) \\ 0 & \text{si } R_i < 3.5 \quad (\text{Estudiante en Permanencia}) \end{cases}$$

Este criterio simula una tasa de deserción estructural realista del **33.59%** (3,863 estudiantes en deserción vs. 7,637 en permanencia).

---

## 3. Estadisticas Descriptivas de Variables Numericas (Pre-Normalizacion)

Las siguientes métricas corresponden a los valores originales calculados sobre la totalidad de las 11,500 observaciones antes de aplicar cualquier proceso de escalamiento o transformación:

| Variable | Media | Desv. Est. | Mínimo | Q1 (25%) | Mediana (50%) | Q3 (75%) | Máximo | Rango Original |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `edad_ingreso` | 19.57 | 2.96 | 15.00 | 17.00 | 19.00 | 22.00 | 32.00 | `[15.00 - 32.00]` |
| `promedio_academico` | 3.41 | 0.63 | 0.87 | 2.98 | 3.41 | 3.84 | 5.00 | `[0.87 - 5.00]` |
| `materias_reprobadas` | 1.09 | 1.20 | 0.00 | 0.00 | 1.00 | 2.00 | 12.00 | `[0.00 - 12.00]` |
| `distancia_hogar_ies_km` | 11.12 | 12.52 | 0.22 | 4.03 | 7.43 | 13.56 | 250.00 | `[0.22 - 250.00]` |
| `puntaje_icfes_percentil` | 57.96 | 17.80 | 1.00 | 46.00 | 58.00 | 70.00 | 100.00 | `[1.00 - 100.00]` |
| `semestre_cursado` | 6.53 | 3.44 | 1.00 | 4.00 | 7.00 | 9.00 | 12.00 | `[1.00 - 12.00]` |
| `estrato_socioeconomico` | 2.61 | 1.20 | 1.00 | 2.00 | 2.00 | 3.00 | 6.00 | `[1.00 - 6.00]` |
| `anio_registro` | 2023.31 | 1.27 | 2021.00 | 2022.00 | 2023.00 | 2024.00 | 2025.00 | `[2021.00 - 2025.00]` |
| `trabaja_mientras_estudia` | 0.38 | 0.48 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | `[0.00 - 1.00]` |
| `beneficiario_icetex` | 0.22 | 0.41 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | `[0.22 - 1.00]` |
| `beneficiario_beca` | 0.19 | 0.39 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | `[0.00 - 1.00]` |

---

## 4. Diagnostico sobre la Necesidad de Escalamiento

1. **Diferencia de Órdenes de Magnitud**:
   - `distancia_hogar_ies_km` abarca una amplitud de 249.78 unidades (0.22 a 250 km).
   - `puntaje_icfes_percentil` abarca una amplitud de 99 unidades (1 a 100).
   - `promedio_academico` opera únicamente en una amplitud de 4.13 unidades (0.87 a 5.00).
   - Las variables indicadoras (`trabaja_mientras_estudia`, `beneficiario_icetex`, `beneficiario_beca`) operan estrictamente en la escala discreta 0 o 1.

2. **Requisito Tecnico para SVM**:
   - El algoritmo SVM con kernel RBF computa distancias entre vectores de características mediante la norma euclidiana $\|x - x'\|^2$.
   - Sin una etapa previa de normalización o estandarización, las variables con rangos numéricos elevados dominan la norma euclidiana, lo que invalida la contribución de variables predictoras en rangos pequeños pero con alto valor explicativo (como el promedio académico).

---

## 5. Configuracion e Hiperparametros del Modelo SVM (`SVC`)

Para el proceso de clasificación de deserción estudiantil se empleó el estimador `SVC` de Scikit-Learn con la siguiente configuración de hiperparámetros:

```python
SVC(kernel='rbf', C=1.0, gamma='scale', cache_size=1000, random_state=42)
```

### 5.1 Justificacion e Implicaciones de Cada Hiperparametro

| Hiperparámetro | Valor Configurado | Razón de Elección | Implicaciones Técnicas en el Modelo |
| :--- | :--- | :--- | :--- |
| **`kernel`** | `'rbf'` (Radial Basis Function) | La separación entre estudiantes que desertan y permanecen no es linealmente separable. El kernel RBF mapea las características a un espacio de dimensión infinita mediante $K(x, x') = \exp(-\gamma \|x - x'\|^2)$. | Permite construir fronteras de decisión curvadas y complejas capaces de capturar interacciones no lineales entre variables socioeconómicas y académicas sin necesidad de calcular explícitamente combinaciones polinómicas. Requiere estrictamente que las variables estén normalizadas. |
| **`C`** | `1.0` | Representa la constante de penalización en la formulación de margen blando (*Soft Margin*). Controla el balance entre maximizar la distancia del margen y minimizar las violaciones de clasificación. | Un valor de $C=1.0$ establece un equilibrio estándar. Evita tanto el sobreajuste (*overfitting*, que ocurriría con $C \gg 1.0$ al tratar de clasificar perfectamente todo el ruido) como el subajuste (*underfitting*, que ocurriría con $C \ll 1.0$ al generar un margen demasiado permisivo). |
| **`gamma`** | `'scale'` | Ajusta el ancho de banda de la función gaussiana RBF dinámicamente como $\gamma = \frac{1}{n\_features \cdot \text{Var}(X)}$. | Garantiza que el alcance de influencia de los vectores de soporte individuales sea proporcional a la escala global de las características transformadas, evitando la creación de "islas" muy localizadas o fronteras excesivamente suaves. |
| **`cache_size`**| `1000` (MB) | Reserva 1,000 MB de memoria RAM dedicados al almacenamiento en caché de la matriz de producto interno del kernel (Matriz Gram). | Optimiza significativamente la velocidad de ejecución y reduce el tiempo de entrenamiento del problema de Programación Cuadrática (QP) en 11,500 datos sin alterar los resultados matemáticos del modelo. |
| **`random_state`** | `42` | Fija la semilla del generador numérico pseudo-aleatorio. | Garantiza la reproducibilidad matemática exacta de los hiperplanos entrenados y de los resultados de evaluación a través de las distintas ejecuciones y experimentos. |

---

## 6. Metodos de Normalizacion y Transformacion Evaluados (Segun Articulo IEEE 2024)

De acuerdo con la metodología expuesta en el artículo científico (*Support Vector Machine for Predicting Student Dropout Under Different Normalization Methods*, IEEE 2024), se evalúa el comportamiento de SVM al aplicar distintas técnicas de transformación y escalamiento de características.

### 6.1 Descripcion Matematica y Operativa de Cada Metodo

| Método de Normalización | Fórmula / Transformación Matemática | Descripción Operativa y Comportamiento | Efecto en la Frontera de Decisión de SVM |
| :--- | :--- | :--- | :--- |
| **`StandardScaler`** | $z = \frac{x - \mu}{\sigma}$ | Resta la media ($\mu$) y divide entre la desviación estándar ($\sigma$). Transforma los datos para tener media 0 y varianza 1 ($\mu=0, \sigma=1$). | Preserva la relación lineal original de las variables mientras iguala su dispersión. Es la opción estándar óptima cuando las características numéricas tienen distribuciones aproximadamente gaussianas. |
| **`MinMaxScaler`** | $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | Escala linealmente todas las características a un rango acotado estricto, típicamente $[0, 1]$. | Mantiene la forma exacta de la distribución original. Sensible a valores atípicos (*outliers*) extremos, ya que $x_{\min}$ y $x_{\max}$ determinan los límites absolutos. |
| **`MaxAbsScaler`** | $x' = \frac{x}{|x_{\max}|}$ | Escala cada característica dividiendo por su valor absoluto máximo, mapeando los datos al rango $[-1, 1]$. | No desplaza el centro de los datos (no resta la media), por lo que preserva la estructura de ceros explícitos (*sparsity*). |
| **`QuantileTransformer`** | $x' = G^{-1}(F_{emp}(x))$ | Aplica una transformación no paramétrica basada en la función de distribución acumulada empírica (ECDF) a cuantiles. | Mapea las características a una distribución uniforme o normal. Atenúa el impacto de valores atípicos y distribuciones fuertemente sesgadas, suavizando las distancias euclidianas. |
| **`PowerTransformer`** *(Yeo-Johnson)* | Transformación de potencia estocástica para estabilizar varianza y minimizar *skewness*. | Transforma las características numéricas no gaussianas para hacerlas lo más cercanas posible a una distribución normal estándar. | Corrige la asimetría en distribuciones sesgadas. En los experimentos empíricos con SVM, este método alcanzó uno de los desempeños más altos ($F1 \approx 0.9085$). |
| **`No Normalization`** *(Baseline)* | $x' = x$ | Mantiene las variables crudas en sus unidades y escalas originales (sin transformación). | Punto de comparación baseline. Provoca una falla severa en el algoritmo SVM ($F1 = 0.0$ o muy bajo) debido a la dominancia desproporcionada de variables con amplitudes grandes. |

### 6.2 Impacto de Codificadores Categóricos Evaluados en el Artículo

| Codificador | Operación | Impacto en SVM |
| :--- | :--- | :--- |
| **`OneHotEncoder`** | Convierte variables categóricas en vectores binarios indicadoras (0 o 1). | Método recomendado para variables nominales sin orden inherente. Evita introducir jerarquías artificiales que distorsionen el cálculo de distancias en SVM. |
| **`OrdinalEncoder` / `LabelEncoder`** | Asigna valores enteros secuenciales ($0, 1, 2, \dots$) a las categorías. | Solo aplicable si existe un orden jerárquico real. Si se aplica a variables nominales, introduce relaciones de orden inexistentes que penalizan el rendimiento del hiperplano ($F1 \approx 0.50$ según el artículo). |
