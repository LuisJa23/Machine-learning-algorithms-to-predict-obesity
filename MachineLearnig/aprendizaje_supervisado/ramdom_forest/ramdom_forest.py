import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar el conjunto de datos
data = pd.read_csv('../data/ObesityDataSet_raw_and_data_sinthetic.csv')

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']

# Convertir variables categóricas a variables dummy
X = pd.get_dummies(X, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'])

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar solo las columnas numéricas usando StandardScaler
numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Árbol de Decisión

# Crear el modelo de Árbol de Decisión
dt_classifier = DecisionTreeClassifier(max_depth=8)
dt_classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_dt = dt_classifier.predict(X_test)

# Evaluar el rendimiento del modelo de Árbol de Decisión
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
classification_rep_dt = classification_report(y_test, y_pred_dt)


# Random Forest

# Crear el modelo de Random Forest
rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=12)
rf_classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_rf = rf_classifier.predict(X_test)

# Evaluar el rendimiento del modelo de Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)

# Guardar el modelo entrenado en un archivo
joblib.dump(rf_classifier, 'random_forest_model.joblib')

# Visualización

# Importancia de características (Random Forest)

# Obtener la importancia de las características
importances = rf_classifier.feature_importances_

# Ordenar las características por importancia
indices = np.argsort(importances)[::-1]

# Crear la visualización con mayor espacio y etiquetas en vertical
plt.figure(figsize=(15, 5))
plt.bar(X.columns[indices], importances[indices])
plt.xlabel("Característica")
plt.ylabel("Importancia")
plt.title("Importancia de características en el Random Forest")

# Ajustar el espaciado y rotar los nombres de las características
plt.xticks(rotation='vertical', ha='right')
plt.subplots_adjust(bottom=0.2)

plt.show()

# Visualizar un árbol individual (Random Forest)

# Extraer un árbol del conjunto de árboles
estimator = rf_classifier.estimators_[0]

# Visualizar el árbol individual
export_graphviz(estimator, out_file='tree.dot', feature_names=X.columns, filled=True, class_names=['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'])

# Gráfico de clasificaciones por género

# Crear un DataFrame con género, predicciones y etiquetas reales
result_df_rf = pd.DataFrame({'Gender_Female': X_test['Gender_Female'], 'Gender_Male': X_test['Gender_Male'], 'Prediction': y_pred_rf, 'Actual': y_test})

# Filtrar el DataFrame para obtener solo las instancias clasificadas correctamente
correctly_classified_df_rf = result_df_rf[result_df_rf['Prediction'] == result_df_rf['Actual']]

# Contar la cantidad total de instancias clasificadas correctamente por género
gender_correct_counts_rf = correctly_classified_df_rf.groupby(['Gender_Female', 'Gender_Male']).size()

# Crear el gráfico de barras
fig, ax = plt.subplots()
bars_rf = gender_correct_counts_rf.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax)

# Agregar etiquetas a cada barra
for i, v in enumerate(gender_correct_counts_rf):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10, color='black')

# Configurar título y etiquetas de ejes
plt.title('Instancias Clasificadas Correctamente por Género (Random Forest)')
plt.xlabel('Género')
plt.ylabel('Cantidad de Instancias Correctamente Clasificadas')

# Agregar etiquetas específicas a cada barra
ax.set_xticklabels(['Mujer', 'Hombre'], rotation=0)

# Mostrar el gráfico
plt.show()

# Imprimir resultados

# Resultados del Árbol de Decisión
print("Resultados del Árbol de Decisión:")
print(f'Accuracy (Árbol de Decisión): {accuracy_dt}')
print(f'Confusion Matrix (Árbol de Decisión):\n{conf_matrix_dt}')
print(f'Classification Report (Árbol de Decisión):\n{classification_rep_dt}')

# Resultados de Random Forest
print("\nResultados de Random Forest:")
print(f'Accuracy (Random Forest): {accuracy_rf}')
print(f'Confusion Matrix (Random Forest):\n{conf_matrix_rf}')
print(f'Classification Report (Random Forest):\n{classification_rep_rf}')
