import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree

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

# Crear el modelo de Árbol de Decisión
dt_classifier = DecisionTreeClassifier(max_depth=5)
dt_classifier.fit(X_train, y_train)

# Visualizar el árbol de decisión con zoom
plt.figure(figsize=(60, 50))
plot_tree(dt_classifier, filled=True, feature_names=X_train.columns, class_names=['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'])

# Ajustar los límites para hacer zoom
plt.xlim([-0.5, 5.5])
plt.ylim([-0.5, 100])

plt.show()

# Realizar predicciones en el conjunto de prueba
y_pred_dt = dt_classifier.predict(X_test)

# Evaluar el rendimiento del modelo de Árbol de Decisión
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
classification_rep_dt = classification_report(y_test, y_pred_dt)

# Guardar el modelo entrenado en un archivo
joblib.dump(dt_classifier, 'decision_tree_model_prueba1.joblib')

# Imprimir resultados del Árbol de Decisión
print(f'Accuracy (Árbol de Decisión): {accuracy_dt}')
print(f'Confusion Matrix (Árbol de Decisión):\n{conf_matrix_dt}')
print(f'Classification Report (Árbol de Decisión):\n{classification_rep_dt}')

# Crear un DataFrame con género, predicciones y etiquetas reales
result_df_dt = pd.DataFrame({'Gender_Female': X_test['Gender_Female'], 'Gender_Male': X_test['Gender_Male'], 'Prediction': y_pred_dt, 'Actual': y_test})

# Filtrar el DataFrame para obtener solo las instancias clasificadas correctamente
correctly_classified_df_dt = result_df_dt[result_df_dt['Prediction'] == result_df_dt['Actual']]

# Contar la cantidad total de instancias clasificadas correctamente por género
gender_correct_counts_dt = correctly_classified_df_dt.groupby(['Gender_Female', 'Gender_Male']).size()

# Crear el gráfico de barras
fig, ax = plt.subplots()
bars_dt = gender_correct_counts_dt.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax)

# Agregar etiquetas a cada barra
for i, v in enumerate(gender_correct_counts_dt):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10, color='black')

# Configurar título y etiquetas de ejes
plt.title('Instancias Clasificadas Correctamente por Género (Árbol de Decisión)')
plt.xlabel('Género')
plt.ylabel('Cantidad de Instancias Correctamente Clasificadas')

# Agregar etiquetas específicas a cada barra
ax.set_xticklabels(['Mujer', 'Hombre'], rotation=0)

# Mostrar el gráfico
plt.show()
