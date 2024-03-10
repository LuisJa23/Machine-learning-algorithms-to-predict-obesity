import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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

# Crear el modelo de Red Neuronal Artificial
# Se define una arquitectura con una capa oculta de 10 neuronas
ann_classifier = MLPClassifier(hidden_layer_sizes=(30,), activation='relu', solver='adam', max_iter=2000)
ann_classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_ann = ann_classifier.predict(X_test)

# Evaluar el rendimiento del modelo de Red Neuronal Artificial
accuracy_ann = accuracy_score(y_test, y_pred_ann)
conf_matrix_ann = confusion_matrix(y_test, y_pred_ann)
classification_rep_ann = classification_report(y_test, y_pred_ann)

# Guardar el modelo entrenado en un archivo
joblib.dump(ann_classifier, 'red_neuronal_model_prueba2.joblib')

# Imprimir resultados de la Red Neuronal Artificial
print(f'Accuracy (Red Neuronal Artificial): {accuracy_ann}')
print(f'Confusion Matrix (Red Neuronal Artificial):\n{conf_matrix_ann}')
print(f'Classification Report (Red Neuronal Artificial):\n{classification_rep_ann}')

# Crear un DataFrame con género, predicciones y etiquetas reales
result_df_ann = pd.DataFrame({'Gender_Female': X_test['Gender_Female'], 'Gender_Male': X_test['Gender_Male'], 'Prediction': y_pred_ann, 'Actual': y_test})

# Filtrar el DataFrame para obtener solo las instancias clasificadas correctamente
correctly_classified_df_ann = result_df_ann[result_df_ann['Prediction'] == result_df_ann['Actual']]

# Contar la cantidad total de instancias clasificadas correctamente por género
gender_correct_counts_ann = correctly_classified_df_ann.groupby(['Gender_Female', 'Gender_Male']).size()

# Crear el gráfico de barras
fig, ax = plt.subplots()
bars_ann = gender_correct_counts_ann.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax)

# Agregar etiquetas a cada barra
for i, v in enumerate(gender_correct_counts_ann):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10, color='black')

# Configurar título y etiquetas de ejes
plt.title('Instancias Clasificadas Correctamente por Género (Red Neuronal Artificial)')
plt.xlabel('Género')
plt.ylabel('Cantidad de Instancias Correctamente Clasificadas')

# Agregar etiquetas específicas a cada barra
ax.set_xticklabels(['Mujer', 'Hombre'], rotation=0)

# Mostrar el gráfico
plt.show()
