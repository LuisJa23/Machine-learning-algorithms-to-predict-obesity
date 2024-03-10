import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

# Crear el modelo de Máquinas de Soporte Vectorial (SVM)
svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_svm = svm_classifier.predict(X_test)

# Evaluar el rendimiento del modelo SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
classification_rep_svm = classification_report(y_test, y_pred_svm)

# Guardar el modelo entrenado en un archivo
joblib.dump(svm_classifier, 'svm_model_prueba1.joblib')

# Imprimir resultados SVM
print(f'Accuracy (SVM): {accuracy_svm}')
print(f'Confusion Matrix (SVM):\n{conf_matrix_svm}')
print(f'Classification Report (SVM):\n{classification_rep_svm}')

# Crear un DataFrame con género, predicciones y etiquetas reales
result_df_svm = pd.DataFrame({'Gender_Female': X_test['Gender_Female'], 'Gender_Male': X_test['Gender_Male'], 'Prediction': y_pred_svm, 'Actual': y_test})

# Filtrar el DataFrame para obtener solo las instancias clasificadas correctamente
correctly_classified_df_svm = result_df_svm[result_df_svm['Prediction'] == result_df_svm['Actual']]

# Contar la cantidad total de instancias clasificadas correctamente por género
gender_correct_counts_svm = correctly_classified_df_svm.groupby(['Gender_Female', 'Gender_Male']).size()

# Crear el gráfico de barras
fig, ax = plt.subplots()
bars_svm = gender_correct_counts_svm.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax)

# Agregar etiquetas a cada barra
for i, v in enumerate(gender_correct_counts_svm):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10, color='black')

# Configurar título y etiquetas de ejes
plt.title('Instancias Clasificadas Correctamente por Género (SVM)')
plt.xlabel('Género')
plt.ylabel('Cantidad de Instancias Correctamente Clasificadas')

# Agregar etiquetas específicas a cada barra
ax.set_xticklabels(['Mujer', 'Hombre'], rotation=0)

# Mostrar el gráfico
plt.show()
