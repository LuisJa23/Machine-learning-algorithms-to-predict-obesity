import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar el conjunto de datos
data = pd.read_csv('../data/ObesityDataSet_raw_and_data_sinthetic.csv')

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = data.drop(['FCVC','NCP','CAEC','SMOKE','TUE','CALC','MTRANS','NObeyesdad'], axis= 1)
y = data['NObeyesdad']

# Convertir variables categóricas a variables dummy
X = pd.get_dummies(X, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'SCC'])

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar solo las columnas numéricas usando StandardScaler

numeric_cols = ['Age', 'Height', 'Weight', 'CH2O', 'FAF']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Crear el modelo de Regresión Logística
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_logistic = logistic_classifier.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
classification_rep_logistic = classification_report(y_test, y_pred_logistic)

# Imprimir resultados
print(f'Accuracy (Regresión Logística): {accuracy_logistic}')
print(f'Confusion Matrix (Regresión Logística):\n{conf_matrix_logistic}')
print(f'Classification Report (Regresión Logística):\n{classification_rep_logistic}')

# Crear un DataFrame con género, predicciones y etiquetas reales
result_df_logistic = pd.DataFrame({'Gender_Female': X_test['Gender_Female'], 'Gender_Male': X_test['Gender_Male'], 'Prediction': y_pred_logistic, 'Actual': y_test})

# Filtrar el DataFrame para obtener solo las instancias clasificadas correctamente
correctly_classified_df_logistic = result_df_logistic[result_df_logistic['Prediction'] == result_df_logistic['Actual']]

# Contar la cantidad total de instancias clasificadas correctamente por género
gender_correct_counts_logistic = correctly_classified_df_logistic.groupby(['Gender_Female', 'Gender_Male']).size()

# Crear el gráfico de barras
fig, ax = plt.subplots()
bars_logistic = gender_correct_counts_logistic.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax)

# Agregar etiquetas a cada barra
for i, v in enumerate(gender_correct_counts_logistic):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10, color='black')

# Configurar título y etiquetas de ejes
plt.title('Instancias Clasificadas Correctamente por Género (Regresión Logística)')
plt.xlabel('Género')
plt.ylabel('Cantidad de Instancias Correctamente Clasificadas')

# Agregar etiquetas específicas a cada barra
ax.set_xticklabels(['Mujer', 'Hombre'], rotation=0)

# Mostrar el gráfico
plt.show()