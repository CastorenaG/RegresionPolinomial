import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función para calcular el Error Cuadrado Medio (MSE)
def calcular_MSE(X_train, y_train, X_test, y_test, degree):
    # Ajustar el modelo de regresión polinomial en el conjunto de entrenamiento
    coef = np.polyfit(X_train, y_train, degree)
    
    # Calcular las predicciones en el conjunto de prueba
    y_pred = np.polyval(coef, X_test)
    
    # Calcular el Error Cuadrado Medio (MSE)
    mse = np.mean((y_test - y_pred) ** 2)
    
    return mse

df = pd.read_csv('c:\\polynomial-regression.csv')

X = df['araba_fiyat'].values
y = df['araba_max_hiz'].values

n = len(X)

# Grado de polinomio a probar
degree = 4  

# Generar una serie de valores de X suavizados
X_smooth = np.linspace(X.min(), X.max(), 100)

# Elegir el índice de un punto de datos para mostrar su gráfica
i = 0  

X_train = np.delete(X, i)  # Dejar uno fuera para validación cruzada
y_train = np.delete(y, i)
X_test = X[i]
y_test = y[i]

mse = calcular_MSE(X_train, y_train, X_test, y_test, degree)

# Ajustar y graficar un modelo de regresión polinomial con el grado especificado
plt.figure(figsize=(12, 8))
plt.scatter(X, y, label='Datos reales', color='b')

coef = np.polyfit(X, y, degree)
y_values = np.polyval(coef, X_smooth)  # Usar X_smooth para valores suavizados
label = f'Grado {degree}, MSE = {mse:.2f}'  # Imprimir el MSE en la etiqueta
plt.plot(X_smooth, y_values, label=label)

plt.xlabel('Araba Fiyat')
plt.ylabel('Araba Max Hız')
plt.legend()
plt.show()
