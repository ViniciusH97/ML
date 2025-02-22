import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Criação de dados

np.random.seed(0)
x = 2 * np.random.rand(100, 1) # aqui ira receber números aleatórios de 1 a 100
y = 4 + 3 * x + np.random.randn(100, 1)


# mostrar os dados
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('dados gerados')
plt.show()

#divisão dos dados em conjuntos para treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#criação do modelo
modelo = LinearRegression()

#Treinando o modelo
modelo.fit(x_train, y_train)

#fazendo previsões
y_previsao = modelo.predict(x_test)

#Mostrar as previsoes

# Visualizar as previsões
plt.scatter(x_test, y_test, color='blue', label='Dados Reais')
plt.plot(x_test, y_previsao, color='red', label='Previsões')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Previsões vs Dados Reais')
plt.legend()
plt.show()

# Mostrar os coeficientes do modelo
print(f'Coeficiente: {modelo.coef_}')
print(f'Intercept: {modelo.intercept_}')