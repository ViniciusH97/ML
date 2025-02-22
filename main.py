import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Criação de dados

np.random.seed(0)
x = 2 * np.random.rand(100, 1) # aqui ira receber números aleatórios de 1 a 100
y = 4 + 3 * x + np.random.randn(100, 1)

