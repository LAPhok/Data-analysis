import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 1. Charger les données
dataset = pd.read_csv(r'C:\BaseDeDonnee3\Remise\examenFinal\income_hapiness.csv')

# 2. Tracer le nuage de points
dataset.plot(x='income', y='happiness', style='o')
plt.title('Relation entre le revenu et le bonheur')
plt.xlabel('Revenu (millions de roupies)')
plt.ylabel('Bonheur (indice)')
plt.show()

# 3. Préparer les données
x = dataset[['income']].values
y = dataset['happiness'].values

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# 4. Entraîner le modèle de régression linéaire
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Afficher les coefficients
print('Intercept:', regressor.intercept_)
# Intercept: 0.09227908323332867
print('Coefficient:', regressor.coef_[0])
# Coefficient: 0.7322225764706634

# Faire des prédictions
y_pred = regressor.predict(X_test)

# Calculer l'erreur quadratique moyenne (MSE)
mse = metrics.mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
# Mean Squared Error: 0.5088435370300813

# 5. Prédire l'indice du bonheur pour un revenu de 11 millions de roupies
income_11m = np.array([11]).reshape(-1, 1)
predicted_happiness = regressor.predict(income_11m)
print('Indice du bonheur prédit pour un revenu de 11 millions de roupies:', predicted_happiness[0])
# Indice du bonheur prédit pour un revenu de 11 millions de roupies: 8.146727424410626
