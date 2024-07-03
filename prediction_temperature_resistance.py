import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Charger les données
dataset = pd.read_csv(r'C:\BaseDeDonnee3\Remise\quiz5\temperature_resistance.csv')

# 1. Tracer le nuage de points
dataset.plot(x='temperature', y='resistance', style='*')
plt.title('Relation entre la temperature et la resistance des pieces')
plt.xlabel('Temperature')
plt.ylabel('Resistance')
plt.show()

# 2. Préparer les données
x = dataset[['temperature']].values
y = dataset['resistance'].values

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 3. Entraîner le modèle de régression linéaire
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Afficher les coefficients
print('Intercept:', regressor.intercept_)
print('Coefficient:', regressor.coef_[0])

# 4. Faire des prédictions
y_pred = regressor.predict(X_test)

# Calculer l'erreur quadratique moyenne (MSE)
mse = metrics.mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# 5. Prédire la résistance pour des températures de 400 et 800 degrés Celsius
temps = np.array([400, 800]).reshape(-1, 1)
predicted_resistances = regressor.predict(temps)
print('Résistance prédite pour 400°C:', predicted_resistances[0])
print('Résistance prédite pour 800°C:', predicted_resistances[1])
