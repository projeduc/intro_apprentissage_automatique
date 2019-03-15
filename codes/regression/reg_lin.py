#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

#lire le fichier csv
data = pandas.read_csv("../../data/maisons_taiwan.csv")

# supprimer la date
data.drop(columns=["date"], inplace=True)

# une fonction pour binariser une colonne
# donnee: le dataframe
# colnom: le nom de la colonne
def binariser(donnee, colnom):
    # sélectinner la colonne et calculer la moyenne
    moy = donnee[colnom].mean()
    # remplacer les valeurs supérieures à la moyenne par 1
    # et le reste par 0
    donnee[colnom] = (donnee[colnom] > moy).astype(float)

# binariser latitude
binariser(data, "latitude")
# binariser longitude
binariser(data, "longitude")

# séparer les données en: entrées et sorties
X = data.iloc[:,:-1] #les caractéristiques
y = data.iloc[:,-1]  #les résulats (classes)

#diviser les données en deux ensembles: entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#construire le modèle
modele = LinearRegression().fit(X_train, y_train)

# afficher les coefficients
print "Coefficients: ", modele.coef_

# prédire les résulats des échantillons de test
y_pred = modele.predict(X_test)

# Evaluation du modèle
print "L'erreur quadratique moyenne: ", mean_squared_error(y_test, y_pred)
print "Score R2: ", r2_score(y_test, y_pred)

#construire le modèle prix = lineaire(age)
age_m = LinearRegression().fit(X_train[["age"]], y_train)

yl_age = age_m.predict(X_test[["age"]])

#construire le modèle prix = poly(age)

# créer des nouvelles caractéristiques
poly = PolynomialFeatures(degree=10, include_bias=False)
age_train = poly.fit_transform(X_train[["age"]])
age_test = poly.fit_transform(X_test[["age"]])
# entrainer un modèle linéaire
age_pm = LinearRegression().fit(age_train, y_train)
# estimer les prix à partir des données de teste
yp_age = age_pm.predict(age_test)


plt.scatter(X_test["age"], y_test, color="black")
new_x, new_y = zip(*sorted(zip(X_test["age"], yl_age)))
plt.plot(new_x, new_y, "v-")
#plt.plot(X_test["age"], yl_age, "v-")
new_x, new_y = zip(*sorted(zip(X_test["age"], yp_age)))
plt.plot(new_x, new_y, "o-")
#plt.plot(X_test["age"], yp_age, "o-")
plt.legend([u"linéaire", u"polynomiale"])

plt.xlabel("age")
plt.ylabel("prix")
plt.title(u"Régression")
plt.grid()
plt.show()
