#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#lire le fichier csv
data = pandas.read_csv("champignons.csv")

#transformer les catégories de chaque caractéristique comme des valeurs numériques
encodeur = LabelEncoder()
for col in data.columns:
    data[col] = encodeur.fit_transform(data[col])


#séparer les entrées (caractéristiques) et la sortie (classe)
y = data.iloc[:, 0]  #les résulats (classes)
X = data.iloc[:, 1:] #les caractéristiques
#print X.head()
#print y.head()

#diviser les données en deux ensembles: entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#construire le modèle
modele = MultinomialNB()

#entraîner le modèle
modele.fit(X_train, y_train)

#préduction d'un échantillon
echantillon = [X_test.iloc[0, :]]
#classe_reel = y_test[0]
print "echantillon: ", echantillon
#print "classe réelle: ", classe_reel
print "classe prédite: ", modele.predict(echantillon)
print "probabilités: ", modele.predict_proba(echantillon)

#Les prédictions
y_pred = modele.predict(X_test)

#évaluation de notre modèle
print "précision: ", accuracy_score(y_test, y_pred)

print "précision: ", modele.score(X_test, y_test)
