#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ici, on va utiliser scikit-learn

import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

#lire le fichier csv
data = pandas.read_csv("../../data/jouer.csv")

# séparer les données en: entrées et sorties
X = data.iloc[:,:-1] #les caractéristiques
y = data.iloc[:,-1]  #les résulats (classes)

X_dum = pandas.get_dummies(X)

# imprimer les premières lignes des données
print X_dum.head()

# créer un estimateur
estimator = DecisionTreeClassifier()
# entrainer l'estimateur
estimator.fit(X_dum, y)
# expoter l'arbre sous format graphviz
export_graphviz(estimator,
    out_file="arbre_cart0.dot",
    feature_names = X_dum.columns,
    class_names=estimator.classes_)
