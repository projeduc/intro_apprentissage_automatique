#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ici, on va utiliser scikit-learn

import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.feature_extraction import DictVectorizer

#lire le fichier csv
data = pandas.read_csv("../../data/jouer.csv")

# séparer les données en: entrées et sorties
X = data.iloc[:,:-1] #les caractéristiques
y = data.iloc[:,-1]  #les résulats (classes)


# One Hot en utilisant pandas
# ============================

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

# DictVectorizer en utilisant scikit-learn
# =========================================

# Transformer X à une liste de dicts
list_dicts = X.T.to_dict().values()
# créer une instance du transformateur
vec = DictVectorizer()
# transformer
new_list_dicts = vec.fit_transform(list_dicts).toarray()
# créer un nouveau DataFrame
X_vec = pandas.DataFrame(new_list_dicts, columns=vec.get_feature_names())

# imprimer les premières lignes des données
print X_vec.head()

estimator.fit(X_vec, y)
# expoter l'arbre sous format graphviz
export_graphviz(estimator,
    out_file="arbre_cart1.dot",
    feature_names = X_vec.columns,
    class_names=estimator.classes_)
