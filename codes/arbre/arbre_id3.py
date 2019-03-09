#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ici, on va utiliser l'outil: https://github.com/svaante/decision-tree-id3
# pip install decision-tree-id3 --user
# On a modifié un peu dans le programme
# afin qu'il puisse accepter les chaines de caractères
# 

import pandas
from id3a import Id3Estimator
from id3a import export_graphviz

#lire le fichier csv
data = pandas.read_csv("../../data/jouer0.csv")

# séparer les données en: entrées et sorties
X = data.iloc[:,:-1] #les caractéristiques
y = data.iloc[:,-1]  #les résulats (classes)


# créer un estimateur
estimator = Id3Estimator()
# entrainer l'estimateur
estimator.fit(X, y)
# expoter l'arbre sous format graphviz
export_graphviz(estimator.tree_, "resultat.dot", data.columns.tolist())
