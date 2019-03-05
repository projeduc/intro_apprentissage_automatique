#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ici, on va utiliser l'outil: https://github.com/svaante/decision-tree-id3
# pip install decision-tree-id3 --user

import pandas
import numpy as np
from id3 import Id3Estimator
from id3 import export_graphviz
from sklearn.preprocessing import OrdinalEncoder


#lire le fichier csv
data = pandas.read_csv("../../data/jouer0.csv")

# séparer les données en: entrées et sorties
X = data.iloc[:,:-1] #les caractéristiques
y = data.iloc[:,-1]  #les résulats (classes)

enc = OrdinalEncoder()

enc.fit(X, y)

X_ = enc.transform(X)

# créer un estimateur
estimator = Id3Estimator()
estimator.fit(enc, y)
export_graphviz(estimator.tree_, 'tree.dot', data.columns.tolist())
