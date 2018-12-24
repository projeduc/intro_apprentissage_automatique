#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas

#lire le fichier csv
data = pandas.read_csv("../../data/champignons.csv")
#affichier les 6 premiers échantillons
print data.head(6)

print
print "Le nombre des valeurs absentes pour chaque caractéristique"
print data.isnull().sum()

print
print "Les classes possibles"
print data['classe'].unique()

print
print "Le nombre des lignes et des colonnes"
print data.shape
