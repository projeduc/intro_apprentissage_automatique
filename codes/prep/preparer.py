#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ici, on va utiliser des outils existants
# Et programmer le reste (ce qu'on a pas pu trouver dans ces outils)
# Le but est de se familiariser avec ces outils

import pandas


#lire le premier fichier (CSV)
adult1 = pandas.read_csv("../../data/adult1.csv", skipinitialspace=True)

#lire le deuxième fichier (CSV)
#     - séparateur (;)
#     - pas d'entête (la première ligne contient des données)
noms = ["class", "age", "sex", "workclass", "education", "hours-per-week", "marital-status"]
adult2 = pandas.read_csv("../../data/adult2.csv", skipinitialspace=True, sep=";", header=None, names=noms)
