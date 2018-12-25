#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ici, on va utiliser des outils existants
# Et programmer le reste (ce qu'on a pas pu trouver dans ces outils)
# Le but est de se familiariser avec ces outils

import pandas
import sqlite3
import numpy
from lxml import etree

#lire le premier fichier (CSV)
adult1 = pandas.read_csv("../../data/adult1.csv", skipinitialspace=True)

#lire le deuxième fichier (CSV)
#     - séparateur (;)
#     - pas d'entête (la première ligne contient des données)
noms = ["class", "age", "sex", "workclass", "education", "hours-per-week", "marital-status"]
adult2 = pandas.read_csv("../../data/adult2.csv", skipinitialspace=True, sep=";", header=None, names=noms)

#lire les données à partir d'un fichier Sqlite
#Les ? doivent être Considérées comme des NaN
con = sqlite3.connect("../../data/adult3.db")
adult3 = pandas.read_sql_query("SELECT * FROM income", con)
adult3 = adult3.replace('?', numpy.nan)

#valider le fichier XML
parser = etree.XMLParser(dtd_validation=True)
arbre = etree.parse("../../data/adult4.xml", parser)

def valeur_noeud(noeud):
    return noeud.text if noeud is not None else numpy.nan

noms2 = ["id", "age", "workclass", "education", "marital-status", "sex", "hours-per-week", "class"]
adult4 = pandas.DataFrame(columns=noms2)

for candidat in arbre.getroot():
    idi = candidat.get("id")
    age = valeur_noeud(candidat.find('age'))
    workclass = valeur_noeud(candidat.find('workclass'))
    education = valeur_noeud(candidat.find('education'))
    marital = valeur_noeud(candidat.find('marital-status'))
    sex = valeur_noeud(candidat.find('sex'))
    hours = valeur_noeud(candidat.find('hours-per-week'))
    klass = valeur_noeud(candidat.find('class'))

    adult4 = adult4.append(
        pandas.Series([idi, age, workclass, education, marital, sex, hours, klass],
        index=noms2), ignore_index=True)


# Renommer les caractéristiques
adult3.rename(columns={'num': 'id', 'hours-per-day': 'hours-per-week'}, inplace=True)

# Ordonner les caractéristiques
ordre = ["age", "workclass", "education", "marital-status", "sex", "hours-per-week", "class"]
adult1 = adult1.reindex(ordre + ["occupation"], axis=1)
#print adult1.head()
adult2 = adult2.reindex(ordre, axis=1)
adult3 = adult3.reindex(ordre + ["id"], axis=1)
adult4 = adult4.reindex(ordre + ["id"], axis=1)

# concaténer les enregistrements des deux tables
adult34 = pandas.concat([adult3, adult4], ignore_index=True)
# définir le type de "id" comme étant entier, et remplacer la colonne
adult34["id"] = pandas.to_numeric(adult34["id"], downcast='integer')
# ordonner les enregistrements par "id"
adult34 = adult34.sort_values(by="id")
# regrouper les par "id", et pour chaque groupe remplacer les
# valeurs absentes par une valeur précédente dans le même groupe
adult34 = adult34.groupby("id").ffill()
# supprimer les enregistrements dupliqués
# on garde les derniers, puisqu'ils sont été réglés
adult34.drop_duplicates('id', keep='last', inplace=True)
