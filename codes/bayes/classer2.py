#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ici, on va construire un classifieur baïf bayésien multinomial de zéro
# Le but est de comprendre comment l'algorithme fonctionne

if __name__ == "__main__":
    import classer2 as c
    data = c.lire_csv("../../data/champignons.csv")
    print data[:4, :2]


def lire_csv(url):
    """
    Une fonction pour extraire une matrice à partir d'un fichier CSV
    séparé par des virgules
    """
    #Ouverture du fichier en mode lecture
    f = open(url,"r")
    #Lecture de toutes les lignes du fichier
    lignes = f.readlines()
    #Fermeture du fichier
    f.close()
    #Création d'un tableau qui contient d'autres tableaux
    #Ces derniers sont créés en éclatant chaque ligne par la virgule
    return [l.split(",") for l in lignes]

"""
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
"""
