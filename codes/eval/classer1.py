#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ici, on va présenter les outils d'évaluation
# pour le classement binaire

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# créer une liste de 3 "1" suivis de 17 "0"
reel = [1] * 3 + [0] * 17
print reel
# créer une list [1, 0, 0, 1] concaténée avec 16 "0"
predit = [1, 0, 0, 1] + [0] * 16
print predit

print "La justesse: " + str(accuracy_score(reel, predit))
print "La précision: " + str(precision_score(reel, predit))
print "Le rappel: " + str(recall_score(reel, predit))
print "La mesure F1: " + str(f1_score(reel, predit))
print "corrélation de matthews: " + str(matthews_corrcoef(reel, predit))

# récupérer et afficher la matrice de confusion
mat_conf = confusion_matrix(reel, predit)
print mat_conf
# récupérer le nombre des VN, FP, FN, TP
vn, fp, fn, vp = mat_conf.ravel()
print vn, fp, fn, vp

#rapport de classification
noms_classes = ["desirable", "indesirable"]
print(classification_report(reel, predit, target_names=noms_classes))
