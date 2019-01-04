#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ici, on va présenter les outils d'évaluation
# pour le classement binaire

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

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
