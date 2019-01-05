#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ici, on va présenter les outils d'évaluation
# pour le classement multi-classes

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 20 chats, 20 chiens et 20 vaches
reel = [0] * 20 + [1] * 20 + [2] * 20
# les 20 chats sont prédites comme 10 chats, 8 chiens et 2 vaches
predit = [0] * 10 + [1] * 8 + [2] * 2
# les 20 chiens sont prédites comme 5 chats, 13 chiens et 2 vaches
predit += [0] * 5 + [1] * 13 + [2] * 2
# les 20 vaches sont prédites comme 0 chats, 3 chiens et 17 vaches
predit += [1] * 3 + [2] * 17

# récupérer et afficher la matrice de confusion
mat_conf = confusion_matrix(reel, predit)
print mat_conf

print "La justesse: ", accuracy_score(reel, predit)

print "La précision: ", precision_score(reel, predit, average="macro")
print "Le rappel: " , recall_score(reel, predit, average="macro")
print "La mesure F1: " , f1_score(reel, predit, average="macro")

print "La précision (chat, chien): ", precision_score(reel, predit, labels=[0, 1], average="macro")

print "corrélation de matthews: ", matthews_corrcoef(reel, predit)



#rapport de classification
noms_classes = ["desirable", "indesirable"]
print(classification_report(reel, predit, target_names=noms_classes))
