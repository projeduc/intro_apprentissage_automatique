# Introduction à l'apprentissage automatique

[Cliquez Ici pour lire le document sous format Jupyter Nootebook](livre.ipynb)

Sommaire:
- [Chapitre I: Introduction](#chapitre-i-introduction)
- [Chapitre II: Préparation de données](#chapitre-ii-préparation-de-données)
- [Chapitre III: Classification naïve bayésienne](#chapitre-iii-classification-naïve-bayésienne)
- [Chapitre IV: Machine à vecteurs de support](#chapitre-iv-machine-à-vecteurs-de-support)
- [Chapitre V: Arbre de décision](#chapitre-v-arbre-de-décision)
- [Chapitre VI: Régression linéaire](#chapitre-vi-régression-linéaire)
- [Chapitre VII: Régression logistique](#chapitre-vii-régression-logistique)
- [Chapitre VIII: Perceptron](#chapitre-viii-perceptron)
- [Chapitre IX: Réseau de neurones artificiels](#chapitre-ix-réseau-de-neurones-artificiels)
- [Chapitre X: Regroupement K-Means](#chapitre-x-regroupement-k-means)
- [Chapitre XI: Auto-encodeurs (Maybe not!!)](#chapitre-XI: Auto-encodeurs)
- [Chapitre XII: Apprentissage par renforcement](#chapitre-xii-apprentissage-par-renforcement)

## Chapitre I: Introduction

### I-1 Motivation

- Certaines tâches sont difficiles à programmer manuellement: Reconnaissance de formes, Traduction par machine, Reconnaissance de la parole, Aide à la décision, etc.
- Les données sont disponibles, qui peuvent être utilisé pour estimer la fonction de notre tâche.

### I-2 Applications

- Santé:
  - Watson santé de IBM: https://www.ibm.com/watson/health/
  - Projet Hanover de Microsoft: https://hanover.azurewebsites.net
  - DeepMind santé de Google: https://deepmind.com/applied/deepmind-health/

- Finance : Prévention de fraude, management de risques, prédiction des investissements, etc.
- Domaine légal : cas de CaseText https://casetext.com
- Traduction: Google traslate https://translate.google.com/
- ... TO BE CONTINUED

### I-3 Types des algorithmes d'apprentissage

#### I-3-1 Apprentissage supervisé

##### Classification

##### Régression

#### I-3-2 Apprentissage non supervisé

##### Clustering (Regroupement)

##### Réduction de dimension

#### I-3-3 Apprentissage par renforcement  

### I-4 Limites de l'apprentissage automatique

### I-5 Outils de l'apprentissage automatique


## Chapitre II: Préparation de données

### II-1

### II-2

### Bibliographie

- https://developers.google.com/machine-learning/data-prep/
- https://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/
- https://www.altexsoft.com/blog/datascience/preparing-your-dataset-for-machine-learning-8-basic-techniques-that-make-your-data-better/

[vec-f]: https://latex.codecogs.com/gif.latex?\overrightarrow{f}
[c-i]: https://latex.codecogs.com/gif.latex?c_i
[f-j]: https://latex.codecogs.com/gif.latex?f_j

## Chapitre III: Classification naïve bayésienne

### III-1 Classification
Voici la théorème de Bayes:

![III-1-bayes]

Le problème de classification revient à estimer la probabilité de chaque classe ![c-i] en se basant sur un vecteur de critères ![vec-f].
Par example, nous voulons estimer la probabilité d'un animal étant: un chien, un chat, une vache ou autre (4 classes) en se basant sur un vecteur de critères: poids, longeur, longueur des pattes et le type de nouriture.
En appliquant la théorème de Bayes:

![III-1-bayes2]

Le dénominateur ne dépend pas de la classe ![c-i]

![III-1-bayes3]

En supposant l'indépedance entre les critères ![f-j] de ![vec-f]:

![III-1-bayes4]

Les probabilités calculées servent à sélectionner la classe la plus probable sachant notre vecteur de critères.
Donc, la classe estimée *c* est celle qui maximize la probabilité conditionnelle.

![III-1-bayes5]

![III-1-bayes6]

[III-1-bayes]: https://latex.codecogs.com/gif.latex?P(A|B)=\frac{P(B|A)P(A)}{P(B)}
[III-1-bayes2]: https://latex.codecogs.com/gif.latex?P(c_i|\overrightarrow{f})=\frac{P(\overrightarrow{f}|c_i)P(c_i)}{P(\overrightarrow{f})}
[III-1-bayes3]: https://latex.codecogs.com/gif.latex?P(c_i|\overrightarrow{f})&space;\propto&space;P(\overrightarrow{f}|c_i)&space;P(c_i)
[III-1-bayes4]: https://latex.codecogs.com/gif.latex?P(c_i|\overrightarrow{f})&space;\propto&space;P(c_i)&space;\prod\limits_{f_j&space;\in&space;\overrightarrow{f}}&space;P(f_j|c_i)
[III-1-bayes5]: https://latex.codecogs.com/gif.latex?c&space;=&space;\arg\max\limits_{ci}&space;P(c_i|\overrightarrow{f})
[III-1-bayes6]: https://latex.codecogs.com/gif.latex?c&space;=&space;\arg\max\limits_{ci}&space;P(c_i)&space;\prod\limits_{f_j&space;\in&space;\overrightarrow{f}}&space;P(f_j|c_i)

### III-2 Apprentissage (Estimation des paramètres du modèle)

Etant donné un ensemble de données, la probabilité d'apparaition d'une classe ![c-i] est estimée comme le nombre des exemplaires de ![c-i] divisé par le nombre total des examplaires dans cette ensemble.

![III-2-pci]

La probabilité de chaque critère ![f-j] sachant une classe ![c-i] est estimée selon le type de ces valeurs: discrètes, binaires ou continues.

#### Loi multinomiale

Si les valeurs de notre critère sont discrètes, on utilise la loi multinomiale.
Par exemple, la couleur des cheveux avec les valeurs: brun, auburn, châtain, roux, blond vénitien, blond et blanc.
La probabilité d'un critère ![f-j] sachant une classe ![c-i] est le nombre des occurrences de ce critère dans la classe  ( ![III-2-mult1] ) divisé par le nombre de ces occurrences dans tout l'ensemble de données.

![III-2-mult2]

#### Loi de Bernoulli
caractéristiques binaires

#### Loi normal
valeurs continues


[III-2-pci]: https://latex.codecogs.com/gif.latex?P(c_i)&space;=&space;\frac{|c_i|}{\sum_{c_j}&space;|c_j|}
[III-2-mult1]: https://latex.codecogs.com/gif.latex?|c_i|_{f_j}
[III-2-mult2]: https://latex.codecogs.com/gif.latex?P(f_j|c_i)&space;=&space;\frac{|c_i|_{f_j}}{\sum_{c_j}&space;|c_j|_{f_j}}

### III-3 Application (Points forts)

### III-4 Limites

### Bobliographie

- https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
- https://syncedreview.com/2017/07/17/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation/
- https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
- https://www.geeksforgeeks.org/naive-bayes-classifiers/
- https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c

## Chapitre IV: Machine à vecteurs de support

## Chapitre V: Machine à vecteurs de support

## Chapitre VI: Régression linéaire

## Chapitre VII: Régression logistique

## Chapitre VIII: Perceptron

## Chapitre IX: Réseau de neurones artificiels

## Chapitre X: Regroupement

## X-1 Regroupement hiérarchique
## X-2 K-Means

### Bibliographie

- https://towardsdatascience.com/unsupervised-learning-with-python-173c51dc7f03

## Chapitre XI: Auto-encodeurs (Maybe not!!)

## Chapitre XII: Apprentissage par renforcement
