# Introduction à l'apprentissage automatique

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
- [Chapitre XI: Auto-encodeur](#chapitre-xi*auto-encodeur)
- [Chapitre XII: Apprentissage par renforcement](#chapitre-xii-apprentissage-par-renforcement)

Glossaire:
- caractéristique: feature
- entraînement: training


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
**_...TODO: Add more_**

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

[vec-f]: https://latex.codecogs.com/png.latex?\overrightarrow{f}
[c-i]: https://latex.codecogs.com/png.latex?c_i
[f-j]: https://latex.codecogs.com/png.latex?f_j
[vec-C]: https://latex.codecogs.com/png.latex?\overrightarrow{C}

## Chapitre III: Classification naïve bayésienne

### III-1 Classification

Voici le théorème de Bayes:

![III-1-bayes]

Le problème de classification revient à estimer la probabilité de chaque classe ![c-i] sachant un vecteur de caractéristiques ![vec-f].
Par exemple, on veut estimer la probabilité d'un animal étant: un chien, un chat, une vache ou autre (4 classes) en utilsant quelques caractéristiques: poids, longueur, longueur des pattes et le type de nourriture.
En appliquant le théorème de Bayes:

![III-1-bayes2]

Le dénominateur ne dépend pas de la classe ![c-i]

![III-1-bayes3]

En supposant l'indépendance entre les critères ![f-j] de ![vec-f] (d'où le nom: naïve):

![III-1-bayes4]

Les probabilités calculées servent à sélectionner la classe la plus probable sachant un vecteur de caractéristiques donné.
Donc, la classe estimée (*c* ) est celle qui maximise la probabilité conditionnelle.

![III-1-bayes5]

![III-1-bayes6]

Techniquement, on utilise l'espace logarithmique puisque le produit des probabilités converge vers zéro.

![III-1-bayes7]

[III-1-bayes]: https://latex.codecogs.com/png.latex?\overbrace{P(A|B)}^{\text{post\'erieure}}=\frac{\overbrace{P(A)}^{\text{ant\'erieure}}\overbrace{P(B|A)}^{\text{vraisemblance}}}{\underbrace{P(B)}_{\text{\'evidence}}}
[III-1-bayes2]: https://latex.codecogs.com/png.latex?P(c_i|\overrightarrow{f})=\frac{P(c_i)P(\overrightarrow{f}|c_i)}{P(\overrightarrow{f})}
[III-1-bayes3]: https://latex.codecogs.com/png.latex?P(c_i|\overrightarrow{f})&space;\propto&space;P(\overrightarrow{f}|c_i)&space;P(c_i)
[III-1-bayes4]: https://latex.codecogs.com/png.latex?P(c_i|\overrightarrow{f})&space;\propto&space;P(c_i)&space;\prod\limits_{f_j&space;\in&space;\overrightarrow{f}}&space;P(f_j|c_i)
[III-1-bayes5]: https://latex.codecogs.com/png.latex?c&space;=&space;\arg\max\limits_{ci}&space;P(c_i|\overrightarrow{f})
[III-1-bayes6]: https://latex.codecogs.com/png.latex?c&space;=&space;\arg\max\limits_{ci}&space;P(c_i)&space;\prod\limits_{f_j&space;\in&space;\overrightarrow{f}}&space;P(f_j|c_i)
[III-1-bayes7]: https://latex.codecogs.com/png.latex?c=\arg\max\limits_{ci}\;\log(P(c_i))+\sum\limits_{f_j\in\overrightarrow{f}}\log(P(f_j|c_i))

### III-2 Apprentissage

Étant donné un ensemble de données d'entraînement avec *N* échantillons, la probabilité d'apparition d'une classe ![c-i] est estimée comme étant le nombre de ses échantillons divisé par le nombre total des échantillons d'entraînement.

![III-2-pci]

La probabilité de chaque caractéristique ![f-j] sachant une classe ![c-i] est estimée selon le type de ces valeurs: discrètes, binaires ou continues.

#### Loi multinomiale

Lorsque les valeurs des caractéristiques sont discrètes, on utilise la loi multinomiale.
Par exemple, la couleur des cheveux avec les valeurs: brun, auburn, châtain, roux, blond vénitien, blond et blanc.
La probabilité d'une caractéristique ![f-j] sachant une classe ![c-i] est le nombre des occurrences de ce critère dans la classe  ( ![III-2-mult1] ) divisé par le nombre de ces occurrences dans tout l'ensemble de données.

![III-2-mult2]

Certaines caractéristiques peuvent ne pas se figurer dans une classe donnée, ce qui va donner une probabilité nulle.
Pour remedier à ce problème, on peut utiliser une fonction de lissage comme le lissage de Lidstone.

![III-2-mult3]

Où: |![vec-f]| est le nombre des caractéristiques.
Alpha: est un nombre dans l'intervalle ]0, 1]. Lorsque sa valeur égale à 1, on appelle ça le laaissage de Laplace.

#### Loi de Bernoulli

Lorsque les valeurs des caractéristiques sont binaires, on utilise la loi de Bernoulli.
Par exemple,

**_...TODO: Complete one day!!_**

#### Loi normal

Lorsque les valeurs des caractéristiques sont continues, on utilise la loi normale (loi gaussienne).
Par exemple, le poids, le prix, etc.
En se basant sur les données d'entraînement avec *N* échantillons, on calcule l'espérance *μ* et la variance *σ²* de chaque caractéristique ![f-j] et chaque classe ![c-i].

![III-2-mu]

![III-2-sigma]

Donc, la probabilité qu'une caractéristique ![f-j] ait une valeur *x* sachant une classe ![c-i] suit la loi normale.

![III-2-normal]

[III-2-pci]: https://latex.codecogs.com/png.latex?P(c_i)&space;=&space;\frac{|c_i|}{N}
[III-2-mult1]: https://latex.codecogs.com/png.latex?|c_i|_{f_j}
[III-2-mult2]: https://latex.codecogs.com/png.latex?P(f_j|c_i)&space;=&space;\frac{|c_i|_{f_j}}{\sum_{c_j}&space;|c_j|_{f_j}}
[III-2-mult3]: https://latex.codecogs.com/png.latex?P(f_j|c_i)&space;=&space;\frac{|c_i|_{f_j}+\alpha}{\sum_{c_j}&space;|c_j|_{f_j}+\alpha|\overrightarrow{f}|}
[III-2-mu]: https://latex.codecogs.com/png.latex?\mu_{ij}=\frac{1}{|c_i|}\sum\limits_{k=1}^{|c_i|}x_k\;/\;x_k\;\in\;f_j\,\cap\,c_i
[III-2-sigma]: https://latex.codecogs.com/png.latex?\sigma^2_{ij}=\frac{1}{|c_i|-1}\sum\limits_{k=1}^{|c_i|}(x_k-\mu_{ij})^2\;/\;x_k\;\in\;f_j\,\cap\,c_i
[III-2-normal]: https://latex.codecogs.com/png.latex?P(f_j=x|c_i)=\frac{1}{2\pi\sigma^2_{ij}}e^{-\frac{(x-\mu_{ij})^2}{2\sigma^2_{ij}}}

### III-3 Exemple

### III-4 Avantages

Les classifieurs naïfs bayésiens, malgès leurs simplicité, ont des points forts:

- Ils ont besoin d'une petite quantité de données d’entraînement.
- Ils sont très rapides par rapport aux autres classifieurs.
- Ils donnent de bonnes résulats dans le cas de filtrage du courrier indésirable et de classification de documents.

### III-5 Limites

Les classifieurs naïfs bayésiens certes sont populaires à cause de leur simplicité.
Mais, une telle simplicité vient avec un coût.

- Les probabilités obtenues en utilisant ces classifieurs ne doivent pas être prises au sérieux.
- S'il existe une grande corrélation entre les caractéristiques, ils vont donner une mauvaise performance.
- Dans le cas des caractéristiques continues (prix, surface, etc.), les données doivent suivre la loi normale.

### Bibliographie

- https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
- https://syncedreview.com/2017/07/17/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation/
- https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
- https://www.geeksforgeeks.org/naive-bayes-classifiers/
- https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
- https://scikit-learn.org/stable/modules/naive_bayes.html
- https://github.com/ctufts/Cheat_Sheets/wiki/Classification-Model-Pros-and-Cons

## Chapitre IV: Machine à vecteurs de support

## Chapitre V: Machine à vecteurs de support

## Chapitre VI: Régression linéaire

## Chapitre VII: Régression logistique

## Chapitre VIII: Perceptron


### Bibliographie
- https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975

## Chapitre IX: Réseau de neurones artificiels

## Chapitre X: Regroupement

## X-1 Regroupement hiérarchique

## X-2 K-Means

### Bibliographie

- https://towardsdatascience.com/unsupervised-learning-with-python-173c51dc7f03

## Chapitre XI: Auto-encodeur (Maybe not!!)

## Chapitre XII: Apprentissage par renforcement
