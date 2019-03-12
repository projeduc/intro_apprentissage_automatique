# Chapitre V: Régression

## Sommaire

[(Retour vers la page principale)](README.md)

- Chapitre V: Régression
  - [V-1 Motivation](#v-1-motivation)
  - [V-2 Régression linéaire](#v-2-régression-linéaire)
  - [V-3 Régression polynomiale](#v-3-régression-polynomiale)
  - [V-4 Régression logistique](#v-4-régression-logistique)
  - [V-5 Avantages](#v-5-avantages)
  - [V-6 Limites](#v-6-limites)
  - [V-7 Un peu de programmation](#v-7-un-peu-de-programmation)

## V-1 Motivation

La régression sert à trouver la relation d'une variable par rapport à une ou plusieurs autres.

Dans l'apprentissage automatique, le but de la régression est d'estimer une valeur (numérique) de sortie à partir des valeurs d'un ensemble de caractéristiques en entrée.
Par exemple, estimer le prix d'une maison en se basant sur sa surface, nombre des étages, son emplacement, etc.
Donc, le problème revient à estimer une fonction de calcul en se basant sur des données d'entrainement.

![V-1-fct]

Il existe plusieurs algorithmes pour la régression:
- Régression linéaire
- Régression polynomiale
- Régression logistique
- Régression quantile
- etc.


[V-1-fct]: https://latex.codecogs.com/png.latex?y=f(x_1,x_2,...,x_n)

[(Sommaire)](#sommaire)

## V-2 Régression linéaire

La régression linéaire simple sert à trouver unne relation d'une variable de sortie (continue) par rapport à une autre.

![V-2-lineare0]

Ici, on va présenter la régression linéaire multiple, qui est une version étendue de la régression linéaire simple.
Etant donnée que cette version est très simple et on aime torturer ceux qui lisent cette petite introduction.
Aussi, savoir comment la version généralisée se fonctionne est plus intéressant.

### V-2-1 Principe

La régression linéaire multiple a comme but de décrire la variation d'une variable dépendante (*y*) associée aux variations de plusieurs variables indépendantes.
Dans le contexte de l'apparentissage automatique, elle sert à estimer une fonction linéaire entre la sortie (avec des valeurs continues, numériques) et les entrées.
La fonction est écrite comme suit:

![V-2-lineare]

Où:

- *y* est la sortie (résulat),
- *xi* est une caractéristique d'entrée,
- *wi* est le poids de cette caractéristique

Dans ce cas, l'apprentissage est le fait d'estimer ces poids en se basant sur des données d'entrées et des résulats attendus.

### V-2-2 La fonction du coût

La fonction du coût aide à trouver l'erreur entre le résulat estimé et le résultat attendu.
Elle est utilisée pour régler les poids des caractéristiques.
Donc, pour trouver les poids les plus optimals, il faut minimiser cette fonction.

Etant donnée un ensemble des données d'entrainement avec *N* échantillons, la fonction du coût la plus utilisée est l'erreur quadratique moyenne (MSE) entre les sorties attendues (*y*) et les sorties estimées (*y^*)

![V-2-mse]

Cette fonction est une fonction convexe; ça veut dire qu'elle n'a pas des minimums locaux.
Donc, elle a un minimum global unique.


### V-2-3 Algorithme du gradient

L'algorithme du gradient est la partie la plus importante dans l'apprentissage automatique par régression linéaire.
Il est utilisé pour mettre à jour les poids de la fonction linéaire en se basant sur la fonction du coût.
C'est un algorithme itératif qui met à jour les poids à chaque itération pour minimiser la fonction du coût.
L'algorithme du gradient est le suivant:

1. Initialiser les poids *wi* à 0. Fixer un pas *α* pour mettre à jour les poids. Aussi, Fixer un seuil de tolérance *ε > 0*.
1. Calculer les gradients de la fonction du coût en *wi*
1. Mettre à jours les poids *wi* en utilisant leurs anciennes valeurs, leurs gradients et le pas *α*
1. Si la fonction du coût *E <= ε* on s'arrête; sinon, on revient à l'étape (2).

#### Le pas

Le pas *α* est une valeur connue entre 0 et 1. *α ∈ ]0, 1]*.
- Si le pas est grand, on risque de manquer la solution optimale.
- S'il est petit, l'algorithme prend du temps à converger.

Il y a une technique pour mettre à jour le pas dynamiquement.
- Si le coût se baisse, augmenter le pas
- Si le coût s'augmente, diminuer le pas

#### Critère d'arrêt

Le seuil de tolérance *ε* est la valeur minimale acceptable pour le coût.
Lorsque le coût atteint ce seuil, on s'arrête.

Lorsque le pas est grand, on peut manquer le minimum.
Dans ce cas, on s'arrête s'il n'y a plus d'amélioration en terme de coût.

Une autre technique est de fixer le nombre maximum des itérations.


#### Les gradients

Le gradient de chaque poids *wi* est calculé en utilisant le dérivé partiel de la fonction du coût par rapport à ce poids.
Donc, le gradient d'un poids *wi* est calculé comme suit:

![V-2-grad]

Le gradient de *w0* est calculé comme suit:

![V-2-grad0]


#### Mise à jour des poids

Les poids sont mis à jours en se basant sur les gradients et le pas comme suit:

![V-2-maj]


### V-2-4 Exemple

On va utiliser un exemple simple pour estimer la fonction suivante:

![V-2-exp]

![exemple de régression](IMG/reg_exp.png)

en se basant sur ces données:

| x | y |
| :---: | :---: |
| 0.0 | 1.0 |
| 0.25 | 0.9571067811865476 |
| 0.5 | 0.5000000000000001 |
| 0.75 | 0.04289321881345254 |
| 1.0 | 0.0 |
| 1.25 | 0.5428932188134523 |
| 1.5 | 1.4999999999999998 |
| 1.75 | 2.4571067811865475 |
| 2.0 | 3.0 |
| 2.25 | 2.957106781186548 |
| 2.5 | 2.5000000000000004 |
| 2.75 | 2.0428932188134534 |
| 3.0 | 2.0 |
| 3.25 | 2.542893218813453 |
| 3.5 | 3.4999999999999996 |
| 3.75 | 4.457106781186546 |


![Exemple regression lineaire](IMG/lin_reg_exp.png)

[V-2-exp]: https://latex.codecogs.com/png.latex?f(x)=x*cos(x*\pi)
[V-2-lineare]: https://latex.codecogs.com/png.latex?y=w_0+w_1x_1+w_2x_2+...+w_nx_n
[V-2-lineare0]: https://latex.codecogs.com/png.latex?y=w_0+w_1x_1
[V-2-mse]: https://latex.codecogs.com/png.latex?E=\frac{1}{N}\sum\limits_{i=1}^{N}(\hat{y}-y)^2
[V-2-grad]: https://latex.codecogs.com/png.latex?\frac{\partial{E}}{\partial{w_i}}=\frac{2}{N}\sum\limits_{i=1}^{N}x_i(\hat{y}-y)
[V-2-grad0]: https://latex.codecogs.com/png.latex?\frac{\partial{E}}{\partial{w_0}}=\frac{2}{N}\sum\limits_{i=1}^{N}(\hat{y}-y)
[V-2-maj]: https://latex.codecogs.com/png.latex?w_i=w_i-\alpha*\frac{\partial{E}}{\partial{w_i}}

[(Sommaire)](#sommaire)

## V-3 Régression polynomiale

La régression polynomiale est un cas spécial de la régression linéaire.
On peut créer de nouvelles caractéristiques dans l'étape de préparation des données en multipliant les valeurs des anciennes caractéristiques.
Par exemple, La régression polynomiale d'ordre 2 sera:

![V-3-poly]

Suivant l'exemple précédent, en applicant la régression polynomiale avec un degrée de 4.

![Exemple regression polynomiale](IMG/poly_reg_exp.png)

[V-3-poly]: https://latex.codecogs.com/png.latex?y=w_0+w_1x_1+w_2x_2+...+w_nx_n+w_{11}x_1^2+...+w_{nn}x_n^2+w_{12}x_1x_2+...

[(Sommaire)](#sommaire)

## V-4 Régression logistique

La régression logistique est utilisée pour le classement et pas la régression.
Mais, elle est considéré comme une méthode de régression puisqu'elle sert à estimer la probabilité d'appartenir à une classe.

[(Sommaire)](#sommaire)

## V-5 Avantages

[(Sommaire)](#sommaire)

## V-6 Limites

[(Sommaire)](#sommaire)


## V-7 Un peu de programmation

## V-7-1 Description des données

On va utiliser l'ensemble des données [Real estate valuation Data Set ](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set).
Ce sont des données pour estimer les prix des maisons (Sindian Dist., New Taipei City, Taiwan.) en se basant sur 7 caractéristiques:

- date: la date de transaction (par exemple: 2013.250=2013 March, 2013.500=2013 June, etc.)
- age: l'age de la maison en nmbre d'années (nombre réel).
- metro: la distance à la station de métro la plus proche (en mètre).
- epicerie: nombre des épiceries près de la maison (nombre entier).
- latitude: latitude en degrée
- longitude: longitude en degrée

La sortie est le prix de la maison par unité (10000 New Taiwan Dollar/Ping, où Ping est l'unité locale, 1 Ping = 3.3 mètres carrés).

On a créé un fichier CSV contenant ces données: [data/maisons_taiwan.csv](data/maisons_taiwan.csv).

[(Sommaire)](#sommaire)

## Bibliographie

- https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a
- https://medium.com/cracking-the-data-science-interview/a-tour-of-the-top-10-algorithms-for-machine-learning-newbies-7228aa8ef541
- https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/
- https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/
- https://towardsdatascience.com/selecting-the-best-machine-learning-algorithm-for-your-regression-problem-20c330bad4ef
- https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a
- https://scikit-learn.org/stable/modules/linear_model.html
- https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931
- http://www.cs.toronto.edu/~hinton/csc2515/notes/lec6tutorial.pdf
