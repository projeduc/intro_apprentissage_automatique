# Chapitre III: Classification naïve bayésienne

La classification est un apprentissage supervisé; ce qui veut dire, on doit entrainer notre système sur un ensemble de données, ensuite on utilise ce modèle pour classer des données de test.

Ici, on va commencer par présenter la phase de classification avant la phase d'entrainement.

[vec-f]: https://latex.codecogs.com/png.latex?\overrightarrow{f}
[c-i]: https://latex.codecogs.com/png.latex?c_i
[f-j]: https://latex.codecogs.com/png.latex?f_j
[vec-C]: https://latex.codecogs.com/png.latex?\overrightarrow{C}

## Sommaire

[(Retour vers la page principale)](README.md)

- Chapitre III: Classification naïve bayésienne
  - [III-1 Classification](#iii-1-classification)
  - [III-2 Apprentissage](#iii-2-apprentissage)
  - [III-3 Exemples](#iii-3-exemples)
  - [III-4 Avantages](#iii-4-avantages)
  - [III-5 Limites](#iii-5-limites)
  - [III-6 un peu de programmation](#iii-6-un-peu-de-programmation)

## III-1 Classification

Voici le théorème de Bayes:

![III-1-bayes]

Le problème de classification revient à estimer la probabilité de chaque classe ![c-i] sachant un vecteur de caractéristiques ![vec-f].
Par exemple, on veut estimer la probabilité d'un animal étant: un chien, un chat, une vache ou autre (4 classes) en utilisant quelques caractéristiques: poids, longueur, longueur des pattes et le type de nourriture.
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

[III-1-bayes]: https://latex.codecogs.com/png.latex?\overbrace{P(A&#124;B)}^{\text{post\'erieure}}=\frac{\overbrace{P(A)}^{\text{ant\'erieure}}\overbrace{P(B&#124;A)}^{\text{vraisemblance}}}{\underbrace{P(B)}_{\text{\'evidence}}}
[III-1-bayes2]: https://latex.codecogs.com/png.latex?P(c_i&#124;\overrightarrow{f})=\frac{P(c_i)P(\overrightarrow{f}&#124;c_i)}{P(\overrightarrow{f})}
[III-1-bayes3]: https://latex.codecogs.com/png.latex?P(c_i&#124;\overrightarrow{f})&space;\propto&space;P(\overrightarrow{f}&#124;c_i)&space;P(c_i)
[III-1-bayes4]: https://latex.codecogs.com/png.latex?P(c_i&#124;\overrightarrow{f})&space;\propto&space;P(c_i)&space;\prod\limits_{f_j&space;\in&space;\overrightarrow{f}}&space;P(f_j&#124;c_i)
[III-1-bayes5]: https://latex.codecogs.com/png.latex?c&space;=&space;\arg\max\limits_{ci}&space;P(c_i&#124;\overrightarrow{f})
[III-1-bayes6]: https://latex.codecogs.com/png.latex?c&space;=&space;\arg\max\limits_{ci}&space;P(c_i)&space;\prod\limits_{f_j&space;\in&space;\overrightarrow{f}}&space;P(f_j&#124;c_i)
[III-1-bayes7]: https://latex.codecogs.com/png.latex?c=\arg\max\limits_{ci}\;\log(P(c_i))+\sum\limits_{f_j\in\overrightarrow{f}}\log(P(f_j&#124;c_i))

[(Sommaire)](#sommaire)

## III-2 Apprentissage

Étant donné un ensemble de données d'entrainement avec *N* échantillons, la probabilité d'apparition d'une classe ![c-i] est estimée comme étant le nombre de ses échantillons divisé par le nombre total des échantillons d'entrainement.

![III-2-pci]

La probabilité de chaque caractéristique ![f-j] sachant une classe ![c-i] est estimée selon le type de ces valeurs: discrètes, binaires ou continues.

### Loi multinomiale

Lorsque les valeurs des caractéristiques sont discrètes, on utilise la loi multinomiale.
Par exemple, la couleur des cheveux avec les valeurs: brun, auburn, châtain, roux, blond vénitien, blond et blanc.
La probabilité d'une caractéristique ![f-j] sachant une classe ![c-i] est le nombre des occurrences de ce critère dans la classe  ( ![III-2-mult1] ) divisé par le nombre de ces occurrences dans tout l'ensemble de données.

![III-2-mult2]

Certaines caractéristiques peuvent ne pas se figurer dans une classe donnée, ce qui va donner une probabilité nulle.
Pour remédier à ce problème, on peut utiliser une fonction de lissage comme le lissage de Lidstone.

![III-2-mult3]

Où: |![vec-f]| est le nombre des caractéristiques.
Alpha: est un nombre dans l'intervalle ]0, 1]. Lorsque sa valeur égale à 1, on appelle ça le laissage de Laplace.

### Loi de Bernoulli

Lorsque les valeurs des caractéristiques sont binaires, on utilise la loi de Bernoulli.
Par exemple,

**_...TODO: Complete one day!!_**

### Loi normal

Lorsque les valeurs des caractéristiques sont continues, on utilise la loi normale (loi gaussienne).
Par exemple, le poids, le prix, etc.
En se basant sur les données d'entrainement avec *N* échantillons, on calcule l'espérance *μ* et la variance *σ²* de chaque caractéristique ![f-j] et chaque classe ![c-i].

![III-2-mu]

![III-2-sigma]

Donc, la probabilité qu'une caractéristique ![f-j] ait une valeur *x* sachant une classe ![c-i] suit la loi normale.

![III-2-normal]

[III-2-pci]: https://latex.codecogs.com/png.latex?P(c_i)&space;=&space;\frac{&#124;c_i&#124;}{N}
[III-2-mult1]: https://latex.codecogs.com/png.latex?&#124;c_i&#124;_{f_j}
[III-2-mult2]: https://latex.codecogs.com/png.latex?P(f_j&#124;c_i)&space;=&space;\frac{&#124;c_i&#124;_{f_j}}{\sum_{c_j}&space;&#124;c_j&#124;_{f_j}}
[III-2-mult3]: https://latex.codecogs.com/png.latex?P(f_j|c_i)&space;=&space;\frac{|c_i|_{f_j}+\alpha}{\sum_{c_j}&space;|c_j|_{f_j}+\alpha|\overrightarrow{f}|}
[III-2-mu]: https://latex.codecogs.com/png.latex?\mu_{ij}=\frac{1}{&#124;c_i&#124;}\sum\limits_{k=1}^{&#124;c_i&#124;}x_k\;/\;x_k\;\in\;f_j\,\cap\,c_i
[III-2-sigma]: https://latex.codecogs.com/png.latex?\sigma^2_{ij}=\frac{1}{&#124;c_i&#124;-1}\sum\limits_{k=1}^{&#124;c_i&#124;}(x_k-\mu_{ij})^2\;/\;x_k\;\in\;f_j\,\cap\,c_i
[III-2-normal]: https://latex.codecogs.com/png.latex?P(f_j=x&#124;c_i)=\frac{1}{2\pi\sigma^2_{ij}}e^{-\frac{(x-\mu_{ij})^2}{2\sigma^2_{ij}}}

[(Sommaire)](#sommaire)

## III-3 Exemples

### Classifieur naïf bayésien suivant la loi normal

Empruntons l'exemple de [Wikipédia](https://fr.wikipedia.org/wiki/Classification_naïve_bayésienne#Classification_des_sexes)

On veut classer une personne donnée en féminin ou masculin selon la taille, le poids et la pointure.
Donc, le vecteur de caractéristique ![vec-f]={taille, poids, pointure} et le vecteur de classes ![vec-c]={féminin, masculin}.
Les données d'apprentissage contiennent 8 échantillons.

| Sexe | Taille (cm) | Poids (kg) |	Pointure (cm) |
| :---: | :---: | :---: | :---: |
|masculin | 182 | 81.6 | 30 |
| masculin| 180| 86.2| 28|
| masculin| 170| 77.1| 30|
| masculin| 180| 74.8| 25|
| féminin| 152| 45.4| 15|
| féminin| 168| 68.0| 20|
| féminin| 165| 59.0| 18|
| féminin| 175| 68.0| 23|

**_Apprentissage:_** La phase d'apprentissage consiste à calculer l'espérance et la variance de chaque caractéristique et classe.

P(masculin) = 4/8 = 0.5

P(féminin) = 4/8 = 0.5

| Sexe | μ (taille) | σ² (taille) | μ (poids) | σ² (poids) | μ (pointure) | σ² (pointure) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| masculin | 178 | 29.333 | 79.92 | 25.476 | 28.25 | 5.5833 |
| féminin | 165 | 92.666 | 60.1 | 114.04 | 19.00 | 11.333 |

**_test:_** On a un échantillons avec les caractéristiques suivantes {taille=183, poids=59, pointure=20}. On veut savoir si c'est féminin ou masculin.

**_...TODO: continue_**

**_...TODO: example about sentiment analysis using multinomial bayes_**

[(Sommaire)](#sommaire)

## III-4 Avantages

Les classifieurs naïfs bayésiens, malgré leurs simplicité, ont des points forts:

- Ils ont besoin d'une petite quantité de données d'entrainement.
- Ils sont très rapides par rapport aux autres classifieurs.
- Ils donnent de bonnes résultats dans le cas de filtrage du courrier indésirable et de classification de documents.

[(Sommaire)](#sommaire)

## III-5 Limites

Les classifieurs naïfs bayésiens certes sont populaires à cause de leur simplicité.
Mais, une telle simplicité vient avec un cout.

- Les probabilités obtenues en utilisant ces classifieurs ne doivent pas être prises au sérieux.
- S'il existe une grande corrélation entre les caractéristiques, ils vont donner une mauvaise performance.
- Dans le cas des caractéristiques continues (prix, surface, etc.), les données doivent suivre la loi normale.

[(Sommaire)](#sommaire)

## III-6 un peu de programmation

Dans cet exercice, on veut classifier les champignons comme toxiques ou non.
On va utiliser l'ensemble de données [mushroom classification](https://archive.ics.uci.edu/ml/datasets/mushroom) pour classer les champignons comme comestibles ou toxiques en se basant sur 22 caractéristiques nominales.
Le fichier est de type CSV contenant 8124 échantillons. Voici la description des colonnes:

1. classe: (e) [edible] comestible; (p) [poisonous] toxique
1. chapeau-forme: (b) [bell]; (c) [conical] conique; (x) [convex] convexe; (f) [flat] plat; (k) [knobbed] noué; (s) [sunken] enfoncé
1. chapeau-surface: (f) [fibrous] fibreuse; (g) [grooves] rainures; (y) [scaly] écailleuse; (s) [smooth] lisse
1. chapeau-couleur: brown=n, buff=b, cinnamon=c, gray=g, green=r,
1. ecchymoses: true=t, false=f
1. odeur: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
1. branchie-attachement: attached=a,descending=d,free=f,notched=n
1. branchie-espacement:             close=c,crowded=w,distant=d
1. branchie-taille:                broad=b,narrow=n
1. branchie-couleur:               black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
1. tige-forme:              enlarging=e,tapering=t
1. tige-racine:               bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
1. tige-surface-dessus-anneau: fibrous=f,scaly=y,silky=k,smooth=s
1. tige-surface-dessous-anneau: fibrous=f,scaly=y,silky=k,smooth=s
1. tige-couleur-dessus-anneau:   brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
1. tige-couleur-dessous-anneau:   brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
1. voile-type:                partial=p,universal=u
1. voile-couleur:               brown=n,orange=o,white=w,yellow=y
1. anneau-nombre:              none=n,one=o,two=t
1. anneau-type:                cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
1. spore-couleur:        black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
1. population:               abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
1. habitat:                  grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

### Tester s'il y a des valeurs manquantes

Consulter le fichier [codes/bayes/tester.py](codes/bayes/tester.py)

Avant tout, on importe le fichier, et on affiche les 6 premières lignes (plus la première qui contient les noms des caractéristiques) afin de vérifier l'importation nos données.

```python
import pandas

#lire le fichier csv
data = pandas.read_csv("champignons.csv")
#affichier les 6 premiers échantillons
print data.head(6)

```

On peut vérifier s'il y a des valeurs manquantes dans les échantillons. Ici, on va afficher le nombre de valeurs manquantes pour chaque caractéristique.

```python
print data.isnull().sum()

```

On peut afficher, également, les différentes catégories possibles pour une caractéristique donnée. Ici, on veut afficher les classes possibles. Attention! si on utilise un nom de caractéristique qui n'existe pas dans les données, on va avoir une erreur.

```python
print data['classe'].unique()

```

On peut vérifier le nombre des lignes et des colonnes. Bien sûre, il faut soustraire une colonne si on veut savoir le nombre des caractéristiques (sans les classes de sortie).

```python
print data.shape

```

### Classifieur naïf bayésien multinomial

Consulter le fichier [codes/bayes/classer.py](codes/bayes/classer.py)

On va lire le fichier en utilisant l'outil **pandas**

```python
data = pandas.read_csv("../../data/champignons.csv")
```

Avant d'utiliser le classifieur naïf bayésien de **scikit-learn**, on doit transformer les catégories de chaque caractéristique en valeurs numériques.
Ceci est possible en utilisant un encodeur des étiquettes (*LabelEncoder*)

```python
from sklearn.preprocessing import LabelEncoder
encodeur = LabelEncoder()
for col in data.columns:
    data[col] = encodeur.fit_transform(data[col])

```

On sépare les données en: entrées (les caractéristiques) et sorties (les classes: comestible ou toxique).
Dans notre fichier, les classes (qui sont le résultat attendu) sont dans la colonne 0, et les autres caractéristiques (les entrées) sont dans les colonnes restantes.

```python
X = data.iloc[:,1:23] #les caractéristiques
y = data.iloc[:, 0]  #les résulats (classes)
```

Ensuite, il faut séparer les données en deux partie: une pour l'entrainement (on prend 80%) et une pour le test (on prend 20%).
On va utiliser [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) de scikit-learn.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

Toutes les caractéristiques sont nominales, donc on va utiliser **MultinomialNB**. il peut avoir comme parametres:
- **alpha** : float, optional (default=1.0).
C'est le lissage de Laplace/Linstone. Si on ne veut pas appliquer un lissage, on met 0.
- **fit_prior** : boolean, optional (default=True).
Est-ce qu'on calcule la probabilité apriori ou non. Si non, les probabilités des classes seront considérées comme uniformes.
- **class_prior** : array-like, size (n_classes,), optional (default=None).
Une liste des probabilités apriori prédéfinies.
La méthode **fit** va entraîner le modèle en fournissant les valeurs des caractéristiques et leurs classes.

```python
from sklearn.naive_bayes import MultinomialNB
modele = MultinomialNB()
modele.fit(X_train, y_train)
```

Pour prédir les classes d'un ensemble d'échantillons (ici, les données de test), on utilise la méthode **predict**.

```python
y_pred = modele.predict(X_test)
```

Enfin, on teste la précision de notre modèle.

```python
from sklearn.metrics import accuracy_score
print "précision: ", accuracy_score(y_test, y_pred)

```

Ou, on peut utiliser la méthode **score** qui donne le même résulat

```python
print "précision: ", modele.score(X_test, y_test)
```

### Sauvegarder le modèle

Après avoir entrainé un modèle, il est souhaitable de le conserver pour un usage ultérieur sans avoir besoin d'entrainer une deuxième fois.
Il y a deux façons de le faire selon [la doc de scikit-learn ](https://scikit-learn.org/stable/modules/model_persistence.html):
- la sérialisation pickle
- la sérialisation joblib

La deuxième est recommandée par scikit-learn.
Après avoir entrainer notre modèle, on le sauvegarde.

```python
from joblib import dump
...
modele.fit(X_train, y_train)
dump(modele, 'mon_modele.joblib')
```

Lorsqu'on veut prédire une classe en utilisant ce modèle, on le relance.

```python
from joblib import load
...
modele = load('mon_modele.joblib')
y_pred2 = modele.predict(X_test2)
```

[(Sommaire)](#sommaire)

## Bibliographie

- https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
- https://syncedreview.com/2017/07/17/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation/
- https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
- https://www.geeksforgeeks.org/naive-bayes-classifiers/
- https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
- https://scikit-learn.org/stable/modules/naive_bayes.html
- https://github.com/ctufts/Cheat_Sheets/wiki/Classification-Model-Pros-and-Cons
- https://mattshomepage.com/articles/2016/Jun/07/bernoulli_nb/
- https://scikit-learn.org/stable/modules/model_persistence.html
