# Chapitre II: Préparation des données

## Sommaire

[(Retour vers la page principale)](README.md)

- Chapitre II: Préparation des données
  - [II-1 Collection des données](#i-1-collection-des-données)
  - [II-2 Nétoyage des données](#ii-2-nétoyage-des-données)
  - [II-3 Transformation des données](#ii-3-transformation-des-données)
  - [II-4 Réduction des données](#ii-4-réduction-des-données)
  - [II-5 Outils de préparation des données](#ii-5-outils-de-préparation-des-données)
  - [II-6 Un peu de programmation](#ii-6-un-peu-de-programmation)

## II-1 Collection des données

### II-1-1 Qualité des données

Critères d'intégrité des données:
- Taille: Nombre des échantillons (enregistrements). Certaines tâches nécessitent une grande taille de données pour qu'elles soient appris.
- Le nombre et le type de caractéristiques (nominales, binaires, ordinales ou continues).
- Le nombre des erreurs d'annotation
- La quantité de bruits dans les données: erreurs et exceptions

### II-1-2 Intégration des données

Quand on veut collecter des données pour l'apprentissage automatique, souvent, on a besoin de combiner des données de différentes sources:
- Données structurées:
  - Bases de données
  - Fichiers de tabuleurs: CSV, etc.
- Données semi-structurées: XML, JSON, etc.
- Données non structurées: documents textes, images, métadonnées, etc.

Il faut, tout d'abord, vérifier l'intégrité des données:
- Vérifier que les fichiers XML sont conformes à leurs définitions [XSD](https://fr.wikipedia.org/wiki/XML_Schema)
- Vérifier que les séparateurs des colonnes dans les fichiers CSV sont correctes (point-virgule ou virgule et pas les deux au même temps).

Quand on joint deux schémas de données, on doit vérifier:
- Problème de nommage: il se peut qu'on ait des données identiques avec des nominations différentes. Par exemple, si on veut joindre deux tables de données **b1** et **b2** qui ont deux attributs avec le même sens mais différents noms **b1.numclient** et **b2.clientid**, on doit unifier les noms des attributs.
- Conflits de valeurs: les valeurs des attributs provenant de sources différentes sont représentées différemment. Par exemple, une source de données qui représente la taille en **cm** et une autre qui représente la taille en **pouces**.
- Redondance: les attributs qu'on puisse déduire des autres, les enregistrements identiques.

### II-1-3 Annotation des données



[(Sommaire)](#sommaire)

## II-2 Nétoyage des données

Les problèmes rencontrés dans les données peuvent être:

- Valeurs omises (données non disponibles): des échantilons (enregistrements) avec des caractéristiques (attributs) sans valeurs. Les causes, entre autres, peuvent être:
  - Mauvais fonctionnement de l'équipement
  - Incohérences avec d'autres données et donc supprimées
  - Non saisies car non (ou mal) comprises
  - Considérées peu importantes au moment de la saisie

- Échantillons dupliqués

- Des mauvaises annotations. Par exemple, un annotateur humain marque un échantillon comme "chat" or l'étiquette correcte est "chien".
  - Incohérence dans les conventions de nommage

- Bruit dans les données. Parmi ces causes:
  - Instrument de mesure défectueux
  - Problème de saisie
  - Problème de transmission

Pour régler ces problèmes:
- Valeurs omises:
  - Suppression
  - Saisie manuelle
  - Remplacement par une constante globale. Par exemple, "inconnu" pour les valeurs nominales ou "0" pour les valeurs numériques.
  - Remplacement par la moyenne dans le cas des valeurs numériques, en préférence de la même classe.
  - Remplacement par la valeur la plus fréquente dans le cas des valeurs nominales.
  - Remplacement par la valeur la plus probable.
- Échantillons dupliqués
  - Suppression
- Bruit (erreur ou variance aléatoire d'une variable mesurée):
  - Binning ou Bucketing (groupement des données par classe). Consulter "Transformation des données".
  - Clustering pour détecter les exceptions
  - Détection automatique des valeurs suspectes et vérification humaine.
  - Lisser les données par des méthodes de régression.

[(Sommaire)](#sommaire)

## II-3 Transformation des données

### II-3-1 Discrétisation en utilisant le groupement (binning, bucketing)

La discrétisation est le fait de convertir les caractéristiques numériques en caractéristiques nominales. Elle est utilisée pour simplifier l'exploitation des données dans certains types d'algorithmes.
- Dans le classifieur naif bayésien multinomial, les attributs doivent avoir des valeurs nominales.
- Certaines caractéristiques numériques sont utiles pour estimer une tâche, mais il n'existe aucune relation linéaire entre ces caractéristiques et cette tâche.

Prenant un exemple sur les prix des maisons suivant le latitude.
On ne peut pas trouver une fonction linéaire entre le latitude et le prix d'une maison, mais on sait que l'emplacement où cette maison se trouve affecte son prix.

| ![prix-maisons](https://developers.google.com/machine-learning/crash-course/images/ScalingBinningPart1.svg) |
|:--:|
| *Exemple sur les prix des maisons [ [Source](https://developers.google.com/machine-learning/crash-course/representation/cleaning-data) ]* |

On peut diviser la plage de latitude sur 11 parties (si on veut plus de précision, on peut augmenter le nombre des parties).

| ![prix-maisons2](https://developers.google.com/machine-learning/crash-course/images/ScalingBinningPart2.svg) |
|:--:|
| *Binning de latitude avec des plages égales [ [Source](https://developers.google.com/machine-learning/crash-course/representation/cleaning-data) ]* |

Donc, pour la caractéristique "latitude" les valeurs vont être représetée par une étiquette entre 1 et 11. Par exemple, ``
latitude 37.4 => 6
``

Dans le cas des algorithmes où on doit manipuler des valeurs numériques (comme les réseaux de neurones), on peut représenter chaque valeur comme un vecteur de 11 booléens (0 ou 1). Par exemple:
``
latitude 37.4 => [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
``

Dans certaines cas, diviser la plage d'une caractéristique en parties égales n'ai pas la bonne solution.
Supposant, nous avons un ensemble de données sur le nombre des automobiles vendues pour chaque prix.

| ![nbr-automobiles](https://developers.google.com/machine-learning/data-prep/images/bucketizing-needed.svg) |
|:--:|
| *Binning des prix des automobiles avec des plages identiques [ [Source](https://developers.google.com/machine-learning/data-prep/transform/bucketing) ]* |

On remarque qu'il y a un seul échantillon pour les prix > 45000. Pour fixer ça, on peut utiliser les quantiles: on divise notre jeu de données en intervalles contenant le même nombre de données.

| ![nbr-automobiles2](https://developers.google.com/machine-learning/data-prep/images/bucketizing-applied.svg) |
|:--:|
| *Binning des prix des automobiles par quantiles [ [Source](https://developers.google.com/machine-learning/data-prep/transform/bucketing) ]* |

### II-3-2 Normalisation

#### Mise à l'échelle min-max

La mise en échelle min-max transorme chaque valeur numérique *x* vers une autre valeur *x' ∈ [0,1]* en utilisant la valeur minimale et la valeur maximale dans les données. Cette normalisation conserve la distance proportionnelle entre les valeurs d'une caractéristique.

![II-3-min-max]

La mise à l'échelle min-max est un bon choix si ces deux conditions sont satisfaites:
- On sait les les limites supérieure et inférieure approximatives des valeurs de la caractéristique concernée (avec peu ou pas de valeurs aberrantes).
- Les valeurs sont presque uniformément réparties sur cette plage ( [min, max]).

Un bon exemple est l'âge. La plupart des valeurs d'âge se situent entre 0 et 90, et qui sont distribuées sur toute cette plage.

En revanche, utiliser cette normalisation sur le revenu est une mauvaise chose. Un petit nombre de personnes ont des revenus très élevés. Si on applique cette normalisation, la plupart des gens seraient réduits à une petite partie de l'échelle.

Cette normalisation offre plus d'avantages si les données se consistent de plusieurs caractéristiques. Ses intérets sont les suivants:
- Aider [l'algorithme du gradient](https://fr.wikipedia.org/wiki/Algorithme_du_gradient) (un algorithme d'optimisation) à converger plus rapidement.
- Eviter le problème des valeurs non définies lorsqu'une valeur dépasse la limite de précision en virgule flottante pendant l'entraînement.
- Apprendre les poids appropriés pour chaque caractéristique; si une caractéristique a un intervalle plus large que les autres, le modèle généré va favoriser cette caractéristique.

#### Coupure

S'il existe des valeurs aberrantes dans les extrémités d'une caractéristique, on applique une coupure max avec une valeur α et/ou min avec une valeur β.

![II-3-coupure]

Par exemple, dans le graphe suivant, qui illustre le nombre de cambres par personnes, on remarque qu'au delà de 4 les valeurs sont très baisses. La solution est d'appliquer une coupure max de 4.

| ![coupure](https://developers.google.com/machine-learning/data-prep/images/norm-clipping-outliers.svg) |
|:--:|
| *Nombre de chambres par personne: avant et après la coupure avec max de 4 personnes [ [Source](https://developers.google.com/machine-learning/data-prep/transform/normalization) ]* |

#### Mise à l'échelle log

Cette transformation est utile lorsque un petit ensemble de valeurs ont plusieurs points, or la plupart des valeurs ont moins de points. Elle sert à compresser la range des valeurs.

![II-3-log]

Par exemple, les évaluations par film.
Dans le schéma suivant, la plupart des films ont moins d'évaluations.

| ![log](https://developers.google.com/machine-learning/data-prep/images/norm-log-scaling-movie-ratings.svg) |
|:--:|
| *Normalisation log des évaluation des films [ [Source](https://developers.google.com/machine-learning/data-prep/transform/normalization) ]* |

#### Z-score

Le Z-score est utilisé pour assurer que la distribution d'une caractéristique ait une moyenne = 0 et un écart type = 1.
C'est utile quand il y a quelques valeurs aberrantes, mais pas si extrême qu'on a besoin d'appliquer une coupure.
Dans certaines ouvrages, cette transformation n'est pas classifiée comme une "normalisation" mais comme étant une "standardisation".
Cela est due au fait qu'elle transforme l'ancienne distribution à une distribution normale.

Etant donnée une caractéristique avec des valeurs *x*, les nouvelles valeurs *x'* peuvent être exprimer par *x*, la moyenne des valeurs *μ* et leurs écart type *σ*.

![II-3-z-score]

### II-3-3 Binarisation

Il existe des cas où on n'a pas besoin des fréquences (nombre d'occurences) d'une caractéristique pour créer un modèle; on a besoin seulement de savoir si cette caractéristique a apparue une fois au moins pour un échantillon. Dans le cas général, on veut vérifier si la fréquence a dépassé un certain seuil **a** ou non. Dans ce cas, on binarise les valeurs de cette caractéristique.

![II-3-bin]

Par exemple, si on veut construire un système de recommandation de chansons, on va simplement avoir besoin de savoir si une personne est intéressée ou a écouté une chanson en particulier.
Cela n'exige pas le nombre de fois qu'une chanson a été écoutée mais, plutôt, les différentes chansons que cette personne a écoutées.

### II-3-4 Créations de nouvelles caractéristiques (Les interactions)

Dans l'apprentissage automatique supervisés, en général, on veut modéliser la sortie (classes discrètes ou valeurs continues) en fonction des valeurs de caractéristiques en entrée.
Par exemple, une équation de régression linéaire simple peut modélise la sortie **y** en se basant sur les caractéristiques **xi** et leurs poids correspondants **wi** comme suit:

![II-3-reg]

Dans ce cas, on a modélisé la sortie en se basant sur des entrées indépendantes l'une de l'autre.
Cependant, souvent dans plusieurs scénarios réels, il est judicieux d'essayer également de capturer les interactions entre les caractéristiques.
Donc, on peut créer de nouvelles caractéristiques en multipliants les anciennes deux à deux (ou encore plus).
Notre équation de régression linéaire sera comme suit:

![II-3-reg2]


[II-3-min-max]: https://latex.codecogs.com/png.latex?x'=\frac{x-x_{min}}{x_{max}-x_{min}}
[II-3-z-score]: https://latex.codecogs.com/png.latex?x'=\frac{x-\mu}{\sigma}
[II-3-coupure]: https://latex.codecogs.com/png.latex?x'=\begin{cases}\alpha&si\;x\ge\alpha\\\\\beta&si\;x\le\beta\\\\x&sinon\end{cases}
[II-3-log]: https://latex.codecogs.com/png.latex?x'=\log(x)
[II-3-bin]: https://latex.codecogs.com/png.latex?x'=\begin{cases}1&si\;x>a\\\\0&sinon\end{cases}
[II-3-reg]: https://latex.codecogs.com/png.latex?y=w_1x_1+w_2x_2+...+w_nx_n
[II-3-reg2]: https://latex.codecogs.com/png.latex?y=w_1x_1+w_2x_2+...+w_nx_n+w_{11}x_1^2+w_{22}x_2^2+w_{12}x_1x_2+...

[(Sommaire)](#sommaire)

## II-4 Réduction des données

### II-4-1 Données imbalancées

### II-4-2 Partitionnement des données

### II-4-3 Randomisation

[(Sommaire)](#sommaire)

## II-5 Outils de préparation des données

| Outil | Licence | Langage |
| :---: | :---: | :---: |
| [pandas](https://pandas.pydata.org) | BSD | Python |
| [scikit-learn](https://scikit-learn.org/stable/) | BSD | Python |

[(Sommaire)](#sommaire)

## II-6 Un peu de programmation

### II-6-1 Lecture de données

Pour lire les données, on va utiliser la bibliothèque **pandas**.
Elle support [plusieurs types de fichiers](https://pandas.pydata.org/pandas-docs/stable/io.html): csv (read\_csv), JSON (read\_json), HTML (read\_html), MS Excel (read\_excel), SQL (read\_sql), etc.

```python
import pandas
```

Ici, on va utiliser l'ensemble des données [Census Income Data Set (Adult)](https://archive.ics.uci.edu/ml/datasets/Census+Income).
Les données se composent de 14 caractéristiques et de 48842 échantilons.
Dans le but de l'exercice, on a réduit le nombre des caractéristiques à 7 et quelques échantillons dispercés sur plusieurs formats de fichiers.

Le premier fichier (data/adult1.csv) est un fichier CSV avec des colonnes séparées par des virgules (50 échantilons).
Le fichier contient les colonnes suivantes (avec l'entête: titres des colonnes) dans l'ordre:
1. age: entier.
1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.  
1. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.  
1. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
1. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.  
1. sex: Female, Male.
1. hours-per-week: entier.
1. class: <=50K, >50K

On sait que le fichier est bien formé; donc, on ne va pas vérifier le format.
On ignore les espaces qui suivent les séparateurs.

```python
adult1 = pandas.read_csv("../../data/adult1.csv", skipinitialspace=True)
```

Le deuxième fichier est un fichier CSV lui aussi, mais les colonnes sont séparées par des points-virgules.
Le fichier est mal-formé; il existe des lignes avec séparation par virgules.
Aussi, il n'y a pas d'entête pour désigner les noms des colonnes (caractéristiques).
Voici le sens des colonnes dans l'ordre:
1. class: Y, N (il gagne plus de 50K)
1. age: entier
1. sex: F, M
1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
1. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
1. hours-per-week: entier.
1. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.

Dans ce cas, on va spécifier que le séparateur est ";", il n'y a pas d'entête (la première ligne contient des données et pas les noms des colonnes) et on spécifie les noms des colonnes.

```python
noms = ["class", "age", "sex", "workclass", "education", "hours-per-week", "marital-status"]
adult2 = pandas.read_csv("../../data/adult2.csv", skipinitialspace=True, sep=";", header=None, names=noms)
```

Pour plus d'options veuillez consulter la documentation de [read_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html).
Le problème qui se pose est qu'on va avoir des lignes avec des "NaN" (valeurs non définies).
On va régler ça dans l'étape de nétoyage des données. 



### II-6-2 Intégration des données

### II-6-3 Nétoyage des données

### II-6-4 Discrétisation

### II-6-5 Mise à l'échelle min-max

### II-6-6 Coupure

### II-6-7 Mise à l'échelle log

### II-6-8 Z-score

### II-6-9 Binarisation


[(Sommaire)](#sommaire)

## Bibliographie

- https://developers.google.com/machine-learning/data-prep/
- https://developers.google.com/machine-learning/crash-course/representation/video-lecture
- https://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/
- https://www.altexsoft.com/blog/datascience/preparing-your-dataset-for-machine-learning-8-basic-techniques-that-make-your-data-better/
- https://www.analyticsindiamag.com/get-started-preparing-data-machine-learning/
- https://docs.microsoft.com/fr-fr/azure/machine-learning/team-data-science-process/prepare-data
- https://www.simplilearn.com/data-preprocessing-tutorial
- https://pandas.pydata.org/pandas-docs/stable/io.html
