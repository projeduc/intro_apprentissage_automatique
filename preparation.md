# Chapitre II: Préparation des données

## Sommaire


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

Il existe des cas où on n'a pas besoin des fréquences (nombre d'occurences) d'une caractéristique pour créer un modèle; on a besoin seulement de savoir si cette caractéristique a apparue une fois au moins pour un échantillon. Dans ce cas, on binarise les valeurs de cette caractéristique.

![II-3-bin]

Par exemple, si on veut construire un système de recommandation de chansons, on va simplement avoir besoin de savoir si une personne est intéressée ou a écouté une chanson en particulier.
Cela n'exige pas le nombre de fois qu'une chanson a été écoutée mais, plutôt, les différentes chansons que cette personne a écoutées.


[II-3-min-max]: https://latex.codecogs.com/png.latex?x'=\frac{x-x_{min}}{x_{max}-x_{min}}
[II-3-z-score]: https://latex.codecogs.com/png.latex?x'=\frac{x-\mu}{\sigma}
[II-3-coupure]: https://latex.codecogs.com/png.latex?x'=\begin{cases}\alpha&si\;x\ge\alpha\\\\\beta&si\;x\le\beta\\\\x&sinon\end{cases}
[II-3-log]: https://latex.codecogs.com/png.latex?x'=\log(x)
[II-3-bin]: https://latex.codecogs.com/png.latex?x'=\begin{cases}1&si\;x\ge1\\\\0&sinon\end{cases}

## II-4 Réduction des données

### II-4-1 Données imbalancées

### II-4-2 Partitionnement des données

### II-4-3 Randomisation

## II-5 Outils de préparation des données

| Outil | Licence | Langage |
| :---: | :---: | :---: |
| [pandas](https://pandas.pydata.org) | BSD | Python |
| [scikit-learn](https://scikit-learn.org/stable/) | BSD | Python |

## II-6 Un peu de programmation

### II-6-1 Lecture de données

### II-6-2 Nétoyage de données

### II-6-3 Discrétisation

## Bibliographie

- https://developers.google.com/machine-learning/data-prep/
- https://developers.google.com/machine-learning/crash-course/representation/video-lecture
- https://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/
- https://www.altexsoft.com/blog/datascience/preparing-your-dataset-for-machine-learning-8-basic-techniques-that-make-your-data-better/
- https://www.analyticsindiamag.com/get-started-preparing-data-machine-learning/
- https://docs.microsoft.com/fr-fr/azure/machine-learning/team-data-science-process/prepare-data
- https://www.simplilearn.com/data-preprocessing-tutorial
