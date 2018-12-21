# Introduction à l'apprentissage automatique

## Sommaire

- [Chapitre I: Introduction](#chapitre-i-introduction)
- [Chapitre II: Préparation des données](#chapitre-ii-préparation-des-données)
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

## Glossaire

- **caractéristique** [feature] Dans une base  de donnée, c'est l'équivalent de "attribut"
- **échantillon** [sample] Dans une base de données, c'est l'équivalent de "enregistrement"
- **entraînement** [training]

## Chapitre I: Introduction

| ![apprentissage automatique](IMG/AA.png) |
|:--:|
| *Apprentissage automatique* |

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

Lorsque nous disposons d'un ensemble de données avec les résulats attendus, on peut entraîner un système sur ces données pour inférer la fonction utilisée pour avoir ces résulats.
En résumé:

- **Source d'apprentissage:** des données annotées (nous avons les résultats attendus)
- **Retour d'information:** direct; à partir des résulats attendues.
- **Fonction:** prédire les future résultats

| ![apprentissage supervisé](IMG/AA-supervise.svg)|
|:--:|
| *Apprentissage supervisé* |

Selon le type d'annotation, on peut remarquer deux types des algorithmes d'apprentissage automatique: classement et régression.

##### Classement (Classification supervisée)

Lorsque le résulat attendu est une classe (groupe).

| Par exemple: |
| :--: |
| Classer un animal comme: chat, chien, vache ou autre en se basant sur le poids, la longueur et le type de nourriture.  |

##### Régression

Lorsque le résulat attendu est une valeur.

| Par exemple: |
| :--: |
| Estimer le prix d'une maison à partir de sa surface, nombre de chambre et l'emplacement. |


#### I-3-2 Apprentissage non supervisé

Lorsque nous disposons d'un ensemble de données non annotées (sans savoir les résulats attendus).
En résumé:

- **Source d'apprentissage:** des données non annotées
- **Retour d'information:** pas de retour; on dispose seulement des données en entrée.
- **Fonction:** rechercher les structures cachées dans les données.

Selon le type de structure que l'algorithme va découvrir, on peut avoir: le regroupement et la réduction de dimension.

##### Clustering (Regroupement)

L'algorithme de regroupement sert à assigner les échantillons similaires dans le même groupe.
Donc, le résulat est un ensemble de groupes contenants les échantillons.

| Par exemple: |
| :--: |
| Regrouper les plantes similaires en se basant sur la couleur, la taile, etc.  |

##### Réduction de dimension

L'algorithme de réduction de dimension a comme but d'apprendre comment représenter des données en entrée avec moins de valeurs.

| Par exemple: |
| :--: |
| Représenter des individus sur un graph de deux dimensions en utilisant la taille, le poids, l'age, la couleur des cheveux, la texture des cheveux et la couleur des yeux  |


#### I-3-3 Apprentissage par renforcement

- **Source d'apprentissage:** le processus de décision
- **Retour d'information:** un système de récompense
- **Fonction:** recherche des structures cachées dans les données.

| ![apprentissage par renforcement](IMG/RL-fr.png) |
|:--:|
| *Apprentissage par renforcement [ [Wikimedia](https://commons.wikimedia.org/wiki/File:Reinforcement_learning_diagram_fr.svg?uselang=fr) ]* |


### I-4 Limites de l'apprentissage automatique

- Pour des tâches complexes, on a besoin d'une grande quantité de données
- Dans le cas de l'apprentissage supervisé, l'annotation de données est une tâche fastidieuse; qui prend beaucoup de temps.
- Le traitement automatique de langages narurels (TALN) reste un défit
- Les données d'entraînement sont souvent biaisées

### I-5 Outils de l'apprentissage automatique

#### I-5-1 frameworks et bibliothèques logicielles

##### Deep Learning

Les outils suivants sont conçus pour l'apprentissage approfondu qui est basé le réseau de neurones.
- Outil: nom et lien de l'outil (ordre alphabétique)
- Licence: la licence de l'outil. Ici, on ne s'interresse que par les outils open sources.
- écrit en: le langage de programmation utilisé pour écrire cet outil.
- interfaces: les langages de programmation qu'on puisse utiliser pour utiliser cet outil (API).

| Outil | Licence | écrit en | interfaces |
| :---: | :---: | :---: | :---: |
| [Caffe](http://caffe.berkeleyvision.org) | BSD | C++ | C++, MATLAB, Python |
| [Deeplearning4j](https://deeplearning4j.org) | Apache 2.0 | C++, Java | Java, Scala, Clojure, Python, Kotlin |
| [Keras](https://keras.io) | MIT | Python | Python, R |
| [Microsoft Cognitive Toolkit](https://www.microsoft.com/en-us/cognitive-toolkit/) | MIT | C++ | Python, C++ |
| [MXNet (Apache)](https://mxnet.apache.org) | Apache 2.0 | C++ | C++, Clojure, Java, Julia, Perl, Python, R, Scala |
| [TensorFlow](https://www.tensorflow.org) | Apache 2.0 | C++, Python | Python (Keras), C/C++, Java, Go, JavaScript, R, Julia, Swift |
| [Theano](http://deeplearning.net/software/theano/) | BSD | Python | Python |
| [Torch](http://torch.ch) | BSD | C, Lua | C, Lua, LuaJIT |

Pour une comparison plus détaillée, veuiller consulter [cette page en Wikipédia](https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software)

##### Générique

La liste suivante contient les outils avec plusieurs algorithmes d'apprentissage automatique.

| Outil | Licence | écrit en | interfaces |
| :---: | :---: | :---: | :---: |
| [Data Analytics Acceleration Library(Intel)](http://software.intel.com/intel-daal) | Apache 2.0 | C++, Python, Java | C++, Python, Java |
| [MLLib(Apache Spark)](http://spark.apache.org/mllib/) | Apache 2.0 | - | Java, R, Python, Scala |
| [ScalaNLP Breeze](https://github.com/scalanlp/breeze) | Apache 2.0 | Scala | Scala |
| [Scikit-learn](https://scikit-learn.org/stable/) | BSD | Python | Python |
| [Shogun](http://www.shogun-toolbox.org) | - | C++ | C++, C#, Java, Lua, Octave, Python, R, Ruby |


#### I-5-2 Apprentissage automatique comme un service

Apprentissage automatique comme un service (MLaaS: Machine Learning as a Service):
- [Amazon Machine Learning](https://aws.amazon.com/aml/)
- [BigML](https://bigml.com)
- [DataRobot](https://www.datarobot.com)
- [Deepai enterprise machine learning](https://deepai.org/enterprise-machine-learning)
- [Deepcognition](https://deepcognition.ai)
- [FloydHub](https://www.floydhub.com)
- [IBM Watson Machine Learning](https://www.ibm.com/cloud/machine-learning)
- [Google Cloud Machine Learning Engine](https://cloud.google.com/ml-engine/)
- [Microsoft Azure Machine Learning studio](https://azure.microsoft.com/fr-fr/services/machine-learning-studio/)
- [MLJAR](https://mljar.com)
- [Open ML](https://www.openml.org)
- [ParallelDots](https://www.paralleldots.com)
- [VALOHAI](https://valohai.com)
- [Vize](https://vize.ai): Traitement d'images

#### I-5-3 Les ressources

##### Repertoires de données

- [Kaggle](https://www.kaggle.com): télécarger les données, faire des compétitions avec des prix.
- [Registry of Open Data on AWS](https://registry.opendata.aws)
- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Visual data](https://www.visualdata.io): Des données sur le traitement d'images.

##### Images

- [COCO](http://cocodataset.org/#home) un ensemble de données de détection, de segmentation et d'annotation des images.
- [COIL-100](http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php) des images de 100 objets différents prises sous tous les angles dans une rotation de 360.
- [ImageNet](http://image-net.org) des images organisées selon la hiérarchie de  [WordNet](http://wordnet.princeton.edu/)
- [Indoor Scene Recognition](http://web.mit.edu/torralba/www/indoor.html) reconnaissance de scènes d'intérieur
- [Labelled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/): reconnaissance faciale sans contraintes
- [LabelMe](http://labelme.csail.mit.edu/Release3.0/browserTools/php/dataset.php) Images annotées
- [LSUN](http://lsun.cs.princeton.edu/2016/) des images concernant un défi pour la classification de la scène.
- [Open Images Dataset de Google](https://ai.googleblog.com/2016/09/introducing-open-images-dataset.html)
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) la reconnaissance des races de chiens.
- [VisualGenome](http://visualgenome.org) une base pour connecter les images au langage

##### Analyse des sentiments

- [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) critiques de films
-[Multi-Domain Sentiment Dataset](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/) commentaires sur les produits d'Amazon
- [Sentiment140](http://help.sentiment140.com/for-students/) des Tweets avec des émoticônes filtrées.
- [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) critiques de films

##### Traitement du langage naturel

- [Amazon Reviews](https://snap.stanford.edu/data/web-Amazon.html)
- [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/)
- [Google Books Ngrams](https://aws.amazon.com/datasets/google-books-ngrams/)
- [Gutenberg eBooks List](http://www.gutenberg.org/wiki/Gutenberg:Offline_Catalogs) Liste annotée de livres électroniques du projet Gutenberg.
- [Hansards text chunks of Canadian Parliament](https://www.isi.edu/natural-language/download/hansard/) texte aligné: Français-Anglais
- [HotspotQA Dataset](https://hotpotqa.github.io) Question-Réponse (réponse automatique)
- [Jeopardy](https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/) Les questions de Jeopardy format Json
- [SMS Spam Collection in English](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)
- [UCI's Spambase](https://archive.ics.uci.edu/ml/datasets/Spambase) filtrage du courrier électronique indésirable
- [Wikipedia Links data](https://code.google.com/archive/p/wiki-links/downloads)
- [Yelp Reviews](https://www.yelp.com/dataset)

##### Auto-conduite

- [Baidu Apolloscapes](http://apolloscape.auto/)
- [Berkeley DeepDrive BDD100k](http://bdd-data.berkeley.edu/)
- [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/node/6132)
- [Cityscapes dataset](https://www.cityscapes-dataset.com/)
- [Comma.ai](https://archive.org/details/comma-dataset)
- [CSSAD Dataset](http://aplicaciones.cimat.mx/Personal/jbhayet/ccsad-dataset)
- [KUL Belgium Traffic Sign Dataset](http://www.vision.ee.ethz.ch/~timofter/traffic_signs/)
- [LaRa Traffic Light Recognition](http://www.lara.prd.fr/benchmarks/trafficlightsrecognition)
- [LISA datasets](http://cvrr.ucsd.edu/LISA/datasets.html)
- [MIT AGE Lab](http://lexfridman.com/automated-synchronization-of-driving-data-video-audio-telemetry-accelerometer/)
- [Oxford's Robotic Car](http://robotcar-dataset.robots.ox.ac.uk/)
- [WPI datasets](http://computing.wpi.edu/dataset.html)


### ANNEXE: Méthodologies de science des données

#### CRISP-DM (Cross-industry standard process for data mining) [Standard ouvert]

| ![CRISP-DM](https://upload.wikimedia.org/wikipedia/commons/b/b9/CRISP-DM_Process_Diagram.png) |
|:--:|
| *CRISP-DM [ [Wikimedia](https://commons.wikimedia.org/wiki/File:CRISP-DM_Process_Diagram.png) ]* |

#### ASUM-DM (Analytics Solutions Unified Method for Data Mining) [IBM]

| ![ASUM-DM](IMG/ASUM-DM.png) |
|:--:|
| *ASUM-DM [ [Source](ftp://ftp.software.ibm.com/software/data/sw-library/services/ASUM.pdf) ]* |

#### TDSP (Team Data Science Process) [Microsoft]

| ![TDSP](https://docs.microsoft.com/fr-fr/azure/machine-learning/team-data-science-process/media/overview/tdsp-lifecycle2.png) |
|:--:|
| *TDSP [ [Source](https://docs.microsoft.com/fr-fr/azure/machine-learning/team-data-science-process/overview) ]* |

### Bibliographie

- https://www.kdnuggets.com/2017/11/3-different-types-machine-learning.html
- https://www.techleer.com/articles/203-machine-learning-algorithm-backbone-of-emerging-technologies/
- https://www.wired.com/story/greedy-brittle-opaque-and-shallow-the-downsides-to-deep-learning/
- https://data-flair.training/blogs/advantages-and-disadvantages-of-machine-learning/
- https://towardsdatascience.com/coding-deep-learning-for-beginners-types-of-machine-learning-b9e651e1ed9d
- https://towardsdatascience.com/selecting-the-best-machine-learning-algorithm-for-your-regression-problem-20c330bad4ef
- https://docs.microsoft.com/fr-fr/azure/machine-learning/team-data-science-process/overview
- https://www.ibm.com/support/knowledgecenter/en/SSEPGG_9.5.0/com.ibm.im.easy.doc/c_dm_process.html
- ftp://public.dhe.ibm.com/software/analytics/spss/documentation/modeler/18.0/en/ModelerCRISPDM.pdf
- https://medium.com/datadriveninvestor/the-50-best-public-datasets-for-machine-learning-d80e9f030279

## Chapitre II: Préparation des données

**_...TODO: Complete one day!!_**

### II-1 Collection des données

#### Qualité des données

Critères d'intégrité des données:
- Taille: Nombre des échantillons (enregistrements). Certaines tâches nécessitent une grande taille de données pour qu'elles soient appris.
- Le nombre et le type de caractéristiques (nominales, binaires, ordinales ou continues).
- Le nombre des erreurs d'annotation
- La quantité de bruits dans les données: erreurs et exceptions

#### Intégration des données



#### Annotation des données


### II-2 Nétoyage des données

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

### II-3 Transformation des données

#### II-3-1 Discrétisation en utilisant le groupement (binning, bucketing)

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

#### II-3-2 Normalisation

##### Mise à l'échelle min-max

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

##### Coupure

S'il existe des valeurs aberrantes dans les extrémités d'une caractéristique, on applique une coupure max avec une valeur α et/ou min avec une valeur β.

![II-3-coupure]

Par exemple, dans le graphe suivant, qui illustre le nombre de cambres par personnes, on remarque qu'au delà de 4 les valeurs sont très baisses. La solution est d'appliquer une coupure max de 4.

| ![coupure](https://developers.google.com/machine-learning/data-prep/images/norm-clipping-outliers.svg) |
|:--:|
| *Nombre de chambres par personne: avant et après la coupure avec max de 4 personnes [ [Source](https://developers.google.com/machine-learning/data-prep/transform/normalization) ]* |

##### Mise à l'échelle log

Cette transformation est utile lorsque un petit ensemble de valeurs ont plusieurs points, or la plupart des valeurs ont moins de points. Elle sert à compresser la range des valeurs.

![II-3-log]

Par exemple, les évaluations par film.
Dans le schéma suivant, la plupart des films ont moins d'évaluations.

| ![log](https://developers.google.com/machine-learning/data-prep/images/norm-log-scaling-movie-ratings.svg) |
|:--:|
| *Normalisation log des évaluation des films [ [Source](https://developers.google.com/machine-learning/data-prep/transform/normalization) ]* |

##### Z-score

#### II-3-3 Binarisation

Il existe des cas où on n'a pas besoin des fréquences (nombre d'occurences) d'une caractéristique pour créer un modèle; on a besoin seulement de savoir si cette caractéristique a apparue une fois au moins pour un échantillon. Dans ce cas, on binarise les valeurs de cette caractéristique.

![II-3-bin]

Par exemple, si on veut construire un système de recommandation de chansons, on va simplement avoir besoin de savoir si une personne est intéressée ou a écouté une chanson en particulier.
Cela n'exige pas le nombre de fois qu'une chanson a été écoutée mais, plutôt, les différentes chansons que cette personne a écoutées.


[II-3-min-max]: https://latex.codecogs.com/png.latex?x'=\frac{x-x_{min}}{x_{max}-x_{min}}
[II-3-coupure]: https://latex.codecogs.com/png.latex?x'=\begin{cases}\alpha&si\;x\ge\alpha\\\\\beta&si\;x\le\beta\\\\x&sinon\end{cases}
[II-3-log]: https://latex.codecogs.com/png.latex?x'=\log(x)
[II-3-bin]: https://latex.codecogs.com/png.latex?x'=\begin{cases}1&si\;x\ge1\\\\0&sinon\end{cases}

### II-4 Réduction des données

#### II-4-1 Données imbalancées

#### II-4-2 Partitionnement des données

#### II-4-3 Randomisation

### II-5 Outils de préparation des données

| Outil | Licence | Langage |
| :---: | :---: | :---: |
| [pandas](https://pandas.pydata.org) | BSD | Python |
| [scikit-learn](https://scikit-learn.org/stable/) | BSD | Python |

### II-6 Un peu de programmation

#### II-6-1 Discrétisation



### Bibliographie

- https://developers.google.com/machine-learning/data-prep/
- https://developers.google.com/machine-learning/crash-course/representation/video-lecture
- https://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/
- https://www.altexsoft.com/blog/datascience/preparing-your-dataset-for-machine-learning-8-basic-techniques-that-make-your-data-better/
- https://www.analyticsindiamag.com/get-started-preparing-data-machine-learning/
- https://docs.microsoft.com/fr-fr/azure/machine-learning/team-data-science-process/prepare-data
- https://www.simplilearn.com/data-preprocessing-tutorial

[vec-f]: https://latex.codecogs.com/png.latex?\overrightarrow{f}
[c-i]: https://latex.codecogs.com/png.latex?c_i
[f-j]: https://latex.codecogs.com/png.latex?f_j
[vec-C]: https://latex.codecogs.com/png.latex?\overrightarrow{C}

## Chapitre III: Classification naïve bayésienne

La classification est un apprentissage supervisé; ce qui veut dire, on doit entraîner notre système sur un ensemble de données, ensuite on utilise ce modèle pour classer des données de test.

Ici, on va commencer par présenter la phase de classification avant la phase d'entraînement.

### III-1 Classification

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
Pour remédier à ce problème, on peut utiliser une fonction de lissage comme le lissage de Lidstone.

![III-2-mult3]

Où: |![vec-f]| est le nombre des caractéristiques.
Alpha: est un nombre dans l'intervalle ]0, 1]. Lorsque sa valeur égale à 1, on appelle ça le laissage de Laplace.

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

#### Classifieur naïf bayésien suivant la loi normal

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

### III-4 Avantages

Les classifieurs naïfs bayésiens, malgré leurs simplicité, ont des points forts:

- Ils ont besoin d'une petite quantité de données d’entraînement.
- Ils sont très rapides par rapport aux autres classifieurs.
- Ils donnent de bonnes résultats dans le cas de filtrage du courrier indésirable et de classification de documents.

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
- https://mattshomepage.com/articles/2016/Jun/07/bernoulli_nb/

## Chapitre IV: Machine à vecteurs de support



## Chapitre V: Arbre de décision

## Chapitre VI: Régression linéaire

## Chapitre VII: Régression logistique

## Chapitre VIII: Perceptron


### Bibliographie
- https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975

## Chapitre IX: Réseau de neurones artificiels

## Chapitre X: Regroupement

### X-1 Regroupement hiérarchique

### X-2 K-Means

### Bibliographie

- https://towardsdatascience.com/unsupervised-learning-with-python-173c51dc7f03

## Chapitre XI: Auto-encodeur (Maybe not!!)

## Chapitre XII: Apprentissage par renforcement
