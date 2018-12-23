# Chapitre I: Introduction

## Sommaire

[(Retour vers la page principale)](README.md)

- Chapitre I: Introduction
  - [I-1 Motivation](#i-1-motivation)
  - [I-2 Applications](#i-2-applications)
  - [I-3 Types des algorithmes d'apprentissage](#i-3-types-des-algorithmes-dapprentissage)
  - [I-4 Limites de l'apprentissage automatique](#i-4-limites-de-lapprentissage-automatique)
  - [I-5 Outils de l'apprentissage automatique](#i-5-outils-de-lapprentissage-automatique)
  - [I-6 Méthodologies de science des données](#i-6-méthodologies-de-science-des-données)


| ![apprentissage automatique](IMG/AA.png) |
|:--:|
| *Apprentissage automatique* |

## I-1 Motivation

- Certaines tâches sont difficiles à programmer manuellement: Reconnaissance de formes, Traduction par machine, Reconnaissance de la parole, Aide à la décision, etc.
- Les données sont disponibles, qui peuvent être utilisé pour estimer la fonction de notre tâche.

[(Sommaire)](#sommaire)

## I-2 Applications

- Santé:
  - Watson santé de IBM: https://www.ibm.com/watson/health/
  - Projet Hanover de Microsoft: https://hanover.azurewebsites.net
  - DeepMind santé de Google: https://deepmind.com/applied/deepmind-health/

- Finance : Prévention de fraude, management de risques, prédiction des investissements, etc.
- Domaine légal : cas de CaseText https://casetext.com
- Traduction: Google traslate https://translate.google.com/
**_...TODO: Add more_**

[(Sommaire)](#sommaire)

## I-3 Types des algorithmes d'apprentissage

### I-3-1 Apprentissage supervisé

Lorsque nous disposons d'un ensemble de données avec les résulats attendus, on peut entraîner un système sur ces données pour inférer la fonction utilisée pour avoir ces résulats.
En résumé:

- **Source d'apprentissage:** des données annotées (nous avons les résultats attendus)
- **Retour d'information:** direct; à partir des résulats attendues.
- **Fonction:** prédire les future résultats

| ![apprentissage supervisé](IMG/AA-supervise.svg)|
|:--:|
| *Apprentissage supervisé* |

Selon le type d'annotation, on peut remarquer deux types des algorithmes d'apprentissage automatique: classement et régression.

#### Classement (Classification supervisée)

Lorsque le résulat attendu est une classe (groupe).

| Par exemple: |
| :--: |
| Classer un animal comme: chat, chien, vache ou autre en se basant sur le poids, la longueur et le type de nourriture.  |

TODO: recall, precision, evaluation methods 

#### Régression

Lorsque le résulat attendu est une valeur.

| Par exemple: |
| :--: |
| Estimer le prix d'une maison à partir de sa surface, nombre de chambre et l'emplacement. |


### I-3-2 Apprentissage non supervisé

Lorsque nous disposons d'un ensemble de données non annotées (sans savoir les résulats attendus).
En résumé:

- **Source d'apprentissage:** des données non annotées
- **Retour d'information:** pas de retour; on dispose seulement des données en entrée.
- **Fonction:** rechercher les structures cachées dans les données.

Selon le type de structure que l'algorithme va découvrir, on peut avoir: le regroupement et la réduction de dimension.

#### Clustering (Regroupement)

L'algorithme de regroupement sert à assigner les échantillons similaires dans le même groupe.
Donc, le résulat est un ensemble de groupes contenants les échantillons.

| Par exemple: |
| :--: |
| Regrouper les plantes similaires en se basant sur la couleur, la taile, etc.  |

#### Réduction de dimension

L'algorithme de réduction de dimension a comme but d'apprendre comment représenter des données en entrée avec moins de valeurs.

| Par exemple: |
| :--: |
| Représenter des individus sur un graph de deux dimensions en utilisant la taille, le poids, l'age, la couleur des cheveux, la texture des cheveux et la couleur des yeux  |


### I-3-3 Apprentissage par renforcement

- **Source d'apprentissage:** le processus de décision
- **Retour d'information:** un système de récompense
- **Fonction:** recherche des structures cachées dans les données.

| ![apprentissage par renforcement](IMG/RL-fr.png) |
|:--:|
| *Apprentissage par renforcement [ [Wikimedia](https://commons.wikimedia.org/wiki/File:Reinforcement_learning_diagram_fr.svg?uselang=fr) ]* |


[(Sommaire)](#sommaire)

## I-4 Limites de l'apprentissage automatique

- Pour des tâches complexes, on a besoin d'une grande quantité de données
- Dans le cas de l'apprentissage supervisé, l'annotation de données est une tâche fastidieuse; qui prend beaucoup de temps.
- Le traitement automatique de langages narurels (TALN) reste un défit
- Les données d'entraînement sont souvent biaisées

[(Sommaire)](#sommaire)

## I-5 Outils de l'apprentissage automatique

### I-5-1 frameworks et bibliothèques logicielles

#### Deep Learning

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

#### Générique

La liste suivante contient les outils avec plusieurs algorithmes d'apprentissage automatique.

| Outil | Licence | écrit en | interfaces |
| :---: | :---: | :---: | :---: |
| [Data Analytics Acceleration Library(Intel)](http://software.intel.com/intel-daal) | Apache 2.0 | C++, Python, Java | C++, Python, Java |
| [MLLib(Apache Spark)](http://spark.apache.org/mllib/) | Apache 2.0 | - | Java, R, Python, Scala |
| [ScalaNLP Breeze](https://github.com/scalanlp/breeze) | Apache 2.0 | Scala | Scala |
| [Scikit-learn](https://scikit-learn.org/stable/) | BSD | Python | Python |
| [Shogun](http://www.shogun-toolbox.org) | - | C++ | C++, C#, Java, Lua, Octave, Python, R, Ruby |


### I-5-2 Apprentissage automatique comme un service

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

### I-5-3 Les ressources

#### Repertoires de données

- [Kaggle](https://www.kaggle.com): télécarger les données, faire des compétitions avec des prix.
- [Registry of Open Data on AWS](https://registry.opendata.aws)
- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Visual data](https://www.visualdata.io): Des données sur le traitement d'images.

#### Images

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

#### Analyse des sentiments

- [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) critiques de films
-[Multi-Domain Sentiment Dataset](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/) commentaires sur les produits d'Amazon
- [Sentiment140](http://help.sentiment140.com/for-students/) des Tweets avec des émoticônes filtrées.
- [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) critiques de films

#### Traitement du langage naturel

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

#### Auto-conduite

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

[(Sommaire)](#sommaire)

## I-6 Méthodologies de science des données

### CRISP-DM (Cross-industry standard process for data mining) [Standard ouvert]

| ![CRISP-DM](https://upload.wikimedia.org/wikipedia/commons/b/b9/CRISP-DM_Process_Diagram.png) |
|:--:|
| *CRISP-DM [ [Wikimedia](https://commons.wikimedia.org/wiki/File:CRISP-DM_Process_Diagram.png) ]* |

### ASUM-DM (Analytics Solutions Unified Method for Data Mining) [IBM]

| ![ASUM-DM](IMG/ASUM-DM.png) |
|:--:|
| *ASUM-DM [ [Source](ftp://ftp.software.ibm.com/software/data/sw-library/services/ASUM.pdf) ]* |

### TDSP (Team Data Science Process) [Microsoft]

| ![TDSP](https://docs.microsoft.com/fr-fr/azure/machine-learning/team-data-science-process/media/overview/tdsp-lifecycle2.png) |
|:--:|
| *TDSP [ [Source](https://docs.microsoft.com/fr-fr/azure/machine-learning/team-data-science-process/overview) ]* |

[(Sommaire)](#sommaire)

## Bibliographie

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
