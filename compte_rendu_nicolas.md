# An empirical study of example forgetting during deep neural network learning
## Authors
* Mariya Tonevay, Carnegie Mellon University
* Alessandro Sordoni, Microsoft Research Montreal
* Remi Tachet des Combes, Microsoft Research Montreal
* Adam Trischler, Microsoft Research Montreal
* Yoshua Bengio, MILA Université de Montréal, CIFAR Senior Fellow
* Geoffrey J. Gordon, Microsoft Research Montreal, Carnegie Mellon University


## Useful links
* https://en.wikipedia.org/wiki/Catastrophic_interference


## Definitions
* Catastrophic forgetting (catastrophic interference) : tendency of an artificial neural network to completely and abruptly forget previously learned information upon learning new information.
* Forgetting event : when an individual training example transitions from being classified correctly to incorrectly over the course of learning.
* Continual learning (CL) : The ability of a model to learn continually from a stream of data, building on what was learnt previously, hence exhibiting positive transfer, as well as being able to remember previously seen tasks
* Unforgettable sample : Un exemple qui n'est jamais oublié entre chaque apprentissage.


# 1 - Introduction
On sait pas vraiment ce que cause le Catastrophic Forgetting. Une hypothèse est le changement de distribution dans les données de différentes tâches. Les features manquent alors de facteurs en communs et les méthodes d'optimisation ont alors du mal à converger vers une solution radicalement différente.

Lors de la descente de gradient, chaque mini-batch peut être considéré comme une mini-tâche.
Au départ, ils cherchent à savoir s'il existe des exemples qui sont constament oubliés ou des exemples qui ne le sont jamais entre chaque apprentissage.
Ils font l'hypothèse que des exemples qui sont souvent oubliés ne sont pas similaires à d'autres exemples. Ils vont donc étudier la proportion des exemples souvent oubliés, ainsi que leur effet sur la frontière de décision du modèle.

Ils font ça en deux étapes :
* Analyser les interactions entre les exemples durant l'apprentissage et sur la frontière de décision. Ils s'intéressent particulièrement à la capacité de compression du dataset.
* Voir si les résultats obtenus permettent de caractériser des exemples en tant qu'importants ou qu'outlier.


# 2 - Related Work
TODO


# 3 - Defining and computing example forgetting
On est dans un processus de classification standard, avec minimisation du risque empirique.
Loss : Cross-entropy
Optimizer : SGD (ou variante)

TODO : Surement mettre ces formules dans le rapport dans le cadre de la définition du processus d'oubli.

Un exemple est considéré comme oublié s'il passe de bien classifié à mal classifié à l'itération d'après.
Inversement, un exemple est considéré comme appris s'il passe de mal classifié à bien classifié à l'itération d'après.



# Conclusion
(i) certain examples are forgotten with high frequency, and some not at all
(ii) a data set’s (un)forgettable examples generalize across neural architectures;
(iii) based on forgetting dynamics, a significant fraction of examples can be omitted from the training data set while still maintaining state-of-the-art generalization performance.
a) Il y'a souvent un grand nombre d'exemples inoubliables
b) Des exemples bruités ou des images visuellement complexes à classifier font partie des exemples les plus oubliés.
c) Retirer un grand nombre d'exmeples un peu oubliés donne toujours de très bons résultats en test.

