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
* Voir si les résultats obtenus permettent de caractériser des exemples en tant qu'exemples importants ou que outliers.


# 2 - Related Work
TODO


# 3 - Defining and computing example forgetting
On est dans un processus de classification standard, avec minimisation du risque empirique.
Loss : Cross-entropy
Optimizer : SGD Mini-Batch avec momentum

TODO : Surement mettre ces formules dans le rapport dans le cadre de la définition du processus d'oubli.

Un exemple est considéré comme oublié s'il passe de bien classifié à mal classifié à l'itération d'après.
Inversement, un exemple est considéré comme appris s'il passe de mal classifié à bien classifié à l'itération d'après.
Un exemple est considéré comme inoubliable s'il n'est jamais oublié durant l'apprentissage.
La marge de classification c'est la différence entre le logit de la bonne classe et le plus haut logit parmi les autres classes.

## 3.1 - Procedural description and experimental setting
Normalement on devrait calculer la prédiction du modèle sur chaque exemple du dataset à chaque mise à jour des poids du modèle, mais trop coûteux. À la place on le fait sur chaque exemple du mini-batch. Ça donne une borne inf du nombre de changements que peut subir un exemple.
Après apprentissage, on va trier les exemples du dataset en fonction du nombre de fois où ils ont été oublié.

Evaluation sur 3 datasets :
* MNIST
* Permuted MNIST (MNIST sur lequel on applique la même permutation de pixels sur chaque exemple).
* CIFAR-10

Modèles :
* Pour MNIST : un CNN avec 2 couches de conv + 1 couche FC. 0.8% d'erreur en test
* Pour CIFAR : Resnet avec cutout (on masque une partie de l'image afin de mettre l'accent sur de petites features). 3.99% d'erreur en test.


# 4 - Characterizing example forgetting
Pour les 3 datasets, il y a 55,012, 45,181 et 15,628 exemples inoubliables en commun sur les 5 seeds, ce qui représente 91.7%, 75.3%, et 31.3% du dataset.
Les datasets avec moins de complexité et de diversité dans les exemples (comme MNIST) semblent avoir + d'exemples inoubliables.


# 5 - Removing unforgettable examples
We observe that when removing a random subset of the dataset, performance rapidly decreases. Comparatively, by removing examples ordered by number of forgetting events, 30% of the dataset can be removed while maintaining comparable generalization performance as the base model trained on the full dataset, and up to 35% can be removed with marginal degradation (less than 0.2%).


# Conclusion
(i) certain examples are forgotten with high frequency, and some not at all
(ii) a data set’s (un)forgettable examples generalize across neural architectures;
(iii) based on forgetting dynamics, a significant fraction of examples can be omitted from the training data set while still maintaining state-of-the-art generalization performance.
a) Il y'a souvent un grand nombre d'exemples inoubliables
b) Des exemples bruités ou des images visuellement complexes à classifier font partie des exemples les plus oubliés.
c) Retirer un grand nombre d'exmeples un peu oubliés donne toujours de très bons résultats en test.

