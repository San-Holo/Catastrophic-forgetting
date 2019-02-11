import random
import os
import re
import pickle
from glob import glob
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def plot_with_deleted_samples(database="mnist", path_dir_files="."):
    best_acc_random = dict()
    best_acc_ordered = dict()
    best_acc = None
    for file in glob(os.path.join(path_dir_files, "*best_acc.txt")):
        file = os.path.basename(file)
        res = re.split((r"dataset_(\w+)__"
                        r"no_dropout_(\w+)__"
                        r"seed_(\d+)__"
                        r"sorting_file_(\w+)__"
                        r"remove_n_(\d+)__"
                        r"keep_lowest_n_(-?\d+)__"
                        r"best_acc.txt"), file)[1:-1]
        _, _, _, sorting_file, remove_n, keep_lowest_n = res
        remove_n = int(remove_n)
        keep_lowest_n = int(keep_lowest_n)

        with open(file) as f:
            best_acc_test = float(f.readlines()[-1].split()[-1])

        if sorting_file == "none":
            best_acc = best_acc_test
        elif (sorting_file == "mnist_sorted") and (keep_lowest_n == 0):
            best_acc_ordered[remove_n] = best_acc_test
        elif (sorting_file == "mnist_sorted") and (keep_lowest_n == -1):
            best_acc_random[remove_n] = best_acc_test

    # Compute list by sorted dict keys
    best_acc_ordered_lst = [best_acc_ordered[k]
                            for k in sorted(best_acc_ordered)]
    best_acc_random_lst = [best_acc_random[k] for k in sorted(best_acc_random)]
    best_acc_lst = [best_acc for _ in range(len(best_acc_random))]
    x = list(range(5000, 50001, 5000))

    print(best_acc_lst)
    print(best_acc_ordered_lst)
    print(best_acc_random_lst)

    axes = plt.gca()
    axes.grid(alpha=0.3)
    # plt.title(u"Accuracy en test sur MNIST en fonction du nombre de\nsuppressions d'exemples effectuées")
    plt.xlabel("Nombre de suppressions")
    plt.ylabel("Accuracy")
    plt.plot(x, best_acc_lst, "r-", label="Sans suppression")
    plt.plot(x, best_acc_ordered_lst, "b-", label="Suppressions ordonnées")
    plt.plot(x, best_acc_random_lst, "y-", label="Suppressions aléatoires")
    plt.legend(loc='lower left')
    plt.xticks(range(5000, 50001, 5000))
    plt.show()


def plot_hist_with_noisy_samples(path_file_raw, path_file_noisy):
    with open(path_file_raw, "rb") as f:
        truc = pickle.load(f)
        forgetting_counts_raw, _ = truc["forgetting counts"], truc["indices"]

    with open(path_file_noisy, "rb") as f:
        truc = pickle.load(f)
        forgetting_counts_noisy, _ = truc["forgetting counts"], truc["indices"]

    forgetting_counts_raw = Counter(forgetting_counts_raw)
    del forgetting_counts_raw[200]

    forgetting_counts_noisy = Counter(forgetting_counts_noisy)
    del forgetting_counts_noisy[200]

    # Plotting
    axes = plt.gca()
    axes.grid(alpha=0.3)
    plt.xlabel("Nombre d'évènements d'oublis")
    plt.ylabel("Pourcentage d'exemples de chaque catégorie")
    plt.bar(*zip(*sorted(forgetting_counts_raw.items())),
            alpha=0.65, label="Exemples normaux")
    plt.bar(*zip(*sorted(forgetting_counts_noisy.items())),
            alpha=0.65, label="Exemples bruités")
    plt.legend(loc='upper right')
    plt.xlim(0, 25)
    plt.ylim(0.000, 0.200)

    plt.show()


if __name__ == '__main__':
    plot_with_deleted_samples()
    # plot_hist_with_noisy_samples(path_file_raw="../../../example_forgetting/cifar10_results/cifar10_raw.pkl",
    #                              path_file_noisy="../../../example_forgetting/cifar10_results/cifar10_noisy.pkl")
