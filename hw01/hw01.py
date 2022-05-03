# Homework 1: Naive Bayes' Classifier
# Beyzanur Çoban 64763
# March 19, 2021

import math
import numpy as np
import pandas as pd

# IMPORTING DATA #
# read data into memory, get training x and y values
x = np.genfromtxt("hw01_images.csv", delimiter=",", skip_footer=200)
y = np.genfromtxt("hw01_labels.csv", skip_footer=200).astype(int)

print(x)

# get test x and y values
x_test = np.genfromtxt("hw01_images.csv", delimiter=",", skip_header=200)
y_test = np.genfromtxt("hw01_labels.csv", skip_header=200).astype(int)

N = y.shape[0]  # number of samples
K = np.max(y)  # number of classes
P = x.shape[1]  # number of parameters

# PARAMETER ESTIMATION #
# mean parameters for both classes
y1_means = [np.mean(x[y == 1, j]) for j in range(P)]
y2_means = [np.mean(x[y == 2, j]) for j in range(P)]
means = np.stack((y1_means, y2_means), axis=-1)
print("Class 1 means:", means[:, 0])
print("Class 2 means:", means[:, 1], "\n")

# standard deviation parameters for both classes
y1_deviations = [np.sqrt(np.mean((x[y == 1, j] - y1_means[j]) ** 2)) for j in range(P)]
y2_deviations = [np.sqrt(np.mean((x[y == 2, j] - y2_means[j]) ** 2)) for j in range(P)]
deviations = np.stack((y1_deviations, y2_deviations), axis=-1)
print("Class 1 deviations:", deviations[:, 0])
print("Class 2 deviations:", deviations[:, 1], "\n")

# class priors
class_priors = [np.mean(y == (c + 1)) for c in range(K)]
print("Class priors:", class_priors, "\n")

# SCORE FUNCTIONS #
# evaluate score functions
class_scores_tr = np.empty([N, K])  # score functions for training data
class_scores_tst = np.empty([N, K])  # score functions for test data

for j in range(K):
    for i in range(N):
        # for training data
        class_scores_tr[i, j] = sum(- 0.5 * np.log(2 * math.pi * deviations[:, j] ** 2)
                                    - 0.5 * (x[i, :] - means[:, j]) ** 2 / deviations[:, j] ** 2)
        class_scores_tr[i, j] = class_scores_tr[i, j] + np.log(class_priors[j])

        # for test data
        class_scores_tst[i, j] = sum(- 0.5 * np.log(2 * math.pi * deviations[:, j] ** 2)
                                     - 0.5 * (x_test[i, :] - means[:, j]) ** 2 / deviations[:, j] ** 2)
        class_scores_tst[i, j] = class_scores_tst[i, j] + np.log(class_priors[j])

# PREDICTED CLASS LABELS #
# y_hat - training data
y_hat_tr = np.where(class_scores_tr[:, 0] > class_scores_tr[:, 1], 1, 2)

# y_hat - test data
y_hat_tst = np.where(class_scores_tst[:, 0] > class_scores_tst[:, 1], 1, 2)

# CONFUSION MATRICES #
# confusion matrix - training data
confusion_matrix_tr = pd.crosstab(y, y_hat_tr, rownames=['y_train'], colnames=['y_hat'])
print(confusion_matrix_tr, "\n")
# confusion matrix - test data
confusion_matrix_tst = pd.crosstab(y_test, y_hat_tst, rownames=['y_test'], colnames=['y_hat'])
print(confusion_matrix_tst)
