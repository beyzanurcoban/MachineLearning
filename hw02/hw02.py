# Homework 2: Discrimination by Regression
# Beyzanur Çoban 64763
# March 27, 2021

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# SIGMOID FUNCTION #
def sigmoid(X, w, w0):
    return 1 / (1 + np.exp(-(np.matmul(X, w) + w0)))


# GRADIENT FUNCTION #
def gradient_w(X, y_truth, y_predicted):
    return np.asarray([-np.sum(
        np.repeat(((y_truth[:, c] - y_predicted[:, c]) * y_predicted[:, c] * (1 - y_predicted[:, c]))[:, None],
                  X.shape[1], axis=1) * X, axis=0) for c in range(K)]).transpose()


def gradient_w0(y_truth, y_predicted):
    y_class = (y_truth - y_predicted) * y_predicted * (1 - y_predicted)
    return -np.sum(y_class)


# IMPORTING DATA #
# read data into memory, get training x and y values
X = np.genfromtxt("hw02_images.csv", delimiter=",", skip_footer=500)
y_truth = np.genfromtxt("hw02_labels.csv", skip_footer=500).astype(int)

# get test x and y values
X_test = np.genfromtxt("hw02_images.csv", delimiter=",", skip_header=500)
y_test = np.genfromtxt("hw02_labels.csv", skip_header=500).astype(int)

N = y_truth.shape[0]  # number of samples
K = np.max(y_truth)  # number of classes
P = X.shape[1]  # number of parameters

# one-of-K encoding for training data labels
Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth - 1] = 1

# one-of-K encoding for test data labels
Y_test = np.zeros((N, K)).astype(int)
Y_test[range(N), y_test - 1] = 1

# ALGORITHM PARAMETERS #
eta = 0.0001
epsilon = 1e-3
max_iteration = 500

w = np.genfromtxt("initial_W.csv", delimiter=",")
w0 = np.genfromtxt("initial_w0.csv", delimiter=",")

# ITERATIVE ALGORITHM #
objective_values = []
iteration = 1

while 1:
    Y_predicted = sigmoid(X, w, w0)
    Y_predicted_test = sigmoid(X_test, w, w0)

    objective_values = np.append(objective_values, np.sum(np.sum(0.5 * (Y_truth - Y_predicted) ** 2, axis=1), axis=0))

    w_old = w
    w0_old = w0

    w = w - eta * gradient_w(X, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)

    if np.sqrt(np.sum((w0 - w0_old)) ** 2 + np.sum((w - w_old) ** 2)) < epsilon:
        break

    if iteration >= max_iteration:
        break

    iteration = iteration + 1

print(w, w0, "\n")

# plot objective function during iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

# calculate confusion matrix for y_train
y_predicted = np.argmax(Y_predicted, axis=1) + 1
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames=['y_predicted'], colnames=['y_train'])
print(confusion_matrix, "\n")

# calculate confusion matrix for y_test
y_predicted_test = np.argmax(Y_predicted_test, axis=1) + 1
confusion_matrix = pd.crosstab(y_predicted_test, y_test, rownames=['y_predicted'], colnames=['y_test'])
print(confusion_matrix, "\n")
