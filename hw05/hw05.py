import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.spatial as spa

# IMPORTING DATA #
data_set = np.genfromtxt("hw05_data_set.csv", delimiter=",", skip_header=1)
means = np.genfromtxt("hw05_initial_centroids.csv", delimiter=",")
K = np.shape(means)[0] # number of clusters
D = np.shape(means)[1] # number of variables
N = np.shape(data_set)[0] # number  of data points
iteration = 100
print("Number of data points:",N)
print("Number of clusters:",K)
print("Number of variables:",D)

# CALCULATIONS #
# calculate distances between centroids and data points
distance = spa.distance_matrix(data_set, means)
# find the nearest centroid for each data point
membership = np.argmin(distance, axis=1)

# INITIAL COVARIANCE MATRIX #
covariances = np.zeros((5,2,2))
for j in range(K):
    covariances[j] = np.dot(np.transpose(data_set[membership==j]-means[j]), (data_set[membership==j]-means[j]))/len(data_set[membership==j])

# INITIAL PRIOR PROBABILITIES #
priors = np.zeros((5))
for k in range(K):
    priors[k] = len(data_set[membership==k])/len(membership)

# EXPECTATION STEP #
def e_step(X, mean, cov, p):
    score = np.zeros((N, K))

    for j in range(K):
        score[:, j] = [st.multivariate_normal.pdf(d, mean[j], cov[j]) * p[j] for d in X]
    for i in range(N):
        score[i] /= np.sum(score[i])
    return score

# MAXIMIZATION STEP #
def m_step_priors(score):
    return score.sum(0)/np.array([N]*K)

def m_step_means(X,score):
    score_sum = score.sum(0)
    temp_mean = np.matmul(np.transpose(score),X)
    for c in range(K):
        temp_mean[c] = temp_mean[c]/score_sum[c]
    return temp_mean


def m_step_covs(X, mean, cov, score):
    inner = np.zeros((D, N))
    for k in range(K):
        for i in range(D):
            inner[i] = np.transpose(X - mean[k])[i] * score[:, k]
        cov[k] = np.dot(inner, (X - mean[k]))
        cov[k] /= np.sum(score[:, k])

    return cov

# IMPLEMENTATION #
for iter in range(iteration):
    print("iteration = ", iter + 1)
    score = e_step(data_set, means, covariances, priors)
    priors = m_step_priors(score)
    means = m_step_means(data_set, score)
    covariances = m_step_covs(data_set, means, covariances, score)

print("Mean Matrix\n", means)

# PLOTTING #
true_means = [[2.5, 2.5], [-2.5, 2.5], [-2.5, -2.5], [2.5, -2.5], [0.0, 0.0]]
true_covs = [[[0.8, -0.6], [-0.6, 0.8]], [[0.8, 0.6], [0.6, 0.8]], [[0.8, -0.6], [-0.6, 0.8]], [[0.8, 0.6], [0.6, 0.8]],
             [[1.6, 0.0], [0.0, 1.6]]]

# multivariate random variables from EM algorithm
em1 = st.multivariate_normal(means[0], covariances[0])
em2 = st.multivariate_normal(means[1], covariances[1])
em3 = st.multivariate_normal(means[2], covariances[2])
em4 = st.multivariate_normal(means[3], covariances[3])
em5 = st.multivariate_normal(means[4], covariances[4])

# original multivariate random variables
or1 = st.multivariate_normal(true_means[0], true_covs[0])
or2 = st.multivariate_normal(true_means[1], true_covs[1])
or3 = st.multivariate_normal(true_means[2], true_covs[2])
or4 = st.multivariate_normal(true_means[3], true_covs[3])
or5 = st.multivariate_normal(true_means[4], true_covs[4])

x = np.arange(-20.0, 20.0, 0.1)
y = np.arange(-20.0, 20.0, 0.1)
X, Y = np.meshgrid(x, y)
loc = np.dstack((X, Y))

colors = np.array(["#1f78b4", "#e31a1c", "#33a02c", "#6a3d9a", "#ff7f00"])

plt.figure(figsize=(8, 8))

for n in range(N):
    cluster = np.argmax(score[n])

    if (cluster == 0):
        plt.plot(data_set[n, 0], data_set[n, 1], ".", markersize=10, c=colors[0])
    elif (cluster == 1):
        plt.plot(data_set[n, 0], data_set[n, 1], ".", markersize=10, c=colors[1])
    elif (cluster == 2):
        plt.plot(data_set[n, 0], data_set[n, 1], ".", markersize=10, c=colors[2])
    elif (cluster == 3):
        plt.plot(data_set[n, 0], data_set[n, 1], ".", markersize=10, c=colors[3])
    else:
        plt.plot(data_set[n, 0], data_set[n, 1], ".", markersize=10, c=colors[4])

plt.contour(X, Y, em1.pdf(loc), [0.05], colors="k")
plt.contour(X, Y, em2.pdf(loc), [0.05], colors="k")
plt.contour(X, Y, em3.pdf(loc), [0.05], colors="k")
plt.contour(X, Y, em4.pdf(loc), [0.05], colors="k")
plt.contour(X, Y, em5.pdf(loc), [0.05], colors="k")

plt.contour(X, Y, or1.pdf(loc), [0.05], colors="k", linestyles="dashed")
plt.contour(X, Y, or2.pdf(loc), [0.05], colors="k", linestyles="dashed")
plt.contour(X, Y, or3.pdf(loc), [0.05], colors="k", linestyles="dashed")
plt.contour(X, Y, or4.pdf(loc), [0.05], colors="k", linestyles="dashed")
plt.contour(X, Y, or5.pdf(loc), [0.05], colors="k", linestyles="dashed")

plt.xlim(-6.5, 6.5)
plt.ylim(-6.5, 6.5)
plt.title("EM Clustering")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
