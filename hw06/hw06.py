# Beyzanur Coban 64763
# ENGR421 Hw06 - Spectral Clustering

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.spatial as spa

# IMPORTING DATA #
X = np.genfromtxt("hw06_data_set.csv", delimiter=",", skip_header=1)
D = np.shape(X)[1] # number of variables
N = np.shape(X)[0] # number  of data points
R = 5
K = 5
print("Number of data points:",N)
print("Number of features:",D)
print("Number of clusters:",K)

# CONNECTIVITY MATRIX #
B = np.zeros((N,N))
sigma = 1.25

for i in range(N):
    for j in range(N):
        if i!=j:
            B[i,j] = (spa.distance.euclidean(X[i],X[j]) <= sigma)

plt.figure(figsize=(8, 8))
for n in range(N):
    plt.plot(X[n, 0], X[n, 1], ".", markersize=10, c="k")

for i in range(N):
    for j in range(i, N):
        if B[i, j] == 1.:
            x1, x2 = X[i, 0], X[j, 0]
            y1, y2 = X[i, 1], X[j, 1]
            plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Connectivity Matrix Visualization")
plt.show()

# LAPLACIAN MATRIX #
D = np.zeros((N,N))
for i in range(N):
    D[i,i] = np.sum(B[i,:])

L = D-B

I = np.identity(N)
D_inv = np.linalg.inv(np.sqrt(D))
L_symmetric = I - np.dot(np.dot(D_inv, B), D_inv)

eigvals, eigvecs = np.linalg.eig(L_symmetric)
idx = np.argsort(eigvals)
eigvals = eigvals[idx]
eigvecs = eigvecs[:,idx]
Z = eigvecs[:,:5]

# INITIAL CENTROIDS #
initial_centroids = np.zeros((K,K))

initial_centroids[0] = Z[84]
initial_centroids[1] = Z[128]
initial_centroids[2] = Z[166]
initial_centroids[3] = Z[186]
initial_centroids[4] = Z[269]

# HELPER UPDATE FUNCTIONS #
def update_centroids(memberships, X, initcentro):
    if memberships is None:
        # initialize centroids
        centroids = initcentro
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k,], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)

# K-MEAN CLUSTERING ALGORITHM IMPLEMENTATION #
centroids = None
memberships = None
iteration = 1
while True:
    print("Iteration#{}:".format(iteration))

    old_centroids = centroids
    centroids = update_centroids(memberships, Z, initial_centroids)
    if np.alltrue(centroids == old_centroids):
        break

    old_memberships = memberships
    memberships = update_memberships(centroids, Z)
    if np.alltrue(memberships == old_memberships):
        break

    iteration = iteration + 1

# PLOTTING RESULTS #
cluster_colors = np.array(["#1f78b4", "#e31a1c", "#33a02c", "#6a3d9a", "#ff7f00"])
plt.figure(figsize=(8, 8))

for c in range(K):
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10, color=cluster_colors[c])

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Resulting Clusters")
plt.show()