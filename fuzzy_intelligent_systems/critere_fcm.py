#**********Les indices de validation de la classification floue (FCM)**********
import numpy as np

# 1. Partition Coefficient (à MAXIMISER)
def partition_coefficient(U):
    return np.sum(U**2) / U.shape[0]


# 2. Partition Entropy (à MINIMISER)
def partition_entropy(U):
    return -np.sum(U * np.log(U + 1e-10)) / U.shape[0]


# 3. Xie-Beni (à MINIMISER)
def xie_beni(X, U, centroids, m):
    n = X.shape[0]

    # compacité
    dist = np.linalg.norm(X[:, None] - centroids, axis=2)**2
    numerator = np.sum((U**m) * dist)

    # séparation minimale entre centres
    min_dist = np.min([
        np.linalg.norm(c1 - c2)**2
        for i, c1 in enumerate(centroids)
        for j, c2 in enumerate(centroids) if i != j
    ])

    return numerator / (n * min_dist + 1e-10)


# 4. Fukuyama-Sugeno (à MINIMISER)
def fukuyama_sugeno(X, U, centroids, m):
    mean_global = np.mean(X, axis=0)

    dist1 = np.linalg.norm(X[:, None] - centroids, axis=2)**2
    term1 = np.sum((U**m) * dist1)

    dist2 = np.linalg.norm(centroids - mean_global, axis=1)**2
    term2 = np.sum((U**m).sum(axis=0) * dist2)

    return term1 - term2
