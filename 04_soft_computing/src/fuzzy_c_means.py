import numpy as np 

class FCM:
    def __init__(self,n_clusters =3 , m =2 , max_iterations = 100 , tol = 1e-5):
        self.n_clusters = n_clusters 
        self.m = m  #a factor that controls how fuzzy the clustering is 
        self.max_iterations = max_iterations
        self.tol = tol #tolerance for convergence, which determines when to stop the algorithm based on the change in the membership matrix U between iterations

    def _initialize_membership(self, X):
        U = np.random.rand(X.shape[0], self.n_clusters) #randomly initialize the membership matrix U with values between 0 and 1
        U = U / np.sum(U, axis=1, keepdims=True) #normalize the membership values so that they sum to 1 for each data point
        return U
    
    def _update_centroids(self, X):
        um = self.U ** self.m #raise the membership matrix to the power of m to calculate the weighted membership values
        self.centroids = (um.T @ X) / np.sum(um.T, axis=1, keepdims=True) #update the centroids by calculating the weighted average of the data points based on their membership values
        #um.T @ X computes the weighted sum of data points for each cluster
        #um.T transposes the membership matrix to shape (n_clusters, n_samples)
        #np.sum(um.T, axis=1, keepdims=True) computes the total weight for each cluster

        #The division calculates the weighted average (centroid) for each cluster


    def _update_membership(self, X): #based on current centroids 
        n_samples = X.shape[0] 
        new_U = np.zeros((n_samples, self.n_clusters)
        ) #to store the updated membership values for each data point and cluster
        #iterate throught each data point and each cluster to calculate the new membership values based on the distance from the data point to the centroids and the current membership values. The formula for updating the membership values is derived from the Fuzzy C-Means algorithm, which assigns higher membership values to clusters that are closer to the data point and lower values to clusters that are farther away.
        for i in range(n_samples):
            for j in range(self.n_clusters):
                #Calculates Euclidean distance from data point i to centroid j
                #how far the point is fromthis cluster's centroid compared to other clusters. The numerator represents the distance from the data point to the current cluster's centroid, while the denominator sums the relative distances to all centroids, raised to the power of 2/(m-1) to account for the fuzziness of the clustering.
                numerator = np.linalg.norm(X[i] - self.centroids[j])
                denominator = 0
                for k in range(self.n_clusters):
                    denom_term = np.linalg.norm(X[i] - self.centroids[k])
                    denominator += (numerator / denom_term) ** (2 / (self.m - 1))
                new_U[i, j] = 1 / denominator
        
        return new_U
    

    def fit(self, X):
        n_samples = X.shape[0]
        self.U = self._initialize_membership(X) #initialize the membership matrix U

        for _ in range(self.max_iterations):
            self._update_centroids(X) #update the centroids based on the current membership matrix
            new_U = self._update_membership(X) #update the membership matrix based on the new centroids

            if np.linalg.norm(new_U - self.U) < self.tol: #check for convergence by comparing the new membership matrix with the previous one
                break
            self.U = new_U #update the membership matrix for the next iteration
    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - self.centroids, axis=1) #calculate the distance from the data point to each centroid
            predictions[i] = np.argmin(distances) #assign the data point to the cluster with the closest centroid

        return predictions
    
    