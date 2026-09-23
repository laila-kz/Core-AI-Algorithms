''' Stochastic Gradient Descent (SGD)
    we train a model by minimizing a cost function J(θ) using an iterative optimization algorithm called Stochastic Gradient Descent (SGD).
    the difference between SGD and Batch Gradient Descent :
            -gradient descent: You compute the error using ALL data points->then update the weigths once 
            -SGD : Use ONE random data point at a time to update the model.
'''
import numpy as np

class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # Shuffle data at the start of each epoch to ensure randomness
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_samples):
                # 1. Select a single random sample
                xi = X_shuffled[i]
                yi = y_shuffled[i]

                # 2. Predict (Linear: y = wx + b)
                prediction = np.dot(xi, self.weights) + self.bias

                # 3. Calculate Gradients for Mean Squared Error
                # Derivative of (pred - y)^2 is 2 * (pred - y)
                error = prediction - yi
                dw = xi * error
                db = error

                # 4. Update Parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
            # Optional: Print progress every 10 epochs
            if epoch % 10 == 0:
                current_loss = np.mean((np.dot(X, self.weights) + self.bias - y)**2)
                print(f"Epoch {epoch}: Loss {current_loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# --- Quick Test ---
if __name__ == "__main__":
    # Generate some dummy data: y = 2x + 5
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([7, 9, 11, 13, 15])

    model = StochasticGradientDescent(learning_rate=0.01, epochs=50)
    model.fit(X, y)

    print(f"\nFinal Weights: {model.weights[0]:.2f} (Target: 2.0)")
    print(f"Final Bias: {model.bias:.2f} (Target: 5.0)")
