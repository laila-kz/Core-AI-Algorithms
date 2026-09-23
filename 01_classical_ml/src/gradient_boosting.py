#Gradient boosting is a powerful machine learning technique that builds an ensemble of weak learners (usually decision trees) in a sequential manner. Each new tree is trained to correct the errors made by the previous trees, resulting in a strong predictive model. Below is a simple implementation of gradient boosting from scratch using Python.
#basically an ml methode where we make a lot of small models where one learns from the errors of the  ones before
# it uses small decision trees as the weak learners and combines them to create a strong predictive model. The idea is to iteratively train new trees to correct the errors made by the previous trees, which helps to improve the overall performance of the model.
# it computes the residuals (the difference between the actual target values and the predicted values) and fits a new tree to these residuals. The predictions from the new tree are then added to the previous predictions, and this process is repeated for a specified number of iterations.
# trains the next model on the residuals of the previous model, which helps to correct the errors made by the previous model and improve the overall performance of the ensemble.
# adds the new tree 
# repeats the process for a specified number of iterations, allowing the model to learn from its mistakes and improve its predictions over time.


try:
    from decision_tree import DecisionTree
except ImportError:
    from .decision_tree import DecisionTree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score


#gradient boosting for regression 
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        
        self.trees = []  # Will store your DecisionTree objects
        self.initial_prediction = None
        self.train_errors = []
        self.val_errors = []


    def fit(self, X, y,X_val=None , y_val =None , verbose= True ):
        #fit the model using the decisionTree class we made before
        #Each new tree predicts the RESIDUALS (negative gradients)

        n_samples = X.shape[0]

        #initial prediction is the mean of the target values
        self.initial_prediction = np.mean(y)
        current_predictions = np.full(n_samples, self.initial_prediction)

        if X_val is not None:
            val_predictions = np.full(X_val.shape[0], self.initial_prediction)

        for i in range(self.n_estimators):
            #calculate residuals
            residuals = y - current_predictions 

            #subsample if needed 
            if self.subsample < 1.0:
                n_subsample = int(n_samples * self.subsample)
                indices = np.random.choice(n_samples, n_subsample, replace=False)
                X_subsample = X[indices]
                residuals_subsample = residuals[indices]
            else:
                X_subsample = X
                residuals_subsample = residuals

            #use decision tree to predict the residuals
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)

            tree.fit(X_subsample, residuals_subsample) # train it on residuals 
            tree_predictions = tree.predict(X) # predict the residuals for all samples

            current_predictions += self.learning_rate * tree_predictions # update the current predictions
            self.trees.append(tree) # store the tree

            #calculate training error
            train_error = mean_squared_error(y, current_predictions)
            self.train_errors.append(train_error)

            #calculate validation error if validation data is provided
            if X_val is not None and y_val is not None:
                val_predictions += self.learning_rate * tree.predict(X_val)
                val_error = mean_squared_error(y_val, val_predictions)
                self.val_errors.append(val_error)

            #print progress
            if verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}/{self.n_estimators}, Train Error: {train_error:.4f}, Val Error: {val_error:.4f}" if X_val is not None else f"Iteration {i + 1}/{self.n_estimators}, Train Error: {train_error:.4f}")

            else:
                print(f"Iteration {i + 1}/{self.n_estimators}, Train Error: {train_error:.4f}")

        return self 
        

    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions
    




#Gradient boosting for classification
class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample

        self.trees = []
        self.initial_prediction = None
        self.train_errors = []
        self.val_errors = []
        self.classes_ = None

    def sigmoid(self, x):
        x= np.clip(x, -250, 250)  # Prevent overflow
        return 1 / (1 + np.exp(-x))
    
    def log_odds(self,p):
        #convert proba to log odds 
        #log(p / (1 - p))

        p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid division by zero
        return np.log(p / (1 - p))
    
    def compute_residuals(self, y_true, y_pred_proba):
        #compute the residuals for classification 
        #residuals = y_true - predicted probabilities
        probabilities = self.sigmoid(y_pred_proba)
        return y_true - probabilities
    
    def fit(self, X ,y, X_val =None, y_val= None , verbose= True):
        #fit the gradient boosting classifier using the decision tree class we made before
        n_samples = X.shape[0]

        y = y.astype(float)  # Ensure labels are float 

        self.classes_ = np.unique(y) #store classes 

        p=  np.mean(y)
        self.initial_prediction = self.log_odds(p) #initial prediction is the log odds of the positive class

        current_predictions = np.full(n_samples, self.initial_prediction)

        if X_val is not None:
            val_predictions = np.full(X_val.shape[0], self.initial_prediction)
        
        if verbose:
            print(f"\nTraining Gradient Boosting Classifier")
            print(f"Initial prediction (log odds): {self.initial_prediction:.4f}")
            print(f"Initial probability: {self.sigmoid(self.initial_prediction):.4f}")
            print("-" * 60)

        for i in range(self.n_estimators):
            residuals = self.compute_residuals(y, current_predictions)

            if self.subsample < 1.0:
                n_subsample = int(n_samples * self.subsample)
                indices = np.random.choice(n_samples, n_subsample, replace=False)
                X_subsample = X[indices]
                residuals_subsample = residuals[indices]
            else:
                X_subsample = X
                residuals_subsample = residuals

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_subsample, residuals_subsample)
            tree_predictions = tree.predict(X)

            current_predictions += self.learning_rate * tree_predictions
            self.trees.append(tree)

            train_error = mean_squared_error(y, self.sigmoid(current_predictions))
            self.train_errors.append(train_error)

            if X_val is not None and y_val is not None:
                val_predictions += self.learning_rate * tree.predict(X_val)
                val_error = mean_squared_error(y_val, self.sigmoid(val_predictions))
                self.val_errors.append(val_error)

            if verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}/{self.n_estimators}, Train Error: {train_error:.4f}, Val Error: {val_error:.4f}" if X_val is not None else f"Iteration {i + 1}/{self.n_estimators}, Train Error: {train_error:.4f}")
            else:
                print(f"Iteration {i + 1}/{self.n_estimators}, Train Error: {train_error:.4f}")


        return self
    
    def predict_proba(self, X):
        #predict class probabilies 
        raw_predictions = np.full(X.shape[0], self.initial_prediction)

        #add each tree s contribution
        for tree in self.trees:
            raw_predictions += self.learning_rate * tree.predict(X)

        probabilities = self.sigmoid(raw_predictions)
        return np.vstack([1 - probabilities, probabilities]).T
    
    def predict(self, X , threshold=0.5):
        #predict class labels based on predicted probabilities 
        proba = self.predict_proba(X)[:, 1]  # Probability of the positive class
        return (proba >= threshold).astype(int)
    

    def log_loss(self, y_true , y_pred_proba):
        #compute log loss "cross  entropy" for evaluation 
        # formule : -[y * log(p) + (1-y) * log(1-p)]

        y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)  # Avoid log(0)
        return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    
    def score(self ,X , y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
























#-----------------------------------------------
# how to use both classes together with a demonstration
if __name__ == "__main__":
    # Generate synthetic regression data
    from sklearn.datasets import make_regression, make_classification

    # Regression example
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    regressor.fit(X_train_reg, y_train_reg, X_val=X_val_reg, y_val=y_val_reg)

    # Classification example
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    classifier.fit(X_train_clf, y_train_clf, X_val=X_val_clf, y_val=y_val_clf)   


