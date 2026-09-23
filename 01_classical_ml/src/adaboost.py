#Adaptive Boosting :
#Adaptive Boosting, commonly known as AdaBoost, is a machine learning algorithm that belongs to the family of ensemble methods. It was introduced by Yoav Freund and Robert Schapire in 1995. The main idea behind AdaBoost is to combine multiple weak learners (usually decision trees) to create a strong learner that can achieve high accuracy.

#what does it do exactly :
#trains the first model and then lokks for the error and then it trains another model to fix the errors of the first one and keeps repeating this process until it reaches the specified number of models or the performance stops improving.
#in the end it sums all the models together to make the final prediction. The models that perform better on the training data are given more weight in the final prediction, while those that perform poorly are given less weight. This way, AdaBoost focuses on the difficult cases and improves the overall performance of the ensemble model.

#which models does it use :
#Decision Stumps are commonly used as weak learners in AdaBoost. A decision stump is a simple decision tree that makes a decision based on a single feature. However, AdaBoost can also be used with other types of weak learners, such as linear classifiers or support vector machines.


#the stages of ada boost are as follows :
#1. Every data point starts with the same weight/importance
#2. The model makes predictions ,It identifies which data points it got wrong
#3. AdaBoost increases the weight of these misclassified points, It decreases weight of correctly classified points
#4. A new weak learner is trained, focusing more on the "hard" (high-weight) points , repeats until predictions are good enough

#adaboost is used mainly for classifying imgs , spam detection , diagnostic illnes, classification probelem 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns


#stump class "weak learner "

class DecisionStump:
    #it is basically a decision tree with only one split depth =1 
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1 #direction of comparison 

        self.alpha = None 
    def fit(self, X, y, sample_weights):
        #finds the best split or this stuump 
        n_samples , n_features = X.shape
        best_error  = float('inf')

        for feature_i in range (n_features):
            feature_values = X[:, feature_i]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values > threshold] = -1
                    error = np.sum(sample_weights * (predictions != y))
                    if error < best_error:
                        best_error = error
                        self.polarity = polarity
                        self.threshold = threshold
                        self.feature_index = feature_i

        return self
    
    def predict(self, X ):
        # make predictions based on the learned parameters
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)

        feature_values = X[:, self.feature_index]
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values > self.threshold] = -1

        return predictions
    

#ada boost class
class AdaBoost:
    def __init__(self, n_estimators=50):
        #n_estimators is the number of weak learners we want to use in our ensemble
        self.n_estimators = n_estimators
        self.models = []
        self.alpha = []
        self.training_errors = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples #initial weights
        y_= np.where(y == 0, -1, 1) #convert labels to -1 and 1

        for t in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y_, w)

            predictions = stump.predict(X)

            #compute the weighted error 
            err = np.sum(w[predictions != y_])/ np.sum(w)

            #compute alpha (importance of this weak learner)
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10)) #add small value to avoid division by zero

            #update sample weights
            w = w * np.exp(-alpha * y_ * predictions)
            w = w / np.sum(w) #normalize weights

            self.models.append(stump)
            self.alpha.append(alpha)
            self.training_errors.append(err)

            if(t +1 ) % 10 == 0:
                print(f"Iteration {t+1}/{self.n_estimators}, Error: {err:.4f}, Alpha: {alpha:.4f}")

        return self
    
    def predict(self, X,  threshold =0 ):
        #make prediction using weighted sum of weak learners

        predictions = np.zeros(X.shape[0])

        #weigthed sum of predictions from all weak learners
        for alpha , model in zip(self.alpha, self.models):
            predictions += alpha * model.predict(X)


        return np.sign(predictions) #return final prediction based on the sign of the weighted sum
    


    def predict_proba(self, X ):
        #get the scores 
        scores = np.zeros(X.shape[0])
        for alpha, model in zip(self.alpha, self.models):
            scores += alpha * model.predict(X)

        #use segmoid to convert to proba
        proba =1 / (1 + np.exp( - scores))
        return np.column_stack([1 - proba , proba])
    
    def score(self, X ,y):
        #computes accuracy 
        y_pred = self.predict(X)
        y_ = np.where(y == 0 , -1 ,1)
        return np.mean(y_pred == y_)
    



#---------------------------------------------
#Demo with synthetic data 




def demo_adaboost():
    print("ADABOOST FROM SCRATCH - COMPLETE DEMONSTRATION")


    np.random.seed(42)
    X,y = make_classification(n_samples = 500 , 
                              n_features = 5,
                              n_informations = 3,
                              n_redundant = 1 ,
                              n_clusters_per_class =1,
                              random_state= 42)
    
    X_train , X_test , y_train , y_test = train_test_split(X, y , test_size=0.2, random =42)


    print(f"\n📊 Dataset Info:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    
    # Train AdaBoost
    print("\n🚀 Training AdaBoost...")
    print("-" * 50)

    ada = AdaBoost(n_estimators=50)

    ada.fit(X_train, y_train)

    # Evaluate on test set
    print("\n📈 Evaluating on test set...")
    train_pred = ada.predict(X-train)
    test_pred = ada.predict(X_test)

    # Convert back to 0/1 for accuracy calculation
    train_acc = np.mean((train_pred == 1) == (y_train == 1))
    test_acc = np.mean((test_pred == 1) == (y_test == 1))
    
    print(f"Training Accuracy: {train_acc:.3f} ({train_acc*100:.1f}%)")
    print(f"Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    
    return ada, X_train, X_test, y_train, y_test


#-----------------------------
#visualization of training error over iterations
def plot_adaboost_learning(ada):
    fig , axes = plt.subplots(1,2,figsize=(12,5))

    # Plot 1: Training error over iterations
    axes[0].plot(ada.training_errors, 'b-', linewidth=2)
    axes[0].set_xlabel('Boosting Round', fontsize=12)
    axes[0].set_ylabel('Weighted Error', fontsize=12)
    axes[0].set_title('Training Error Over Time', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Alpha values (stump weights)
    axes[1].plot(ada.alphas, 'r-', linewidth=2)
    axes[1].set_xlabel('Boosting Round', fontsize=12)
    axes[1].set_ylabel('Alpha (Stump Weight)', fontsize=12)
    axes[1].set_title('Stump Importance Over Time', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_decision_boundary(ada, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = ada.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='Set1', edgecolor='k')
    plt.title('AdaBoost Decision Boundary', fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend(title='Class')
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_stump_sequence(ada, X, y , n_stumps=5):
    #how the first stumps split the data 
    n_samples = X.shape[0]
    fig, axes = plt.subplot(1, n_stumps, figsize=(20, 4))

    #first two features
    X_vis = X[:, :2]

    for i in range(n_stumps):
        stump = ada.models[i]
        predictions = stump.predict(X_vis)

        axes[i].scatter(X_vis[:, 0], X_vis[:, 1], c=predictions, cmap='coolwarm', edgecolor='k')
        axes[i].set_title(f'Stump {i+1} (Alpha={ada.alpha[i]:.2f})', fontsize=12)
        axes[i].set_xlabel('Feature 1', fontsize=10)
        axes[i].set_ylabel('Feature 2', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()







#-----------------------------------
#compare with sklearn implementation
def compare_with_sklearn(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import AdaBoostClassifier

    print("\n🔍 Comparing with scikit-learn's AdaBoost...")
    print("our Ada boost")
    our_ada = AdaBoost(n_estimators=50)
    our_ada.fit(X_train, y_train)
    our_pred = our_ada.predict(X_test)
    our_acc = np.mean((our_pred == 1) == (y_test == 1))
    print(f"Accuracy: {our_acc:.4f}")

    #sklearn s ada boost 
    print("\nscikit-learn's AdaBoost")
    sklearn_ada = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth=1),  # Stumps
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
)
    sklearn_ada.fit(X_train, y_train)
    sklearn_pred = sklearn_ada.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    print(f"Accuracy: {sklearn_acc:.4f}")

    print(f"\nComparison:")
    print(f"Our AdaBoost Accuracy: {our_acc:.4f}")
    print(f"Scikit-learn AdaBoost Accuracy: {sklearn_acc:.4f}")
    print(f"\nDifference: {abs(our_acc - sklearn_acc):.4f}")
    
    return our_ada, sklearn_ada

#how samples weights change during training 
def visualize_sample_weights(ada, X, y):
    
    np.random.seed(42)
    
    # Create simple 2D dataset
    X = np.random.randn(20, 2)
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Train AdaBoost and track weights
    ada = AdaBoostFromScratch(n_estimators=5)
    
    # Manual training to capture weights at each step
    n_samples = X.shape[0]
    w = np.ones(n_samples) / n_samples
    y_ = np.where(y == 0, -1, 1)
    
    weight_history = [w.copy()]
    
    for t in range(5):
        stump = DecisionStump()
        stump.fit(X, y_, w)
        
        predictions = stump.predict(X)
        err = np.sum(w[predictions != y_]) / np.sum(w)
        alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
        
        w = w * np.exp(-alpha * y_ * predictions)
        w = w / np.sum(w)
        
        weight_history.append(w.copy())
    
    # Plot weight evolution
    for i, ax in enumerate(axes.flatten()[:6]):
        if i < len(weight_history):
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', 
                                s=200 * weight_history[i] * len(weight_history[i]), 
                                alpha=0.6, edgecolors='black')
            
            # Add weight values as text
            for j, (x, y_point) in enumerate(X):
                ax.text(x, y_point+0.1, f'{weight_history[i][j]:.2f}', 
                       ha='center', fontsize=8)
            
            if i == 0:
                ax.set_title(f'Initial - All Weights Equal')
            else:
                ax.set_title(f'After Stump {i} - Focus on Misclassified')
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Sample Weight Evolution: Wrong Predictions Get Higher Weights', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    ada, X_train , X_test , y_train , y_test = demo_adaboost()

    plot_adaboost_learning(ada)

    if X_train.shape[1] >= 2:
        plot_decision_boundary(ada, X_train, y_train)
        visualize_stump_sequence(ada, X_train, y_train)

    visualize_sample_weights(ada, X_train, y_train)

    compare_with_sklearn(X_train, X_test, y_train, y_test)

    
    