#what is XGBoost 
# eXtreme Gradient Boosting (XGBoost) is a powerful machine learning algorithm that is widely used for classification and regression tasks. It is an implementation of gradient boosting that is designed to be efficient, flexible, and portable. XGBoost is known for its high performance and scalability, making it a popular choice for data scientists and machine learning practitioners.
# is an optimized implementato of the gradient boosting algorithm 
#proven succeess in machine learning competitions and real-world applications
#XGBoost is an ensemble learning method that combines the predictions of multiple weak learners (decision trees) to create a strong predictive model. It works by iteratively adding new trees to the model, where each new tree is trained to correct the errors made by the previous trees. This process continues until a specified number of trees is reached or the model's performance stops improving.
#good for supervised learning tasks, such as classification and regression


#what is gradient boosting
# machine learning algorithm that sequentialliy ensembles weak predictive models intoa  single stronger model
#The idea is to build a model in a stage-wise fashion, where each new model is trained to correct the errors made by the previous models. The final prediction is made by combining the predictions of all the individual models, typically through a weighted average or majority vote.
#Gradient boosting works by minimizing a loss function, which measures the difference between the predicted values and the actual values. The algorithm iteratively adds new models to the ensemble, where each new model is trained to minimize the residual errors of the previous models. This process continues until a specified number of models is reached or the performance of the ensemble stops improving.
#it is good for both classification and regression tasks, and it can handle a wide range of data types and distributions. Gradient boosting is known for its high predictive accuracy and is often used in machine learning competitions and real-world applications.


#most common weak models to the ensemle are decision trees, which are simple and interpretable models that can capture complex relationships in the data. However, other types of weak learners can also be used, such as linear models or neural networks, depending on the specific problem and dataset. The choice of weak learner can affect the performance of the ensemble, and it is often a hyperparameter that can be tuned during model training.

#KEY IMPROVEMENTS OF XGBOOST OVER TRADITIONAL GRADIENT BOOSTING:
#1. Regularization: XGBoost includes L1 and L2 regularization terms in the objective function, which helps to prevent overfitting and improve the generalization of the model.
#the obj function of xgboost has a regularization term added to the loss function, which helps to control the complexity of the model and prevent overfitting. The regularization term can be either L1 (Lasso) or L2 (Ridge) regularization, or a combination of both. This allows XGBoost to effectively handle high-dimensional data and reduce the risk of overfitting, especially when dealing with large datasets.

#2. more scalable and efficient: XGBoost is designed to be highly efficient and scalable, making it suitable for large datasets and distributed computing environments. It uses a novel tree learning algorithm that can handle sparse data and missing values, and it also supports parallel processing and GPU acceleration.

#in conclusion XGBOost is a faster framework that can build better models


#Steps to implement XGBoost:
#1. Import the necessary libraries and modules, including XGBoost and any other required libraries for data preprocessing and evaluation.
#2. Load and preprocess the dataset, including handling missing values, encoding categorical variables, and splitting the data into training and testing sets.
#3. Create an instance of the XGBoost model and specify the hyperparameters, such
#4. Train the XGBoost model on the training data using the fit method.
#5. Evaluate the performance of the model on the testing data using appropriate evaluation metrics, such as accuracy, precision, recall, or mean squared error.
#6. Tune the hyperparameters of the XGBoost model using techniques such as grid search


#Example code for implementing XGBoost in Python:

from tkinter import _test
import xgboost as xgb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import warnings
warnings.filterwarnings('ignore')

#set random seed for reproducibility
np.random.seed(42)
print("*"*70)

print("XGBOOST TUTORIAL - PRACTICAL EXAMPLES")
print("*"*70)


#Example 1: Classification with XGBoost "binary classification  "
print("Example 1: Classification with XGBoost")
#generate synthetic dataset for binary classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Method 1: Using sklearn wrapper
from XGBoost import XGBClassifier

model_sk =XGBClassifier(
    n_estimators=100,        # Number of boosting rounds
    max_depth=3,              # Maximum tree depth
    learning_rate=0.1,        # Step size shrinkage
    subsample=0.8,            # Sample ratio per tree
    colsample_bytree=0.8,     # Feature ratio per tree
    reg_lambda=1.0,           # L2 regularization
    reg_alpha=0.0,            # L1 regularization
    random_state=42

)

#train the model
model_sk.fit(X_train, y_train)

#make predictions
y_pred_sk = model_sk.predict(X_test)
y_pred_proba_sk = model_sk.predict_proba(X_test)[:, 1]

#evaluate the model
accuracy_sk = accuracy_score(y_test, y_pred_sk)
print(f"Sklearn XGBoost Accuracy: {accuracy_sk:.4f}")
print("Classification Report:")
print(f"Train Score: {model_sk.score(X_train, y_train):.4f}")
print(f"Test Score: {model_sk.score(X_test, y_test):.4f}")


#Example 2: Classification with XGBoost "multiclass classification"
print("\nExample 2: Classification with XGBoost - Multiclass Classification")


#generate synthetic dataset for multiclass classification
X_multi, y_multi = make_classification(n_samples=1000, 
    n_features=10, 
    n_informative=8, 
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
    )

#split the dataset into training and testing sets
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

#XGboost automatically detects the number of classes and uses softmax as the objective function for multiclass classification
model_multi = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42
)

#train the model
model_multi.fit(X_train_multi, y_train_multi)

#make predictions
y_pred_multi = model_multi.predict(X_test_multi)
y_pred_proba_multi = model_multi.predict_proba(X_test_multi)

#evaluate the model
print(f"Accuracy: {accuracy_score(y_test_multi, y_pred_multi):.4f}")
print(f"Classes: {model_multi.classes_}")
print(f"Sample probabilities for the first 5 test samples:\n{y_pred_proba_multi[:5]}")




#Example 3: Regression with XGBoost
print("\nExample 3: Regression with XGBoost")
from XGBoost import XGBRegressor

#generate synthetic dataset for regression
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

#split the dataset into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

#Regression model 
model_reg = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)

#train the model
model_reg.fit(X_train_reg, y_train_reg)

#make predictions
y_pred_reg = model_reg.predict(X_test_reg)
#evaluate the model
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Mean Squared Error: {mse_reg:.4f}")
print(f"RMSE: {np.sqrt(mse_reg):.4f}")






#Example 4: Naive XGBoost api 
print("\nExample 4: Naive XGBoost api")


#convert to DMatrix format(optimized data structure for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#specify parameters for XGBoost
params = {'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'eval_metric': 'logloss',
    'seed': 42
}

#train the model using the naive XGBoost API
evals = [(dtrain, 'train'), (dtest, 'eval')]
model_naive = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10)

#make predictions
y_pred_naive = model_naive.predict(dtest)
y_pred_class = (y_pred_naive > 0.5).astype(int)

print(f"Naive XGBoost Accuracy: {accuracy_score(y_test, y_pred_class):.4f}")






# Exemple 5 : Cross-validation 

print("\nExample 5: Cross-validation and hyperparameter tuning with XGBoost")

#use sklean wrapper for cross-validation and hyperparameter tuning
cv_scores = cross_val_score(model_sk, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

#use native XGboose cv 
dtrain_full = xgb.DMatrix(X, label=y)
cv_results = xgb.cv(params, dtrain_full, num_boost_round=100, nfold=5, metrics='logloss', early_stopping_rounds=10, seed=42)
print(f"XGBoost CV Log Loss: {cv_results['test-logloss-mean'].min():.4f} at round {cv_results['test-logloss-mean'].idxmin()}")



#Example 6:  hyperparameter tuning with XGBoost

print("\nExample 6: Hyperparameter tuning with GridSearchCV")
param_grid ={
        'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]

}

#use smaller grid for the demo 
simple_grid = {
    'max_depth': [3],
    'learning_rate': [0.1, 0.3],
    'n_estimators': [50],
}
grid_search = GridSearchCV(estimator=XGBClassifier(random_state=42), param_grid=simple_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
print(f"Test Accuracy with Best Hyperparameters: {grid_search.score(X_test, y_test):.4f}")





#Example 7: Feature importance with XGBoost
print("\nExample 7: Feature importance with XGBoost")


#get features importance from the model
importance = model_sk.feature_importances_
feature_names = [f"Feature {i}" for i in range(X.shape[1])]

#create importance dataframe
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)
print("top 5 important features:")
print(importance_df.head())

#plot feature importance
plt.figure(figsize=(10 ,6))
plt.barh(importance_df['Feature'][:5], importance_df['Importance'][:5])
plt.xlabel('Importance')
plt.title('Top 5 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()





#Example 8: Early stopping with XGBoost
print("\nExample 8: Early stopping with XGBoost")
#train the model with early stopping
model_early = XGBClassifier(    
    n_estimators=1000,  # Large number of boosting rounds
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42
)

evals = [(dtrain, 'train'), (dtest, 'eval')]
model_early.fit(X_train, y_train, eval_set=evals, early_stopping_rounds=10, verbose=False)

print(f"Best iteration: {model_early.best_iteration}")
print(f"Test Accuracy at Best Iteration: {model_early.score(X_test, y_test):.4f}")







#Example 9: Handling missing values with XGBoost
print("\nExample 9: Handling missing values with XGBoost")

#generate synthetic dataset with missing values
X_missing, y_missing = make_classification(n_samples=1000, n_features=20
                                           
                                             , random_state=42) 

#introduce missing values randomly
missing_rate = 0.1
n_missing_samples = int(missing_rate * X_missing.size)
missing_indices = np.random.choice(X_missing.size, n_missing_samples, replace=False)
X_missing.ravel()[missing_indices] = np.nan

#split the dataset into training and testing sets
X_train_miss, X_test_miss, y_train_miss, y_test_miss = train_test_split(X_missing, y_missing, test_size=0.2, random_state=42)
#train the model on data with missing values
model_miss = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42
)

model_miss.fit(X_train_miss, y_train_miss)
#make predictions
y_pred_miss = model_miss.predict(X_test_miss)
#evaluate the model
accuracy_miss = accuracy_score(y_test_miss, y_pred_miss)
print(f"Accuracy with Missing Values: {accuracy_miss:.4f}")

#example 10: XGBoost with GPU acceleration
print("\nExample 10: XGBoost with GPU acceleration")
#check if GPU is available
if xgb.core._has_cuda():
    print("GPU is available. Training with GPU acceleration.")
    model_gpu = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        tree_method='gpu_hist'  # Use GPU for training
    )
    model_gpu.fit(X_train, y_train)
    y_pred_gpu = model_gpu.predict(X_test)
    accuracy_gpu = accuracy_score(y_test, y_pred_gpu)
    print(f"Accuracy with GPU Acceleration: {accuracy_gpu:.4f}")

else:
    print("GPU is not available. Please install XGBoost with GPU support to use this feature.")



#exemple 11:  load and save model 
print("\nExample 11: Saving and loading XGBoost model")

#save sklearn wrapper model
model_sk.save_model("xgb_sklearn_model.json")

#load the model
loaded_model_sk = XGBClassifier()
loaded_model_sk.load_model("xgb_sklearn_model.json")

#verefy that the loaded model gives the same predictions
y_pred_loaded_sk = loaded_model_sk.predict(X_test)
print(f"Loaded model accuracy: {accuracy_score(y_test, y_pred_loaded_sk):.4f}")

# Save native model

model_naive.save_model("xgb_native_model.json")

print("✅ Models saved successfully!")















# ============================================
# ADVANCED XGBOOST TECHNIQUES
# ============================================


#TECHNIQUE 1: custom onjective fucntion 
def custom_obj(y_true , y_pred):
    grad = y_pred - y_true  # Gradient for squared error 
    #the direction of descending error
    hess = np.ones_like(y_true)  # Hessian for squared error
    #the speed of convergence, for squared error it is constant
    return grad, hess


def custom_eval(y_true, y_pred):
    error = np.sqrt(np.mean((y_true - y_pred) ** 2))  # RMSE
    return 'rmse', error

model_custom = XGBRegressor(objective=custom_obj, eval_metric=custom_eval, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)



#TECHNIQUE 2: monotonic constraints
#monotonic constraints can be applied to ensure that the model's predictions are monotonic with respect to certain features. For example, if we know that increasing a feature should not decrease the predicted value, we can set a positive monotonic constraint on that feature. This can help improve interpretability and ensure that the model behaves in a way that is consistent with domain knowledge.

monotone_constraints = (1,0,0,0,0,0,0,0,0,0)  # Enforce monotonicity on the first feature
model_monotone = XGBRegressor(monotone_constraints=monotone_constraints, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)

model_monotone.fit(X_train_reg, y_train_reg)
print(f"Monotone Constrained Model RMSE: {np.sqrt(mean_squared_error(y_test_reg, model_monotone.predict(X_test_reg))):.4f}")




#TECHNIQUE 3: interaction constaints 
#tells the model which features are allowed to interact with each other. This can help reduce overfitting and improve interpretability by limiting the complexity of the model. For example, if we know that certain features should not interact with each other, we can set interaction constraints to prevent the model from learning those interactions.
#reduces overfitting by limiting the interactions between features, which can lead to a simpler and more interpretable model. By specifying which features are allowed to interact, we can also incorporate domain knowledge into the model and ensure that it behaves in a way that is consistent with our understanding of the problem.


interaction_constraints = [['Feature_0', 'Feature_1'], ['Feature_2', 'Feature_3']]  # Allow interactions between features 0 and 1, and between features 2 and 3

model_interact = XGBRegressor(interaction_constraints=interaction_constraints, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)

model_interact.fit(X_train_reg, y_train_reg)
print(f"Interaction Constrained Model RMSE: {np.sqrt(mean_squared_error(y_test_reg, model_interact.predict(X_test_reg))):.4f}")





#TECHNIQUE 4: GPU training 
#use GPU for training can significantly speed up the training process, especially for large datasets and complex models. XGBoost supports GPU acceleration through the 'tree_method' parameter, which can be set to 'gpu_hist' to use the GPU for training. This can lead to much faster training times compared to using the CPU, allowing you to experiment with larger datasets and more complex models without being limited by computational resources.
#instaead of using CPU 

#GPu is faster 

try:
    model_gpu = XGBClassifier(tree_method='gpu_hist', gpu_id=0, random_state=42)
    model_gpu.fit(X_train, y_train)
    print("GPU training successful!")
except:
    print("GPU not available, using CPU")







#TECHNIQUE 5: continouse training
#train the model on new data without having to start from zero 

model_cont = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model_cont.fit(X_train, y_train)

#get booster 
booster = model_cont.get_booster()

#train additional rounds on new data 
new_data = xgb.DMatrix(X_train[500:], label=y_train[500:])
booster - xgb.train(
    params = model_cont.get_params(),
    dtrain = new_data,
    num_boost_round = 50,
    xgb_model = booster  # Continue training from the existing model

)

print (f"Model updated with new data ")





#  TECHNIQUE 6: plotting results 
xgb.plot_importance(model_sk, max_num_features=10)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

xgb.plot_tree(model_sk, num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.title("Visualization of the First Tree")
plt.show()

