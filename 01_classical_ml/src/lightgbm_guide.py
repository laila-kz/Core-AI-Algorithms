#lightGBM shares the same gradient boosting with XGBoost but LightGBM introduces innovations that make it incredibly fast and memory-efficient, especially for large datasets.
#a better vesion of gradient boosting 
#same principal as XGBoost but with different implementation and optimizations
#lightGBM s way of growing the tree is leaf-wise (best-first) rather than level-wise (breadth-first) like XGBoost. This allows LightGBM to focus on the most promising branches of the tree, leading to faster convergence and better accuracy.



#Key features of LightGBM include:
#1.Leaf-wise Tree Growth
#2.Histogram-based algorithm : deviding data ti buns  
#3. Gradient-based One-Side Sampling : LightGBM uses a technique called Gradient-based One-Side Sampling (GOSS) to speed up training. It focuses on the data points with larger gradients, which are more informative for training, while randomly sampling from the data points with smaller gradients. This allows LightGBM to maintain high accuracy while reducing the number of data points it needs to process.
#4.Exclusive Feature Bundling (EFB): combine features that are mutually exclusive (i.e., they never take non-zero values simultaneously) into a single feature, which reduces the number of features and speeds up training without sacrificing accuracy.
#5.Native Handling of Categorical Features and Missing Values

#when to use lightGBM:
# ✅ Big data
# ✅ Many features
# ✅ Production systems
# ✅ Kaggle big competitions


#how to avoid overfitting with lightGBM:
#max_depth: limits the maximum depth of the tree, preventing it from becoming too complex and overfitting the training data.
#min_data_in_leaf: sets the minimum number of samples required to be in a leaf node. This helps to prevent the model from creating leaves that are too small and overfitting the training data.
#feature_fraction: specifies the fraction of features to be randomly selected for each tree. This can help to reduce overfitting by introducing randomness into the feature selection process.

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#------------------------------
# Example usage of LightGBM for binary classification



# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#define and train the model
model_clf = lgb.LGBMClassifier(objective='binary',          # Task type
    boosting_type='gbdt',         # Traditional gradient boosting
    num_leaves=31,                 # Number of leaves in one tree
    learning_rate=0.05,            # Shrinkage factor
    n_estimators=100,              # Number of boosting rounds
    random_state=42,
    verbose=-1                     # Suppress training messages
)

model_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

#predict probabilities
y_pred_proba = model_clf.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
print(f'Predicted probabilities for the positive class: {y_pred_proba[:5]}')





#------------------------------
# Example usage of LightGBM for regression


from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

#create regression data
X,y = make_regression(
    n_samples = 1000,
    n_features = 10,
    noise =10,
    random_state= 42
)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42) 

#define and train the model
model_reg = lgb.LGBMRegressor(
    objective='regression',         # Task type
    boosting_type='gbdt',         # Traditional gradient boosting
    num_leaves=31,                 # Number of leaves in one tree
    learning_rate=0.05,            # Shrinkage factor
    n_estimators=100,              # Number of boosting rounds
    random_state=42,
    verbose=-1                     # Suppress training messages
)

model_reg.fit(X_train, y_train)


# Predict and evaluate
y_pred = model_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')






#--------------------------------
#advanced features and tips 

#1. Early Stopping: LightGBM supports early stopping, which can help prevent overfitting by stopping the training process when the performance on a validation set starts to degrade. You can use the `early_stopping_rounds` parameter in the `fit` method to specify the number of rounds to wait before stopping.
#This is a crucial feature. You train on a separate validation set, and LightGBM automatically stops when the performance on that set stops improving

#to stop the model when it starts memorizing the training data instead of learning general patterns, you can use early stopping. This is done by monitoring the performance on a validation set and stopping the training process when the performance starts to degrade.

#split training data into training and validation sets
X_train_sub , X_val , y_train_sub , y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#model 
model_clf = lgb.LGBMClassifier(n_estimators = 1000)
#n_estimators is the number of boosting rounds, we set it to a large number because we will use early stopping to determine the optimal number of rounds
#the model will build 1000 trees, but it will stop early if the performance on the validation set does not improve for a certain number of rounds
#which is dangerous because it can lead to overfitting if the model is allowed to train for too long without stopping

model_clf.fit(
    X_train_sub, y_train_sub, 
    eval_set=[(X_val, y_val)],  #tells the model to keep watching the performance on the validation set during training
    callbacks=[lgb.early_stopping(stopping_rounds=10)]  # Stop if no improvement for 10 rounds
#if the performance on the validation set does not improve for 10 consecutive rounds, the training will stop
    verbose=False
)


print(f'Best iteration: {model_clf.best_iteration_}')  # Best iteration determined by early stopping
print(f'Best score on validation set: {model_clf.best_score_}')  # Best score on the validation set at the best iteration




#-----------------------------------
#Handeling catgorical features
#LightGBM has native, highly efficient support for categorical features. Instead of one-hot encoding, you can just tell it which columns are categorical

#make a dataframe with categorical features
X_cat = pd.DataFrame({
    'age': np.random.randint(18, 70, 500),
    'gender': np.random.choice(['M', 'F'], 500),
    'income': np.random.randint(20000, 100000, 500)
})

#make a label 
y_cat = (X_cat['income'] > 50000).astype(int)  # Binary target based on income
#classifie if rich or not based on income

#split data
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42)

#model
model_cat = lgb.LGBMClassifier(random_state=42, verbose=-1)
model_cat.fit(X_train_cat, y_train_cat, categorical_feature=['gender'])

print("Model with categorical feature trained.")







#-----------------------------------
#understanding your model with feature importance
#we can which features are most important for the model's predictions using feature importance scores. LightGBM provides several methods to calculate feature importance, including 'split' (the number of times a feature is used in splits) and 'gain' (the average gain of splits which use the feature).

import matplotlib.pyplot as plt

# Get feature importances
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
importance = model_clf.feature_importances_

# Plot them
lgb.plot_importance(model_clf, figsize=(10, 6), max_num_features=10)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()




#-----------------------------------
#Hyperparameter tuning with LightGBM

#Note : the difference between hyperparamters and parameters 
#Parameters are what the model used to make predictions, they are learned from the data during training (e.g., weights in a linear model, split points in a decision tree).
#Hyperparameters : you chose them before training, they control the learning process and the structure of the model (e.g., learning rate, number of trees, max depth).
#learning_rate ,num_leaves, n_estimators, max_depth


from sklearn.model_selection import GridSearchCV
#grid search tests a lpt of combinations of hyperparameters to find the best one, but it can be very time-consuming, especially for large datasets and many hyperparameters.


params_grid = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100]
}


# Create a base model
model = lgb.LGBMClassifier(random_state=42, verbose=-1)

# Grid search
grid = GridSearchCV(estimator=model, param_grid=params_grid, cv=3, scoring='accuracy', verbose=1)
grid.fit(X_train, y_train)

print(f"Best parameters found: {grid.best_params_}")
print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")
