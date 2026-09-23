#what is CatBoost :
#CatBoost is a machine learning algorithm that belongs to the family of gradient boosting algorithms. It is developed by Yandex, a Russian technology company. CatBoost stands for "Categorical Boosting" and is designed to handle categorical features effectively, which is a common challenge in machine learning.
#CatBoost is known for its ability to handle categorical features without the need for extensive preprocessing, such as one-hot encoding. It uses a technique called "ordered boosting" to build decision trees, which helps to reduce overfitting and improve the model's performance. CatBoost also includes various regularization techniques to further enhance its generalization capabilities.

# builds decision trees sequentially to minimize errors, excelling at handling categorical data directly without manual pre-processing
#it builds a tree and looks for the eroros then it builds another tree to fix the issues and keeps repeating till it reaches the specified number of trees or the performance stops improving


import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Generate synthetic data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.normal(50000, 15000, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'city': np.random.choice(['New York', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_samples),
    'target': np.random.randint(0, 2, n_samples)
})

#introdue some logic to make the target meaningful
data.loc[(data['age'] > 40) & (data['education'].isin(['Master', 'PhD'])), 'target'] = 1
data.loc[(data['city'] == 'New York') & (data['income'] > 60000), 'target'] = 1

#sepaarte features and target
X= data.drop('target', axis=1)
y = data['target']

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#MOST IMPORTANT : IDENTIFY THE CATEGORICAL FEATURES
cat_features =['gender', 'education', 'city']


model_clf = CatBoostClassifier(iterations=200,           # Number of boosting rounds
    learning_rate=0.1,        
    depth=6,                   # Tree depth
    loss_function='Logloss',   # For binary classification
    verbose=50,                # Print progress every 50 iterations
    random_seed=42
)

model_clf.fit( X_train, y_train,
    cat_features=cat_features,   # 👈 Specify which columns are categorical
    eval_set=(X_test, y_test),   # Watch performance on test set
    plot=True                     # 👈 This opens a cool interactive plot!
)



# Predict and evaluate
y_pred = model_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Get predicted probabilities
y_pred_proba = model_clf.predict_proba(X_test)[:, 1]
print(f"Sample probabilities: {y_pred_proba[:5]}")







#------------------------------
#Regression example with CatBoost


from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# Create regression data
X_reg = X.copy()
y_reg = X['income'] + np.random.normal(0, 5000, n_samples)  # Predict income

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Initialize regressor
model_reg = CatBoostRegressor(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    verbose=50,
    random_seed=42
)

# Train
model_reg.fit(
    X_train_r, y_train_r,
    cat_features=cat_features,
    eval_set=(X_test_r, y_test_r)
)

# Predict and evaluate
y_pred_r = model_reg.predict(X_test_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
print(f"RMSE: {rmse:.2f}")










#------------------------------
#Advanced tips:




#Find the best categorical features: CatBoost can automatically detect categorical features, but you can also specify them manually. Experiment with different sets of categorical features to see which ones improve performance.
model_tuned = CatBoostClassifier(
    iterations = 200,
    one_hot_max_size=10,  # Limit one-hot encoding to top 10 categories
    verbose = False , 
    random_seed=42
)

model_tuned.fit(X_train, y_train, cat_features=cat_features)




#understand feature importance: CatBoost provides built-in methods to analyze feature importance. Use these insights to identify which features are most influential in your model's predictions and consider removing less important features to simplify the model.

import matplotlib.pyplot as plt

# Get standard feature importance
feature_importance = model_clf.feature_importances_
feature_names = X.columns

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 👇 Get feature interaction importance (which features work together)
interaction_importance = model_clf.get_feature_importance(type='Interaction')
print("Feature Interaction Strengths:")
print(interaction_importance)








#Hyperparameter tuning: Like other boosting algorithms, CatBoost has several hyperparameters that can be tuned to improve performance. Consider using techniques like grid search or random search to find the optimal combination of parameters for your specific dataset and problem.
from sklearn.model_selection import GridSearchCV

# Define a simple parameter grid
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5]  # L2 regularization on leaf values
}

# Create a base model
model = CatBoostClassifier(iterations=100, verbose=False, random_seed=42)

# Grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
grid.fit(X_train, y_train, cat_features=cat_features)

print(f"Best parameters: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")




#save and load the mdoel 
model_clf.save_model('catboost_model.cbm')

loaded_model = CatBoostClassifier()
loaded_model.load_model('catboost_model.cbm')

y_pred_loaded = loaded_model.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"Accuracy of loaded model: {accuracy_loaded:.4f}")







#Native GPU support: CatBoost can leverage GPU acceleration for faster training on large datasets. If you have a compatible GPU, consider enabling this feature to speed up the training process.
# Use GPU for training
model_gpu = CatBoostClassifier(
    iterations=200,
    task_type='GPU',      # 👈 This tells CatBoost to use GPU
    devices='0',           # Specify which GPU(s) to use
    verbose=False
)

model_gpu.fit(X_train, y_train, cat_features=cat_features)



