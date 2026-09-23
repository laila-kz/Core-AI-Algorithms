#***************************
#EDA necessary plots 
#***************************


#------------------------------
#1- histogram
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-whitegrid')
sns.set_palette('husl')


# Create sample data (ages of customers)
np.random.seed(42)
ages = np.random.normal(35, 12, 1000)  # mean=35, std=12, 1000 samples
ages = np.clip(ages, 18, 70)  # Clip between 18 and 70

plt.figure(figsize=(10, 6))
plt.hist(ages , bins = 20, color='skyblue', alpha = 0.7 ,edgecolor='black')
plt.title('Distribution of Customer Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True , alpha=0.3)
plt.axvline(np.mean(ages), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(ages):.2f}')
plt.axvline(np.median(ages), color='green', linestyle='dashed', linewidth=1, label=f'Median: {np.median(ages):.2f}')
plt.legend()
plt.tight_layout()
plt.show()







#------------------------------
#2- boxplot
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('seaborn-whitegrid')
sns.set_palette('husl')

np.random.seed(42)
ages = np.random.normal(35, 12, 1000)  # mean=35, std=12, 1000 samples
ages = np.clip(ages, 18, 70)  # Clip between 18

plt.figure(figsize=(8, 6))
plt.boxplot(ages, vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue', color='black'), medianprops=dict(color='red', linewidth=2))
plt.title('Boxplot of Customer Ages')
plt.xlabel('Age')
plt.show()







#------------------------------
#3- scatter plot


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-whitegrid')
sns.set_palette('husl')

# Sample data
np.random.seed(42)
ages = np.random.normal(35, 12, 100)
salary = ages * 1000 + np.random.normal(0, 5000, 100)  # salary related to age

plt.figure(figsize=(10, 6))
plt.scatter(ages , salary , alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Scatter Plot of Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()




#------------------------------
#4- Heatmap

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


plt.style.use('seaborn-whitegrid')
sns.set_palette('husl')

np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(18, 70, 100),
    'income': np.random.normal(50000, 15000, 100),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
    'city': np.random.choice(['New York', 'LA', 'Chicago', 'Houston', 'Phoenix'], 100)
})

df = pd.DataFrame(data)

corr = df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()








#***************************
#Model evaluation plots 
#****************************


#-----------------------------
#learning curve


from sklean.model_selection import learning_curve
from sklearn.datasets import LinearRegression
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

model = LinearRegression()

train_sizes, train_scores, test_scores = learning_curve(LinearRegression(), X, y, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training error', marker='o')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation error', marker='o')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid()
plt.show()






#-----------------------------
#Loss curve
# it shows the loss value during training for each epoch. It helps to understand how well the model is learning and if it is converging.

train_loss =[ 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05]
val_loss = [ 0.9, 0.7, 0.5, 0.35, 0.3, 0.25, 0.2, 0.18, 0.15, 0.1]

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss', marker='o')
plt.plot(val_loss, label='Validation Loss', marker='o')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()










#-----------------------------
#ROC curve
#it shows the performance of a classification model at all classification thresholds. It plots the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.

from sklean.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model= LogisticRegression()
model.fit(X_train, y_train)

y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr , label = f"AUc = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()








#-----------------------------
#Precision-Recall curve
#it shows the trade-off between precision and recall for different threshold values. It is particularly useful when dealing with imbalanced datasets.
#it shows balance between precision and recall for different threshold values. It is particularly useful when dealing with imbalanced datasets.

#difference between recall and precision:
#Recall : how many positive samples were correctly identified by the model out of all actual positive samples. It is calculated as TP / (TP + FN).
#Precision : how many positive predictions were actually correct out of all positive predictions made by the model. It is calculated as TP / (TP + FP).


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    
                                                    test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_probs = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

ap = average_precision_score(y_test, y_probs)


plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label=f'AP = {ap:.2f}')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid()
plt.show()









#-----------------------------
#Confusion matrix
#it shows the performance of a classification model by displaying the counts of true positives, true negatives, false positives, and false negatives. It helps to understand the types of errors the model is making.

#what to look for : false Positives , False negatives 
#when to use it : Logistic regression , Decision tree , Random forest , SVM , neural networks



from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

disp =ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()













#-----------------------------  
#Feature importance plot
#hightlight the most important features that contribute to the model's predictions. It helps to understand which features are driving the model's decisions and can be used for feature selection or interpretation.


from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
feature_names = [f'Feature {i}' for i in range(X.shape[1])]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()







#-----------------------------
#residual Plot 

#it shows the residuals (the differences between the observed and predicted values) on the y-axis and the predicted values on the x-axis. It helps to assess the goodness of fit of a regression model and to identify any patterns or heteroscedasticity in the residuals.




from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7, color='skyblue', edgecolor='black')
plt.axhline(0, color='red', linestyle='dashed', linewidth=1)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid()
plt.show()








#-----------------------------
#Calibration curve
#it shows the relationship between predicted probabilities and observed outcomes. It helps to assess the calibration of a classification model, which is how well the predicted probabilities reflect the true likelihood of an event.

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_probs = model.predict_proba(X_test)[:, 1]
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)


plt.figure(figsize=(10, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.title('Calibration Curve')  
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.grid()
plt.show()







#-----------------------------
#SHAP Summary Plot
#it explains why ur madel made a prediction by showing the contribution of each feature to the prediction. It helps to understand the importance of features and their interactions in the model's decision-making process.


import shap 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


X, y = load_breast_cancer(return_X_y=True)
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test)












#-----------------------------
#t-SNE plot
#it is a dimensionality reduction technique that helps to visualize high-dimensional data in a lower-dimensional
