# A decision tree is a supervised learning algorithm that works
# by recursively splitting the data based on feature values to make predictions. 

# Nodes:
# Root Node: Topmost node representing the entire dataset
# Internal Nodes: Nodes where splits happen
# Leaf Nodes: Terminal nodes with predictions

# Splitting Criteria:(how to split)
# Gini Impurity: Measures how often a randomly chosen element would be incorrectly labeled
# Entropy/Information Gain: Measures reduction in uncertainty
# Variance Reduction: For regression trees

# Tree Building Process:
# Start with all data at root node
# Find best feature and split point
# Split data into child nodes
# Repeat recursively until stopping criteria


#taining :
#calculate information gain for each feature and split point
#devide set with that feature and value that gives the highest information gain
#devide tree and do the same for all created braches 
#untill stopping criteria is met (max depth, min samples, etc.)

#testing:
#follow the tree intil u reach a leaf node 
#return the most common class label in that leaf node (for classification) or the mean value (for regression)


#information gain:Entropy(Parent) - Weighted Average * Entropy(Children)
#stop criteria: maximum depth , minimum samples per leaf, minimum samples for split, etc.
#entropy is the lack of order in the data, so we want to minimize it. P(X)= number of samples in class X / total number of samples
#entroy = - sum(p_i * log2(p_i)) where p_i is the proportion of class i in the node
#Entropy is a measure of impurity or uncertainty in a dataset. 
#        Low entropy = Everyone agrees, very certain, pure group
#        High entropy = Everyone disagrees, very uncertain, mixed group

import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature  # Feature index for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = None  # Value for leaf nodes (class label or mean value)

    def is_leaf_node(self):
        return self.value is not None
    

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2, n_features=None):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features = None
    


    def fit(self, X, y):
        self.n_features = X.shape[1] if self.n_features is None else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))


        #check stopping criteria
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split): 
            leaf_value = self._most_common_label(y) 
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        #find best split 
        best_threshold ,best_feature = self._best_split(X, y, feat_idxs)



        #create child nodes and recursively grow the tree
        left_idxs , right_idxs = self._split(X[:, best_feature] , best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)
    

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _best_split(self, X, y, feat_idxs):  
        #all the possible split out there what s the best one 
        best_gain = -1 
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs: 
            X_column = X[:, feat_idx] 
            thresholds = np.unique(X_column) 
            for threshold in thresholds: 
                gain = self._information_gain(y, X_column, threshold) 

                if(gain > best_gain): 
                    best_gain = gain 
                    split_idx = feat_idx 
                    split_threshold = threshold 
        return split_threshold, split_idx 
    
    def _information_gain(self, y, X_column, threshold):
#parent entropy parent_entropy = self._entropy(y) #generate split left_idxs = np.where(X_column <= threshold)[0] right_idxs = np.where(X_column > threshold)[0] if len(left_idxs) == 0 or len(right_idxs) == 0: return 0 #weighted average of the entropy of the children n = len(y) n_left, n_right = len(left_idxs), len(right_idxs) e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs]) child_entropy = (n_left / n) * e_left + (n_right / n) * e_right #information gain is difference in parent and children entropies ig = parent_entropy - child_entropy return ig def _entropy(self, y): hist = np.bincount(y) ps = hist / len(y) return -np.sum([p * np.log2(p) for p in ps if p > 0])
        # parent entropy
        parent_entroy = self.entropy(y) #generate split left_idxs = np.where(X_column <= threshold)[0] right_idxs = np.where(X_column > threshold)[0] if len(left_idxs) == 0 or len(right_idxs) == 0: return 0 #weighted average of the entropy of the children n = len(y) n_left, n_right = len(left_idxs), len(right_idxs) e_left, e_right = self.entropy(y[left_idxs]), self.entropy(y[right_idxs]) child_entropy = (n_left / n) * e_left + (n_right / n) * e_right #information gain is difference in parent and children entropies ig = parent_entroy - child_entropy return ig
        #create children 
        left_idx , right_idx = self._split(X_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0: return 0

        # calculate the weighted entropy of the children
        n = len(y)
        n_l , n_r = len(left_idx), len(right_idx) 
        e_l , e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx]) 
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        #calculate information gain as the difference in parent and children entropies
        information_gain = parent_entroy - child_entropy
        return information_gain
    


    def _split(self, X_column, threshold):
        left_idxs  = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y) #count the number of samples in each class ps = hist / len(y) #proportion of samples in each class return -np.sum([p * np.log2(p) for p in ps if p > 0]) #calculate entropy using the formula
        ps = hist/ len(y)
        for p in ps:
            if p > 0: 
                return -np.sum(p * np.log2(p))
            return 0
        




    def predict(self, X):
        return  np.array([self._traverse_tree(x) for x in X])
    
    def _traverse_tree(self, x, node=None):
        if node is None:
            node = self.root

        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    