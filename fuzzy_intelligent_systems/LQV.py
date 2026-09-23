import numpy as np 

class LVQ:
    def __init__(self, learning_rate=0.01, n_prototypes_per_class=1, epochs=100):
        self.learning_rate = learning_rate #control the step size of prototype updates
        self.epochs  = epochs #nb of times we train the model
        self.n_prototypes_per_class = n_prototypes_per_class #nb of prototypes for training the model

    def _initialize_prototypes(self, X, y):
        self.prototypes = [] #empty list for prototypes and their labels 
        self.prototype_labels = []
        self.classes = np.unique(y) #unique classes form the training data 

        #for eahc class we need to filter the data to only include samples from that class 
        for cl in self.classes:
            X_cl = X[y == cl] #select samples from the class cl
            for _ in range(self.n_prototypes_per_class):
                idx = np.random.choice(len(X_cl)) #randomly select a sample from the class to initialize the prototype
                self.prototypes.append(X_cl[idx]) #store them as initial prototypes and their corresponding labels
                self.prototype_labels.append(cl)
        #convert the list of prototypes and labels to numpy arrays for efficient computation during training and prediction
        self.prototypes = np.array(self.prototypes)
        self.prototype_labels = np.array(self.prototype_labels)

    def fit(self, X, y):
        self._initialize_prototypes(X, y) #initalize the prototypes based on the training data

        #iterate the specified number of epochs and throught each sample in the training data to update the prototypes based on their distance to the samples and their labels
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x = X[i]
                label = y[i]
                distances = np.linalg.norm(self.prototypes - x, axis=1) #euclidient distance between the sample and each prototype to find the closest prototype (winner) and update it based on whether its label matches the sample's label or not
                winner_idx = np.argmin(distances)  #get the closest prototype index
                winner_label = self.prototype_labels[winner_idx] #get the label of the closest prototype

                #if the winner's label matches the sample's label, we move the prototype closer to the sample; otherwise, we move it away from the sample. This is done by adjusting the prototype's position in the feature space based on the learning rate and the difference between the sample and the prototype.
                self.prototypes[winner_idx] += self.learning_rate * (x - self.prototypes[winner_idx]) 
                
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.prototypes - x, axis=1) #distance from the samples to all prototypes to find the closest prototype and assign its label as the predicted class for the sample
            winner_idx = np.argmin(distances)
            predictions.append(self.prototype_labels[winner_idx])
        return np.array(predictions)