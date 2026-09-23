#Convolutional Neural Network : a neural network specialed in images and videos recognition and classification


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split



# 1- fonctions d activation : RELU , Softmax , Sigmoid 
class ReLU:
    #f(x) = max(0, x)
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backword(self, grad_output):
        return grad_output * (self.input > 0)
    
class Softmax:
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
    
    def backword(self, grad_output):
        return grad_output
    
class Sigmoid:
    #f(x) = 1/(1+e^(-x))
    def forward(self, x):
        self.output = 1 /(1 + np.exp(-np.clip(x , -250 , 250)))
        return self.output
    
    def backword(self, grad_output):
        return grad_output * self.output * (1 - self.output)
    


#2. Loss function : how much the predicted output is far from the actual output : Cross Entropy Loss
#MSELoss : Mean Squared Error Loss for regression problems
#CrossEntropyLoss : for classification problems

class CrossEntropyLoss:
    #it is build with softmax gradient
    def forward(self, predictions , targets ):
        #apply softmax to the predictions
        exp_predictions = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        softmax_predictions = exp_predictions / np.sum(exp_predictions, axis=1, keepdims=True)
        #calculate the cross entropy loss
        n_samples = len(predictions)
        correct_log_probs =-np.log(softmax_predictions[range(n_samples), targets.argmax(axis=1)] + 1e-10)
        loss = np.sum(correct_log_probs) / n_samples

        self.softmax_preds = softmax_predictions
        self.targets = targets

        return loss
    
    def backword(self):
        #gradient of cross entropy with softmax
        return (self.softmax_preds - self.targets) / len(self.targets)
    

class MSELoss:
    def forward(self, perdictions , targets):
        self.predictions = perdictions
        self.targets = targets
        return np.mean((perdictions - targets) ** 2)
    
    def backword(self):
        return 2 * (self.predictions - self.targets) / len(self.targets)
    




#-------------------------
#CNN : Convolutional Neural Network

#the CNN layer : convolutional layer , pooling layer , fully connected layer
#what it does: it takes a picture as input and applies filter "kernesl " and returns feature maps 
#forawrd pass : it applies the convolution operation between the input and the kernels and returns the feature maps
#backward pass : it calculates the gradients of the kernels and the input and updates the kernels using gradient descent
#update the kernels : kernel = kernel - learning_rate * gradient

class Conv2D:
    def _init_(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.in_channels = in_channels #the number of channels in the input image (e.g. 3 for RGB)
        self.out_channels = out_channels #the number of filters (kernels) we want to apply to the input image
        self.kernel_size = kernel_size #the size of the filter (e.g. 3 for a 3x3 filter)
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale  #initialize the weights of the filters using He initialization
        self.biases = np.zeros(out_channels)

        self.d_weights = None 
        self.d_biases = None
        self.input = None


    #forward takes an input image and applies the convolution operation between the input and the kernels and returns the feature maps

    def forward(self, x):
        self.input = x 
        batch_size, in_channels, height , width = x.shape

        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            height += 2 * self.padding
            width += 2 * self.padding

        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        #perform the actual convolution operation
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        receptive_field = x_padded[b, :, h_start:h_end, w_start:w_end]

                        #apply the convolution operation
                        output[b, oc, h, w] = np.sum(receptive_field * self.weights[oc]) + self.biases[oc]

        return output
    
    #backward pass : it calculates the gradients of the kernels and the input and updates the kernels using gradient descent
    def backward(self, grad_output):
        batch_size = grad_output.shape[0]
        _, _, out_height, out_width = grad_output.shape

        #get input dim 
        if self.padding > 0:
            padded_input = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            _, _, padded_h, padded_w = padded_input.shape
        else:
            padded_input = self.input
            _, _, padded_h, padded_w = self.input.shape[2], self.input.shape[3]

        #initialize gradients
        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)
        grad_input = np.zeros_like(padded_input)

        #compute the gradient 
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        receptive_field = padded_input[b, :, h_start:h_end, w_start:w_end]
                        self.d_weights[oc] += grad_output[b, oc, h, w] * receptive_field
                        self.d_biases[oc] += grad_output[b, oc, h, w]

                        if self.padding == 0 :
                            grad_input[b, :, h_start:h_end, w_start:w_end] += \
                                grad_output[b, oc, h, w] * self.weights[oc]
                            
        #normalize the gradients by the batch size
        self.d_weights /= batch_size
        self.d_biases /= batch_size
        return grad_input 
    

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases



#Max Polling Layer : it is used to downsample the feature maps and reduce the spatial dimensions of the input
#it minimize the computational cost and the number of parameters in the model
#it takes the maximum value in a given window of the input feature map and returns a down
class MaxPool2D:
    def __init__(self, pool_size =2, stride =2):
        self.pool_size = pool_size 
        self.stride = stride 
        self.input = None
        self.max_indices = None

    def forward(self, x):
        self.input = x
        batch_size , channels, height, width = x.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size

                        window = x[b, c, h_start:h_end, w_start:w_end]
                        # Find max value and its position
                        max_val = np.max(window)
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        
                        output[b, c, h, w] = max_val
                        self.max_indices[b, c, h, w] = [h_start + max_pos[0], w_start + max_pos[1]]
        
        return output
    

    def backward(self, grad_output):
        batch_size, channels, out_height, out_width = grad_output.shape
        grad_input = np.zeros_like(self.input)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        max_h, max_w = self.max_indices[b, c, h, w]
                        grad_input[b, c, max_h, max_w] += grad_output[b, c, h, w]

        return grad_input
    




#fully connected layer : it is used to connect the output of the convolutional and pooling layers to the output layer of the model
#it takes the output of the convolutional and pooling layers and flattens it into a 1D vector and applies a linear transformation to it

class Linear :
    def _init_(self, in_features , out_features):
        self.in_features = in_features 
        self.out_features = out_features

        #initialize weights and biases using He initialization
        scale = np.sqrt(2. / in_features)
        self.weights = np.random.randn(in_features, out_features) * scale
        self.biases = np.zeros(out_features)

        self.d_weights = None
        self.d_biases = None
        self.input = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases
    
    def backward(self, grad_output):
        self.d_weights = np.dot(self.input.T, grad_output)
        self.d_biases = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input
    
    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases






#flatten layer : it is used to flatten the output of the convolutional and pooling layers into a 1D vector that can be fed into the fully connected layer

class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(self.input_shape[0], -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
    






#FInally we can create a simple CNN model by stacking the layers together and defining the forward and backward pass of the model   


class CNN :
    def __init__(self):
        self.layers= []
        self.loss_function = None
        self.loss_history = []
        self.accuracy_history = []

    def add(self, layer): #add a layer to the model
        self.layers.append(layer)

    def compile(self, loss): #configure the model for training by specifying the loss function and the optimizer
        if loss == 'cross_entropy':
            self.loss_function = CrossEntropyLoss()
        elif loss == 'mse':
            self.loss_function = MSELoss()
        else:
            raise ValueError("Unsupported loss function")
        
    def forward(self, X): #forward pass through all layers
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad_output): #backward pass through all layers
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def train_step(self, X ,y ,learning_rate):
        # X : batch of images , y : batch of labels
        #forward + loss + backward + update
        predictions = self.forward(X)
        loss = self.loss_function.forward(predictions, y)
        self.loss_history.append(loss)                                                  
        grad_loss = self.loss_function.backword()
        self.backward(grad_loss)

        for layer in self.layers :
            if hasattr(layer, 'update'):
                layer.update(learning_rate)

        return loss , predictions
    

    def fit(self, X_train , y_train , epochs =10 , batch_size =32, learning_rate=0.01 , X_val = None , y_val = None):
        #train the model
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            #shuffle the training data
            indices = np.random.permutation(n_samples)
            x_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            n_batches = 0

            #mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                loss , _ = self.train_step(X_batch, y_batch, learning_rate)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            train_pred = self.predict(X_train)
            train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
            self.accuracy_history.append(train_acc)
            self.accuracy_history.append(train_acc)


            #validate the model
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train Accuracy: {train_acc:.4f} - Val Accuracy: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train Accuracy: {train_acc:.4f}")


    def predict(self, X):
        #make predictions
        preds  = self.forward(X)
        return np.argmax(preds, axis=1)
    
    def predict_proba(self, X):
        #predict probabilities
        return self.forward(X)
    




#Data Preparing 
def prep_mnist_data():
    #a function that loads and prepares the MNIST dataset for training and testing the CNN model
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    #1- Normalize the data : we have already normalized the pixel values to be between 0 and 1 by dividing by 255.0
    #2- Reshape the data : we need to reshape the data to be in the format (n_samples, n_channels, height, width) for the CNN model
    X = X.reshape(-1, 1, 28, 28)

    #convert labels to one-hot encoding
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.shape[0]), y] = 1

    #3- Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    #4- split validation set from training set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    return X_train, y_train, X_val, y_val, X_test, y_test






#Build and Train the CNN :
def build_cnn():
    #build a simple CNN model and train it on the MNIST dataset
    model = CNN()

    #add layers to the model    
    #1- Conv layer 1 : 1 input channel (grayscale) , 8 output channels (filters) , kernel size 3x3
    
    model.add(Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1))
    model.ReLU()
    #2- Max pooling layer : pool size 2x2 , stride 2
    model.add(MaxPool2D(pool_size=2, stride=2))
    #3- Conv layer 2 : 8 input channels , 16 output channels ,
    model.add(Conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1))
    model.ReLU()
    model.add(MaxPool2D(pool_size=2, stride=2))

    #4- Flatten layer
    model.add(Flatten())

    #5- Fully connected layer : 16*7*7 input features , 10 output features (classes)
    model.add(Linear(in_features=16 * 7 * 7, out_features=128))
    model.add(ReLU())
    model.add(Linear(in_features=128, out_features=10))
    model.add(Softmax())

    #compile the model with cross entropy loss
    model.compile(loss='cross_entropy')

    return model





#visualize the training process
def plot_training_history(model):
    #a function that plots the training history of the model
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(model.accuracy_history, label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()






if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("="*70)
    print("BUILDING CNN FROM SCRATCH - MNIST CLASSIFICATION")
    print("="*70)
    
    # Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prep_mnist_data()
    
    # Build model
    print("\n🏗️ Building CNN architecture...")
    model = build_cnn()
    print("✓ Model built successfully!")
    print("\nArchitecture:")
    print("- Conv2D (8 filters, 3x3) → ReLU → MaxPool")
    print("- Conv2D (16 filters, 3x3) → ReLU → MaxPool")
    print("- Flatten → Dense(128) → ReLU → Dense(10) → Softmax")
    
    # Train model
    print("\n Training CNN...")
    print("-" * 60)
    
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        learning_rate=0.01,
        X_val=X_val,
        y_val=y_val,
        verbose=True
    )
    
    # Evaluate on test set
    test_pred = model.predict(X_test)
    test_true = np.argmax(y_test, axis=1)
    test_acc = np.mean(test_pred == test_true)
    
    print("\n" + "="*70)
    print(f" FINAL TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("="*70)
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(model.loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(model.accuracy_history, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize predictions
    plot_training_history(model, X_test, y_test)
    
    # Show sample predictions
    print("\n SAMPLE PREDICTIONS (First 10 test images):")
    print("-" * 60)
    print(f"{'Image #':<8} {'Predicted':<12} {'Actual':<10} {'Correct?'}")
    print("-" * 60)
    
    for i in range(10):
        pred = test_pred[i]
        actual = test_true[i]
        correct = "✓" if pred == actual else "✗"
        print(f"{i+1:<8} {pred:<12} {actual:<10} {correct}")