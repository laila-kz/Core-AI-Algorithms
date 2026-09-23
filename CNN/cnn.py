#build a tiny CNN using PyTorch
# INPUT: [1, 1, 8, 8]  (batch=1, channels=1, height=8, width=8)
#     ↓
# CONV1: [1, 2, 8, 8]  (2 feature maps, same size due to padding)
#     ↓
# RELU:  [1, 2, 8, 8]  (same shape, just non-linear)
#     ↓
# POOL:  [1, 2, 4, 4]  (downsampled by 2x)
#     ↓
# CONV2: [1, 4, 4, 4]  (4 feature maps)
#     ↓
# RELU:  [1, 4, 4, 4]  (same shape)
#     ↓
# POOL:  [1, 4, 2, 2]  (downsampled again)
#     ↓
# FLATTEN: [1, 16]     (4*2*2 = 16 features)
#     ↓
# FC1:    [1, 8]       (8 hidden neurons)
#     ↓
# FC2:    [1, 2]       (2 class scores)
#     ↓
# SOFTMAX: [1, 2]      (probabilities for each class)



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.functional as F


#Step 1: create a tiny dataset of 8x8 images with two classes (0 and 1)
def create_dataset():
    images =[]
    labels =[]

    for i in range(10):
        img = np.zeros((8, 8))

        for x in range(8):
            for y in range(8):
                if (x + y) % 2 == 0:
                    img[x, y] = 1  # Class 0: checkerboard pattern
                else:
                    img[x, y] = 0  # Class 1: inverse checkerboard pattern
        images.append(img)
        labels.append(0)  # Class 0 = checkerboard
    for i in range(10):
        img = np.zeros((8, 8))

        for x in range(8):
            for y in range(8):
                if (x + y) % 2 == 0:
                    img[x, y] = 0  # Class 1: inverse checkerboard pattern
                else:
                    img[x, y] = 1  # Class 0: checkerboard pattern
        images.append(img)
        labels.append(1)  # Class 1 = inverse checkerboard

    images = np.array(images).reshape(-1, 1, 8, 8)  # Reshape to (batch_size, channels, height, width)
    labels = np.array(labels)
    return images, labels


#test 
fig , axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(create_dataset()[0][i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Label: {create_dataset()[1][i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()


#Step 2: Define the CNN architecture
class TinyCNN(nn.Module):
    def _init_(self):
        super(TinyCNN, self)._init_()

        #first conv layer : input channels=1, output channels=2, kernel size=3x3, padding=1
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)

        #second conv layer : input channels=2, output channels=4, kernel size=3x3, padding=1
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, padding=1)

        #polling layer : kernel size=2x2, stride=2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #fully connected layer : input features=4*2*2=16, output features=8
        self.fc1 = nn.Linear(4 * 2 * 2, 8)
        self.fc2 = nn.Linear(8, 2)  # output features=2 (for 2 classes)


    def forward(self, x):
        #thi shows how data flow through the network
        print(f"\n Input shape: {x.shape}")
        
        # FIRST BLOCK: Conv -> ReLU -> Pool
        x = self.conv1(x)
        print(f"After conv1: {x.shape}  [batch, 2 channels, 8x8]")
        x = F.relu(x)
        print(f"After ReLU: {x.shape}  [same shape, just non-linear]")
        x = self.pool(x)
        print(f"After pool1: {x.shape}  [batch, 2 channels, 4x4]")
        
        # SECOND BLOCK: Conv -> ReLU -> Pool
        x = self.conv2(x)
        print(f"After conv2: {x.shape}  [batch, 4 channels, 4x4]")
        x = F.relu(x)
        print(f"After ReLU: {x.shape}")
        x = self.pool(x)
        print(f"After pool2: {x.shape}  [batch, 4 channels, 2x2]")
        
        # FLATTEN
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch
        print(f"After flatten: {x.shape}  [batch, 4*2*2 = 16 features]")
        
        # FULLY CONNECTED LAYERS
        x = self.fc1(x)
        print(f"After fc1: {x.shape}  [batch, 8 features]")
        x = F.relu(x)
        x = self.fc2(x)
        print(f"After fc2: {x.shape}  [batch, 2 classes]")
        
        # Apply softmax to get probabilities
        # (usually done in loss function, but shown here for clarity)
        probs = F.softmax(x, dim=1)
        print(f"After softmax: {probs.shape}  [probabilities for each class]")
        
        return x  # Return raw logits for training



#Create model instance and test with a sample input
model = TinyCNN()
print("Model architecture:")
print(model)



#How the data actually flows through the network
print("\n" + "="*70)
print("🔍 TRACING DATA FLOW - ONE IMAGE THROUGH THE NETWORK")
print("="*70)

# Take one image
single_image = images[0:1]  # Keep batch dimension: (1, 1, 8, 8)
print(f"Single image shape: {single_image.shape}")

# Pass through network (with print statements)
with torch.no_grad():  # No gradients needed for this demo
    output = model(single_image)

print(f"\n📤 Final output (logits): {output}")
print(f"Predicted class: {torch.argmax(output, dim=1).item()}")
print(f"Actual class: {labels[0].item()}")




#visualise each layer's output for the first image
def visualize_layer_outputs():
    model.eval()
    image = images[0:1]  # First image
    
    # Hook to capture intermediate outputs
    outputs = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            outputs[name] = output.detach()
        return hook
    
    # Register hooks
    model.conv1.register_forward_hook(hook_fn('conv1'))
    model.conv2.register_forward_hook(hook_fn('conv2'))
    model.pool.register_forward_hook(hook_fn('pool'))
    
    # Forward pass
    with torch.no_grad():
        _ = model(image)
    
    # Visualize
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    # Original image
    axes[0, 0].imshow(image[0, 0].numpy(), cmap='gray')
    axes[0, 0].set_title('Original Input')
    axes[0, 0].axis('off')
    
    # Conv1 outputs (2 channels)
    conv1_out = outputs['conv1'][0]  # First image in batch
    for i in range(2):
        axes[1, i].imshow(conv1_out[i].numpy(), cmap='viridis')
        axes[1, i].set_title(f'Conv1 - Channel {i}')
        axes[1, i].axis('off')
    
    # After ReLU + Pool
    pool_out = outputs['pool'][0]
    axes[2, 0].imshow(pool_out[0].numpy(), cmap='viridis')
    axes[2, 0].set_title('After Pool - Channel 0')
    axes[2, 0].axis('off')
    axes[2, 1].imshow(pool_out[1].numpy(), cmap='viridis')
    axes[2, 1].set_title('After Pool - Channel 1')
    axes[2, 1].axis('off')
    
    # Conv2 outputs (4 channels)
    conv2_out = outputs['conv2'][0]
    for i in range(4):
        axes[1, i].imshow(conv2_out[i].numpy(), cmap='plasma')
        axes[1, i].set_title(f'Conv2 - Channel {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_layer_outputs()



#Step 3: Train the CNN on the dataset
print("\n" + "="*70)
print("🚀 TRAINING THE TINY CNN")
print("="*70)

# Recreate model (fresh start)
model = TinyCNN()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 50
loss_history = []
accuracy_history = []

for epoch in range(n_epochs):
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == labels).sum().item() / len(labels)
    
    loss_history.append(loss.item())
    accuracy_history.append(accuracy)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}')


#Plot training loss and accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(loss_history, 'b-', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(accuracy_history, 'g-', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()




#test the trained model on the new dataset (just to see predictions)
print("\n" + "="*70)
print("🎯 TESTING ON NEW IMAGES")
print("="*70)

# Create test images (slightly different)
test_images = []
test_labels = []

# Circle with noise
img = np.zeros((8, 8))
for x in range(8):
    for y in range(8):
        if (x-3.5)**2 + (y-3.5)**2 < 7:
            img[x, y] = 1
test_images.append(img)
test_labels.append(0)

# Square slightly shifted
img = np.zeros((8, 8))
img[3:7, 3:7] = 1
test_images.append(img)
test_labels.append(1)

# Convert to tensors
test_images = torch.FloatTensor(test_images).unsqueeze(1)
test_labels = torch.LongTensor(test_labels)

# Predict
model.eval()
with torch.no_grad():
    outputs = model(test_images)
    probabilities = F.softmax(outputs, dim=1)
    predictions = torch.argmax(outputs, dim=1)

# Show results
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for i in range(2):
    axes[i].imshow(test_images[i, 0].numpy(), cmap='gray')
    pred_class = "Circle" if predictions[i] == 0 else "Square"
    true_class = "Circle" if test_labels[i] == 0 else "Square"
    color = 'green' if predictions[i] == test_labels[i] else 'red'
    axes[i].set_title(f'Pred: {pred_class}\nTrue: {true_class}', color=color)
    axes[i].axis('off')
    
    print(f"\nImage {i+1}:")
    print(f"  True label: {true_class}")
    print(f"  Predicted: {pred_class}")
    print(f"  Probabilities: Circle={probabilities[i,0]:.3f}, Square={probabilities[i,1]:.3f}")

plt.tight_layout()
plt.show()





#visualize the learned filters of each convolutional layer
print("\n" + "="*70)
print("🔬 VISUALIZING LEARNED FILTERS")
print("="*70)

# Get conv1 weights
weights = model.conv1.weight.data.numpy()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for i in range(2):
    axes[i].imshow(weights[i, 0], cmap='coolwarm', vmin=-1, vmax=1)
    axes[i].set_title(f'Conv1 - Filter {i+1}')
    axes[i].axis('off')
    plt.colorbar(axes[i].imshow(weights[i, 0], cmap='coolwarm'), ax=axes[i])

plt.suptitle('What the Network Learned - First Layer Filters', fontsize=14)
plt.tight_layout()
plt.show()

print("\n📝 WHAT YOU JUST LEARNED:")
print("• How data shape changes through each layer")
print("• What convolution, ReLU, and pooling actually do")
print("• How gradients flow backward during training")
print("• How the network learns to detect patterns")
print("• How to make predictions on new images")
