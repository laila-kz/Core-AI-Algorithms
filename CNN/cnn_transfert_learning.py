# Take a model trained on a big task → adapt it to your smaller task instead of starting from zero

# Take a pre-trained model
# Keep the early layers
# Replace the last layer
# Train only the new layer (or fine-tune some layers)


# Instead of learning:
# f(x; θ) from scratch
# We reuse θ_pretrained and optimize only small subset θ_new.

#this reduces: overfitting , required training data , training time



import torch 
import torchvision.models as models
import torch.nn as nn

#load the pretrained model
model = models.resnet18(pretrained=True)

#freeze the early layers
for param in model.parameters():
    param.requires_grad = False

#replace the last layer
model.fc = nn.Linear(model.fc.in_features, 10)  # Assuming we have 10 classes

#train only the new layer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)





# You adapt a model by answering 3 questions:

# 1️⃣ What is my task?
# 2️⃣ What does the pretrained model output?
# 3️⃣ How different is my data from the original data?









#--------------------------------------
# small practical exemple :
#---------------------------------------

# Dataset
# ↓
# Preprocess
# ↓
# Load EfficientNet (pretrained)
# ↓
# Replace classifier
# ↓
# Freeze layers
# ↓
# Train



#EfficientNet was pretrained on, ImageNet already contains cats and dogs.

#So we will work on a Transfert learning cnn for a binary classification task (cats vs dogs) using EfficientNet as the base model.

#minimal code for transfert learning with EfficientNet on cats vs dogs dataset (Data set can be found in kaggle)

import torch 
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


#Preprocessing :resizing, converting to tensor , and Normlization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#Load the dataset
train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
val_dataset = datasets.ImageFolder(root='dataset/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


#Load the pretrained EfficientNet
model = models.efficientnet_b0(pretrained=True)

#freeze the feature extractor layers
for params in model.features.parameters():
    params.requires_grad = False


#replace the classifier
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features , 2) # Assuming we have 2 classes (cats and dogs)


#Training + set up 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

#Training loop
for epoch in range(3):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')



# you can even export the model as onnx format for deployment
dummpy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(model, dummpy_input, "efficientnet_cats_dogs.onnx", input_names=['input'], output_names=['output'], opset_version=11)

