# Assignment: Object Classification on Caltech-101 using ResNet18

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# =====================
# 1. Data Preparation
# =====================
# Set path to your dataset folder
# Folder structure:
# caltech-101-img/
# ├── accordion/
# ├── airplanes/
# ├── anchor/
# └── ...
data_dir = "./caltech-101-img"

# Train/Validation transforms
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset using ImageFolder (expects subfolders as class names)
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)

# Split dataset into train (80%) and validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
trainset, valset = random_split(full_dataset, [train_size, val_size])

# Use different transforms for validation
valset.dataset.transform = transform_val

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

print(f"Total classes: {len(full_dataset.classes)}")
print(f"Training samples: {len(trainset)}, Validation samples: {len(valset)}")

# =====================
# 2. Load Pretrained ResNet18
# =====================
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last residual block for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final classifier
num_features = model.fc.in_features
num_classes = len(full_dataset.classes)

model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes),
    nn.LogSoftmax(dim=1)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# =====================
# 3. Loss & Optimizer
# =====================
criterion = nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)

# =====================
# 4. Training Loop
# =====================
train_loss_history, val_loss_history, val_acc_history = [], [], []
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {running_loss/len(trainloader):.4f}, "
          f"Val Loss: {val_loss/len(valloader):.4f}, "
          f"Val Acc: {acc:.2f}%")

    train_loss_history.append(running_loss/len(trainloader))
    val_loss_history.append(val_loss/len(valloader))
    val_acc_history.append(acc)

# =====================
# 5. Plot Loss & Accuracy
# =====================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")

plt.subplot(1,2,2)
plt.plot(val_acc_history, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Validation Accuracy")

plt.show()
