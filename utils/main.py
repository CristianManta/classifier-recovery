from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imshow(img):
    # Unnormalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Function to get dataset loaders
def get_data_loader(dataset_name='MNIST', batch_size=64, train=True):
    if dataset_name == 'MNIST':
        dataset_class = torchvision.datasets.MNIST
        # Adjust transform to repeat grayscale channel to 3 channels
        transform = Compose([
            Resize(224),
            ToTensor(),
            Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the single channel to 3 channels
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Apply normalization for 3 channels
        ])
    elif dataset_name == 'CIFAR10':
        dataset_class = torchvision.datasets.CIFAR10
        transform = Compose([
            Resize(224),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    dataset = dataset_class(root='./data', train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def display_samples(dataset_name, apply_transform=False):
    dataloader = get_data_loader(dataset_name, apply_transform=apply_transform)
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Show images
    imshow(torchvision.utils.make_grid(images))

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedResNet18, self).__init__()
        original_model = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.fc = nn.Linear(512, num_classes)  # Adapted for CIFAR10 and MNIST

    def forward(self, x): # shape: (batch_size, C, H, W)
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the features
        return self.fc(x)
    
class CustomCNN28(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN28, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x): # shape: (batch_size, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x # shape: (batch_size, num_classes)
    
class GMMMLP(nn.Module):
    def __init__(self):
        super(GMMMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        return self.layers(x)
    
class CustomCNN10(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN10, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Output: (batch_size, 16, 10, 10)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)  # Output: (batch_size, 32, 8, 8)
        self.pool = nn.MaxPool2d(2, 2)  # Output: (batch_size, 32, 4, 4)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train_model(model, dataloader, epochs=10, device='cuda'):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(dataloader, total=len(dataloader))
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%')

        progress_bar.close()
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        # Save model checkpoint
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        print(f'Model saved as model_epoch_{epoch+1}.pth')

def infer(model, dataloader):
    model.eval()  # Set model to evaluation mode
    outputs = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)  # Move data to GPU
            output = model(inputs)
            outputs.append(output.cpu())  # Move the outputs back to CPU for further processing
    return torch.cat(outputs)