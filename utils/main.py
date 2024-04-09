import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim

def get_data_loader(dataset_name='MNIST', batch_size=64, train=True):
    if dataset_name == 'MNIST':
        dataset_class = torchvision.datasets.MNIST
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(3),  # Convert grayscale to 3-channel
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalizing for 3 channels
        ])
    elif dataset_name == 'CIFAR10':
        dataset_class = torchvision.datasets.CIFAR10
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    dataset = dataset_class(root='./data', train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedResNet18, self).__init__()
        original_model = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])  # Exclude the last fc layer
        self.fc = nn.Linear(512, num_classes)  # Adapted for CIFAR10 and MNIST

    def forward(self, x, before_softmax=False):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the features
        if before_softmax:
            return x
        x = self.fc(x)
        return x

def train_model(model, dataloader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def infer(model, dataloader, before_softmax=False):
    model.eval()
    outputs = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            output = model(inputs, before_softmax=before_softmax)
            outputs.append(output)
    return torch.cat(outputs)