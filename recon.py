import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# Datos: imágenes 28x28 en escala de grises, normalizadas
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
test_ds  = datasets.MNIST("data", train=False, download=True, transform=tf)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl  = DataLoader(test_ds, batch_size=1000)

# Red: 784 -> 128 -> 64 -> 10
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10),
).to(device)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(test_ds)

for epoch in range(5):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    print(f"Época {epoch+1}: precisión en test = {evaluate():.4f}")

import os

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/mnist_model.pth")

print("Modelo guardado correctamente.")