
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Configuración
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 0.001


# Transformaciones para aumentar la variedad de escritura
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=12,
        translate=(0.10, 0.10),
        scale=(0.85, 1.15),
        shear=8
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# Descargar y cargar MNIST
train_dataset = datasets.MNIST(
    DATA_DIR,
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.MNIST(
    DATA_DIR,
    train=False,
    download=True,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)


# Red neuronal convolucional
class MathCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Inicializar modelo
model = MathCNN().to(DEVICE)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Evaluación
def evaluar():
    model.eval()

    correctos = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(images).argmax(1)

            correctos += (predictions == labels).sum().item()
            total += labels.size(0)

    return correctos / total


# Entrenamiento
print(f"Dispositivo: {DEVICE}")
print("Comenzando entrenamiento...\n")

for epoch in range(EPOCHS):

    model.train()
    perdida_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        perdida_total += loss.item()

    precision = evaluar()

    print(
        f"Época {epoch + 1}/{EPOCHS} | "
        f"Pérdida: {perdida_total / len(train_loader):.4f} | "
        f"Precisión: {precision * 100:.2f}%"
    )


# Guardar modelo
MODEL_DIR.mkdir(exist_ok=True)

model_path = MODEL_DIR / "mnist_cnn.pth"

torch.save(model.state_dict(), model_path)

print("\nEntrenamiento completado.")
print(f"Modelo guardado en: {model_path}")
