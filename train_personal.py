
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    WeightedRandomSampler
)
from torchvision import datasets, transforms
from PIL import Image


# ==========================================
# CONFIGURACIÓN
# ==========================================

BASE_DIR = Path(r"E:\Programms\MathVision")
DATA_DIR = BASE_DIR / "data"
USER_DIR = DATA_DIR / "user_digits"
MODEL_DIR = BASE_DIR / "models"

ORIGINAL_MODEL = MODEL_DIR / "mnist_cnn.pth"
PERSONAL_MODEL = MODEL_DIR / "mnist_cnn_personal.pth"

BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 0.0001

random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu")

print("====================================")
print("       MATHVISION - PERSONAL")
print("====================================")
print(f"Dispositivo: {device}")


# ==========================================
# COMPROBACIONES
# ==========================================

if not ORIGINAL_MODEL.exists():
    raise FileNotFoundError(
        f"No encuentro el modelo original: {ORIGINAL_MODEL}"
    )

if not USER_DIR.exists():
    raise FileNotFoundError(
        f"No encuentro los datos personales: {USER_DIR}"
    )

for digit in range(10):
    files = list((USER_DIR / str(digit)).glob("*.png"))

    if len(files) != 5:
        raise ValueError(
            f"El dígito {digit} tiene {len(files)} imágenes. "
            "Se esperaban 5."
        )

print("50 imágenes personales verificadas.")


# ==========================================
# ARQUITECTURA CNN
# ==========================================

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


# ==========================================
# TRANSFORMACIONES
# ==========================================

mnist_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,
        translate=(0.08, 0.08),
        scale=(0.90, 1.10)
    ),
    transforms.ToTensor()
])

user_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.04, 0.04),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor()
])

test_transform = transforms.ToTensor()


# ==========================================
# CARGA DE DATOS
# ==========================================

print("Cargando MNIST...")

mnist_train = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=mnist_transform
)

mnist_test = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=test_transform
)

print("Cargando tus números...")

user_dataset = datasets.ImageFolder(
    root=str(USER_DIR),
    transform=user_transform
)

if len(user_dataset) != 50:
    raise ValueError(
        f"Se han encontrado {len(user_dataset)} imágenes, "
        "pero deberían ser 50."
    )

print(f"MNIST: {len(mnist_train)} imágenes")
print(f"Tus números: {len(user_dataset)} imágenes")


# ==========================================
# COMBINAR DATASETS
# ==========================================

combined_dataset = ConcatDataset([
    mnist_train,
    user_dataset
])

# Damos más peso a tus números para que la red
# los tenga suficientemente en cuenta.

weights = (
    [1.0] * len(mnist_train)
    + [200.0] * len(user_dataset)
)

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(mnist_train),
    replacement=True
)

train_loader = DataLoader(
    combined_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

test_loader = DataLoader(
    mnist_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# CARGAR MODELO ANTERIOR
# ==========================================

print("Cargando CNN original...")

model = MathCNN().to(device)

checkpoint = torch.load(
    ORIGINAL_MODEL,
    map_location=device,
    weights_only=True
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint)

print("Modelo original cargado correctamente.")


# ==========================================
# ENTRENAMIENTO PERSONALIZADO
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nComenzando entrenamiento personalizado...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    average_loss = total_loss / len(train_loader)

    print(f"Época {epoch + 1}/{EPOCHS}")
    print(f"Pérdida: {average_loss:.4f}")
    print(f"Precisión de entrenamiento: {train_accuracy:.2f}%")


# ==========================================
# EVALUACIÓN MNIST
# ==========================================

print("\nEvaluando con MNIST de prueba...")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total

print(f"Precisión MNIST: {test_accuracy:.2f}%")


# ==========================================
# GUARDAR MODELO
# ==========================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

torch.save(
    model.state_dict(),
    PERSONAL_MODEL
)

print("\n====================================")
print("ENTRENAMIENTO FINALIZADO")
print("====================================")
print(f"Modelo guardado en:\n{PERSONAL_MODEL}")
print(f"Precisión MNIST: {test_accuracy:.2f}%")
