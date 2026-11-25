
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms


BASE_DIR = Path(r"E:\Programms\MathVision")
MODEL_PATH = BASE_DIR / "models" / "mnist_cnn_personal.pth"
DEVICE = torch.device("cpu")


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


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No encuentro el modelo: {MODEL_PATH}")

    model = MathCNN().to(DEVICE)

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)

    # Comprobar el color del fondo mediante los bordes.
    width, height = image.size

    border_pixels = (
        list(image.crop((0, 0, width, 2)).getdata())
        + list(image.crop((0, height - 2, width, height)).getdata())
        + list(image.crop((0, 0, 2, height)).getdata())
        + list(image.crop((width - 2, 0, width, height)).getdata())
    )

    background = sum(border_pixels) / len(border_pixels)

    # Queremos siempre fondo negro y trazos blancos.
    if background > 127:
        image = ImageOps.invert(image)

    # Separar el número del fondo.
    image = image.point(lambda p: 255 if p > 100 else 0)

    bbox = image.getbbox()

    if bbox is None:
        raise ValueError("No se ha detectado ningún trazo.")

    image = image.crop(bbox)
    image.thumbnail((20, 20))

    canvas = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    tensor = transforms.ToTensor()(canvas)

    return tensor.unsqueeze(0)


def predict(model, image_path):
    image = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]

    values, indices = torch.topk(probabilities, 3)

    print(f"\nImagen: {Path(image_path).name}")
    print(f"Predicción: {indices[0].item()}")
    print(f"Confianza: {values[0].item() * 100:.2f}%")
    print("Alternativas:")

    for value, index in zip(values, indices):
        print(f"  {index.item()}: {value.item() * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Reconocimiento de dígitos MathVision"
    )

    parser.add_argument(
        "images",
        nargs="+",
        help="Imágenes que queremos reconocer"
    )

    args = parser.parse_args()

    model = load_model()

    print("MathVision - Modelo personalizado")
    print("Modelo cargado correctamente.")

    for image_path in args.images:
        path = Path(image_path)

        if not path.exists():
            print(f"\nNo se encuentra: {path}")
            continue

        try:
            predict(model, path)
        except Exception as error:
            print(f"\nError procesando {path.name}: {error}")


if __name__ == "__main__":
    main()
