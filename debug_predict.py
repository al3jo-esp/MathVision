
from pathlib import Path
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms

BASE = Path(r"E:\Programms\MathVision")
MODEL = BASE / "models" / "mnist_cnn_personal.pth"
OUTPUT = BASE / "debug"


class MathCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
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
        return self.classifier(self.features(x))


def preprocess(path):
    image = Image.open(path).convert("L")

    print(f"\nImagen: {path.name}")
    print(f"Dimensiones: {image.size}")

    image = ImageOps.autocontrast(image)

    width, height = image.size

    borders = (
        list(image.crop((0, 0, width, 2)).getdata())
        + list(image.crop((0, height - 2, width, height)).getdata())
        + list(image.crop((0, 0, 2, height)).getdata())
        + list(image.crop((width - 2, 0, width, height)).getdata())
    )

    background = sum(borders) / len(borders)

    print(f"Brillo de los bordes: {background:.2f}")

    if background > 127:
        image = ImageOps.invert(image)

    image = image.point(lambda p: 255 if p > 100 else 0)

    bbox = image.getbbox()

    print(f"Área detectada: {bbox}")

    if bbox is None:
        raise ValueError("No se detectan trazos.")

    image = image.crop(bbox)
    image.thumbnail((20, 20))

    canvas = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    return canvas


def main():

    print("================================")
    print("MATHVISION - DIAGNÓSTICO")
    print("================================")

    OUTPUT.mkdir(exist_ok=True)

    if not MODEL.exists():
        raise FileNotFoundError(f"No encuentro el modelo: {MODEL}")

    model = MathCNN()

    model.load_state_dict(
        torch.load(
            MODEL,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    print("Modelo cargado correctamente.")

    files = [
        "prueba0.png",
        "prueba1.png",
        "prueba2.png",
        "prueba3.png",
        "prueba4.png",
        "prueba5.png",
        "prueba6.png",
        "prueba7.png",
        "prueba8.png",
        "prueba9.png"
    ]

    for filename in files:

        path = BASE / filename

        if not path.exists():
            print(f"No encuentro: {filename}")
            continue

        try:
            image = preprocess(path)

            debug_path = OUTPUT / f"procesada_{filename}"
            image.save(debug_path)

            tensor = transforms.ToTensor()(image).unsqueeze(0)

            with torch.no_grad():
                probabilities = torch.softmax(
                    model(tensor), dim=1
                )[0]

            values, indices = torch.topk(probabilities, 3)

            print("\nPredicciones:")

            for value, index in zip(values, indices):
                print(
                    f"Número {index.item()}: "
                    f"{value.item() * 100:.2f}%"
                )

            print(f"Imagen procesada: {debug_path}")

        except Exception as error:
            print(f"Error con {filename}: {error}")

    print("\nDiagnóstico terminado.")


if __name__ == "__main__":
    main()
