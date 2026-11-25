from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn

from PIL import Image, ImageOps
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ==========================================
# CONFIGURACIÓN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "mnist_cnn_personal.pth"

DEVICE = torch.device("cpu")


# ==========================================
# ARQUITECTURA DE LA CNN
# ==========================================

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


# ==========================================
# CARGA DEL MODELO
# ==========================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No encuentro el modelo: {MODEL_PATH}"
        )

    model = MathCNN().to(DEVICE)

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model


model = load_model()


# ==========================================
# PREPROCESAMIENTO DE IMÁGENES
# ==========================================

def preprocess_image(image_bytes):

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("L")

    image = ImageOps.autocontrast(image)

    width, height = image.size

    borders = (
        list(image.crop((0, 0, width, 2)).getdata())
        + list(image.crop((0, height - 2, width, height)).getdata())
        + list(image.crop((0, 0, 2, height)).getdata())
        + list(image.crop((width - 2, 0, width, height)).getdata())
    )

    background = sum(borders) / len(borders)

    if background > 127:
        image = ImageOps.invert(image)

    image = image.point(
        lambda p: 255 if p > 100 else 0
    )

    bbox = image.getbbox()

    if bbox is None:
        raise ValueError("No se han detectado trazos.")

    image = image.crop(bbox)
    image.thumbnail((20, 20))

    canvas = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    tensor = transforms.ToTensor()(canvas)

    return tensor.unsqueeze(0)


# ==========================================
# API DE MATHVISION
# ==========================================

app = FastAPI(
    title="MathVision API",
    description="API de reconocimiento matemático mediante inteligencia artificial.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():
    return {
        "app": "MathVision",
        "status": "online",
        "message": "El servidor está funcionando correctamente."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen."
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise ValueError("La imagen está vacía.")

        tensor = preprocess_image(image_bytes).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        values, indices = torch.topk(probabilities, 3)

        alternatives = []

        for value, index in zip(values, indices):
            alternatives.append({
                "number": int(index.item()),
                "confidence": round(float(value.item()) * 100, 2)
            })

        return {
            "success": True,
            "prediction": alternatives[0]["number"],
            "confidence": alternatives[0]["confidence"],
            "alternatives": alternatives
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
