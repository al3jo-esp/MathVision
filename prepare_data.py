
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "mis_numeros.png"
OUTPUT_DIR = BASE_DIR / "data" / "user_digits"

# Separadores verticales de tu imagen.
X_BOUNDS = [0, 142, 346, 548, 705, 867,
            991, 1127, 1274, 1431, 1889]

# Límites horizontales adaptados a los dibujos.
Y_BOUNDS = [0, 110, 225, 350, 470, 610]


def preparar_digito(recorte):
    gris = recorte.convert("L")

    mascara = gris.point(
        lambda p: 255 if p < 180 else 0
    )

    caja = mascara.getbbox()

    if caja is None:
        return None

    imagen = ImageOps.invert(gris.crop(caja))
    imagen.thumbnail((20, 20), Image.Resampling.LANCZOS)

    lienzo = Image.new("L", (28, 28), 0)

    x = (28 - imagen.width) // 2
    y = (28 - imagen.height) // 2

    lienzo.paste(imagen, (x, y))

    return lienzo


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH)

    original = Image.open(IMAGE_PATH).convert("L")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vista = Image.new("RGB", (1000, 500), "white")
    dibujo = ImageDraw.Draw(vista)

    encontrados = 0
    vacios = []

    for numero in range(10):
        carpeta = OUTPUT_DIR / str(numero)
        carpeta.mkdir(parents=True, exist_ok=True)

        for fila in range(5):
            x1 = X_BOUNDS[numero] + 8
            x2 = X_BOUNDS[numero + 1] - 8

            y1 = Y_BOUNDS[fila] + 2
            y2 = Y_BOUNDS[fila + 1] - 2

            recorte = original.crop((x1, y1, x2, y2))
            digito = preparar_digito(recorte)

            if digito is None:
                vacios.append((numero, fila + 1))
                continue

            ruta = carpeta / f"{numero}_{fila + 1}.png"
            digito.save(ruta)

            grande = digito.resize((80, 80))
            vista.paste(
                grande.convert("RGB"),
                (numero * 100 + 10, fila * 100 + 5)
            )

            dibujo.text(
                (numero * 100 + 42, fila * 100 + 85),
                str(numero),
                fill="red"
            )

            encontrados += 1

    vista.save(BASE_DIR / "mis_numeros_recortados.png")

    print(f"\nNúmeros encontrados: {encontrados}")

    if vacios:
        print("Casillas vacías:")
        for numero, fila in vacios:
            print(f"  Dígito {numero}, fila {fila}")

    print("\nVista previa guardada en:")
    print(BASE_DIR / "mis_numeros_recortados.png")


if __name__ == "__main__":
    main()
