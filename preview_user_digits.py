
from pathlib import Path
from PIL import Image, ImageDraw

BASE_DIR = Path(r"E:\Programms\MathVision")
DATA_DIR = BASE_DIR / "data" / "user_digits"
OUTPUT = BASE_DIR / "vista_previa_digitos.png"

CELL_WIDTH = 120
CELL_HEIGHT = 120

print("Iniciando vista previa...")

canvas = Image.new("RGB", (10 * CELL_WIDTH, 5 * CELL_HEIGHT), "white")
draw = ImageDraw.Draw(canvas)

for digit in range(10):
    folder = DATA_DIR / str(digit)
    files = sorted(folder.glob("*.png"))

    if len(files) != 5:
        raise ValueError(
            f"La carpeta {digit} tiene {len(files)} imágenes."
        )

    for sample, file in enumerate(files):
        image = Image.open(file).convert("L")
        image.thumbnail((75, 75))

        x = digit * CELL_WIDTH
        y = sample * CELL_HEIGHT

        draw.text((x + 5, y + 5), f"Digito {digit}", fill="black")
        canvas.paste(image.convert("RGB"), (x + 20, y + 30))

canvas.save(OUTPUT)

print("Vista previa creada correctamente.")
print(f"Archivo: {OUTPUT}")
