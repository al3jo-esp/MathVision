
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk, ImageOps


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "mis_numeros.png"
OUTPUT_DIR = BASE_DIR / "data" / "user_digits"

NUM_DIGITS = 10
SAMPLES_PER_DIGIT = 5

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(
        f"No se encuentra la imagen: {IMAGE_PATH}"
    )

original = Image.open(IMAGE_PATH).convert("L")

# Ajustar imagen al tamaño de la ventana.
scale = min(
    950 / original.width,
    520 / original.height,
    1
)

display_size = (
    int(original.width * scale),
    int(original.height * scale)
)

display_image = original.resize(
    display_size,
    Image.Resampling.LANCZOS
)


# --------------------------------------------------
# VENTANA PRINCIPAL
# --------------------------------------------------

root = tk.Tk()
root.title("MathVision - Selector de dígitos")
root.resizable(False, False)

# Crear la imagen después de crear la ventana.
image_tk = ImageTk.PhotoImage(
    display_image,
    master=root
)


# --------------------------------------------------
# ESTADO
# --------------------------------------------------

current_digit = 0
current_sample = 1

start = None
rectangle = None


# --------------------------------------------------
# INTERFAZ
# --------------------------------------------------

label = tk.Label(
    root,
    font=("Arial", 16, "bold")
)
label.pack(pady=10)

canvas = tk.Canvas(
    root,
    width=display_size[0],
    height=display_size[1],
    cursor="cross"
)

canvas.pack()

canvas.create_image(
    0,
    0,
    anchor="nw",
    image=image_tk
)


# --------------------------------------------------
# FUNCIONES
# --------------------------------------------------

def actualizar_texto():
    label.config(
        text=(
            f"Dígito: {current_digit}    "
            f"Muestra: {current_sample}/{SAMPLES_PER_DIGIT}"
        )
    )


def comenzar(event):
    global start, rectangle

    start = (event.x, event.y)

    if rectangle is not None:
        canvas.delete(rectangle)

    rectangle = canvas.create_rectangle(
        event.x,
        event.y,
        event.x,
        event.y,
        outline="red",
        width=3
    )


def mover(event):
    if start is not None and rectangle is not None:
        canvas.coords(
            rectangle,
            start[0],
            start[1],
            event.x,
            event.y
        )


def preparar_digito(recorte):

    # Detectar trazos oscuros.
    mascara = recorte.point(
        lambda pixel: 255 if pixel < 180 else 0
    )

    caja = mascara.getbbox()

    if caja is None:
        return None

    # Recortar y convertir a fondo negro con trazo blanco.
    imagen = ImageOps.invert(
        recorte.crop(caja)
    )

    # Mantener proporciones.
    imagen.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Crear lienzo de 28x28.
    lienzo = Image.new("L", (28, 28), 0)

    x = (28 - imagen.width) // 2
    y = (28 - imagen.height) // 2

    lienzo.paste(imagen, (x, y))

    return lienzo


def guardar():

    global current_digit
    global current_sample
    global start
    global rectangle

    if start is None or rectangle is None:
        messagebox.showwarning(
            "Selecciona un número",
            "Primero rodea un número con el ratón."
        )
        return

    x1, y1, x2, y2 = canvas.coords(rectangle)

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    # Convertir coordenadas de pantalla a coordenadas originales.
    x1 = max(0, min(original.width, int(x1 / scale)))
    x2 = max(0, min(original.width, int(x2 / scale)))
    y1 = max(0, min(original.height, int(y1 / scale)))
    y2 = max(0, min(original.height, int(y2 / scale)))

    if x2 - x1 < 5 or y2 - y1 < 5:
        messagebox.showwarning(
            "Selección demasiado pequeña",
            "Selecciona el número completo."
        )
        return

    recorte = original.crop((x1, y1, x2, y2))

    digito = preparar_digito(recorte)

    if digito is None:
        messagebox.showwarning(
            "No se ha detectado ningún trazo",
            "Vuelve a seleccionar el número."
        )
        return

    # Guardar en la carpeta correspondiente.
    carpeta = OUTPUT_DIR / str(current_digit)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta = carpeta / (
        f"{current_digit}_{current_sample}.png"
    )

    digito.save(ruta)

    print(f"Guardado: {ruta}")

    # Avanzar a la siguiente muestra.
    current_sample += 1

    if current_sample > SAMPLES_PER_DIGIT:
        current_digit += 1
        current_sample = 1

    if current_digit >= NUM_DIGITS:
        messagebox.showinfo(
            "Proceso terminado",
            "¡Los 50 números se han guardado correctamente!"
        )
        root.destroy()
        return

    start = None
    rectangle = None

    actualizar_texto()


# --------------------------------------------------
# CONTROLES
# --------------------------------------------------

canvas.bind("<Button-1>", comenzar)
canvas.bind("<B1-Motion>", mover)

boton = tk.Button(
    root,
    text="Guardar número seleccionado",
    font=("Arial", 12),
    command=guardar
)

boton.pack(pady=10)

instrucciones = tk.Label(
    root,
    text=(
        "Rodea el número completo con el ratón y pulsa "
        "'Guardar número seleccionado'.\n"
        "Selecciona los 5 ejemplos de cada dígito "
        "de arriba a abajo."
    ),
    wraplength=950,
    justify="center"
)

instrucciones.pack(pady=10)

actualizar_texto()

root.mainloop()
