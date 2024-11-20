# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import chess
import chess.svg
import matplotlib.pyplot as plt
from IPython.display import display, SVG
from wand.image import Image as WandImage
from tkinter import Tk, Label, Entry, Button, Toplevel, StringVar
from PIL import Image, ImageTk
import io
import svgwrite

# Configuración de rutas
model_dir = 'Proyectos/Proyectos/ML_CHESS/models/'
move_suggester_path = model_dir + 'move_suggester.h5'
tokenizer_path = model_dir + 'tokenizer.pkl'

def prepare_and_save_tokenizer(moves, save_path):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(moves)
    with open(save_path, 'wb') as f:
        joblib.dump(tokenizer, f)
    return tokenizer

# Función para convertir el tablero a imagen

def board_to_image(board):
    svg_data = chess.svg.board(board=board)
    # Convertir SVG a PNG usando wand
    with WandImage(blob=svg_data.encode('utf-8'), format="svg") as image:
        image.format = 'png'
        png_data = image.make_blob()
    
    # Convertir el PNG en un objeto de imagen de PIL
    image = Image.open(io.BytesIO(png_data))
    return ImageTk.PhotoImage(image)

# Función para predecir el siguiente movimiento
def predict_next_move(model, tokenizer, input_text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')
    predicted = model.predict(sequence)
    next_move = tokenizer.index_word[predicted.argmax()]
    return next_move

# Cargar modelo de sugerencia de movimientos
print("Cargando modelos...")
try:
    move_suggester = load_model(move_suggester_path)
except FileNotFoundError:
    print(f"No se encontró el archivo de modelo en {move_suggester_path}")
    exit()

# Preparar y guardar tokenizador si no existe
try:
    with open(tokenizer_path, 'rb') as f:
        tokenizer = joblib.load(f)
except FileNotFoundError:
    print("No se encontró el archivo de tokenizador. Creando uno nuevo...")
    data = pd.read_csv('Proyectos/Proyectos/ML_CHESS/DATA/raw/games.csv')  # Ruta de los datos
    moves = data['moves'].values
    tokenizer = prepare_and_save_tokenizer(moves, tokenizer_path)
    print(f"Tokenizador guardado en {tokenizer_path}")

# Definir longitud máxima de secuencia
max_sequence_len = 20  # Ajustar según el entrenamiento

# Crear la ventana principal de la interfaz
root = Tk()
root.title("Sugerencia de Movimiento de Ajedrez")

# Variables de entrada y salida
input_var = StringVar()

# Función para mostrar el tablero y sugerir el siguiente movimiento
def show_board():
    moves = input_var.get()
    board = chess.Board()
    for move in moves.split():
        try:
            board.push_san(move)
        except ValueError:
            Label(popup, text=f"Movimiento inválido: {move}", fg="red").pack()
            return

    next_move = predict_next_move(move_suggester, tokenizer, moves, max_sequence_len)
    moves_with_suggestion = moves + ' ' + next_move

    # Actualizar tablero con el movimiento sugerido
    board.push_san(next_move)
    board_image = board_to_image(board)

    # Mostrar la imagen en el popup
    img_label.config(image=board_image)
    img_label.image = board_image

    Label(popup, text=f"Siguiente movimiento sugerido: {next_move}").pack()

# Crear un popup con la interfaz
popup = Toplevel(root)
popup.title("Tablero de Ajedrez")

# Entrada de texto
Label(popup, text="Introduce movimientos en formato PGN:").pack()
entry = Entry(popup, textvariable=input_var, width=50)
entry.pack()

# Botón para mostrar el tablero
Button(popup, text="Mostrar Tablero y Sugerir Movimiento", command=show_board).pack()

# Label de la imagen del tablero
img_label = Label(popup)
img_label.pack()

root.mainloop()

print("Programa finalizado.")