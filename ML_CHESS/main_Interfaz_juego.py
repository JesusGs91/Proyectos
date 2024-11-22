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
import os
import threading
import time
os.environ['MAGICK_HOME'] = '/opt/homebrew'
from wand.image import Image as WandImage
from tkinter import Tk, Label, Entry, Button, Toplevel, StringVar
from PIL import Image, ImageTk
import io
from tkinter.filedialog import asksaveasfilename


# Configuración de rutas
model_dir = 'models/'
move_suggester_path = model_dir + 'move_suggester.h5'
tokenizer_path = model_dir + 'tokenizer.pkl'

# Variable global para el historial de movimientos
move_history = []

 #Crear un popup para mostrar el historial de movimientos
def create_history_popup():
    """Crea una ventana para mostrar el historial de movimientos."""
    history_popup = Toplevel(root)
    history_popup.title("Historial de Movimientos")

    # Crear una tabla para movimientos
    Label(history_popup, text="Turno", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=10, pady=5)
    Label(history_popup, text="Tú", font=("Arial", 12, "bold")).grid(row=0, column=1, padx=10, pady=5)
    Label(history_popup, text="Máquina", font=("Arial", 12, "bold")).grid(row=0, column=2, padx=10, pady=5)

    for idx, (user_move, machine_move) in enumerate(move_history):
        Label(history_popup, text=f"{idx + 1}", font=("Arial", 10)).grid(row=idx + 1, column=0, padx=10, pady=5)
        Label(history_popup, text=user_move, font=("Arial", 10)).grid(row=idx + 1, column=1, padx=10, pady=5)
        Label(history_popup, text=machine_move, font=("Arial", 10)).grid(row=idx + 1, column=2, padx=10, pady=5)

    # Botón para exportar a archivo de texto
    Button(history_popup, text="Exportar a TXT", command=export_history_to_txt).grid(row=len(move_history) + 1, column=0, columnspan=3, pady=10)

# Exportar historial a un archivo de texto
def export_history_to_txt():
    """Exporta el historial de movimientos a un archivo de texto con una ubicación seleccionada por el usuario."""
    file_path = asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Archivos de texto", "*.txt")],
        title="Guardar historial como..."
    )
    
    if not file_path: 
        return

    try:
        with open(file_path, "w") as f:
            f.write("Historial de Movimientos\n")
            f.write("Turno\tTú\tMáquina\n")
            for idx, (user_move, machine_move) in enumerate(move_history):
                f.write(f"{idx + 1}\t{user_move}\t{machine_move or '-'}\n")
        print(f"Historial exportado correctamente a {file_path}")
    except Exception as e:
        print(f"Error al guardar el historial: {str(e)}")

def prepare_and_save_tokenizer(moves, save_path):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(moves)
    with open(save_path, 'wb') as f:
        joblib.dump(tokenizer, f)
    return tokenizer

# Función para convertir el tablero a imagen
def board_to_image(board):
    svg_data = chess.svg.board(board=board)
    # Convertir SVG a PNG
    with WandImage(blob=svg_data.encode('utf-8'), format="svg") as image:
        image.format = 'png'
        png_data = image.make_blob()
    
    # Convertir el PNG en un objeto de imagen
    image = Image.open(io.BytesIO(png_data))
    return ImageTk.PhotoImage(image)

# Función para predecir el siguiente movimiento
def predict_next_move(model, tokenizer, input_text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')
    predicted = model.predict(sequence)
    next_move = tokenizer.index_word.get(predicted.argmax(), "(Movimiento no reconocido)")

    # Validar y normalizar el movimiento predicho
    if next_move and next_move != "(Movimiento no reconocido)":
        if next_move.lower() in ['o', 'o-o', 'o-o-o']:
            next_move = next_move.upper()
        elif next_move[0] in 'rnbqk':
            next_move = next_move[0].upper() + next_move[1:]

    return next_move

# Variables globales para el temporizador
game_time = 300  # 5 minutos en segundos
time_remaining = game_time
timer_running = True

def update_timer():
    """Actualiza el temporizador cada segundo."""
    global time_remaining, timer_running
    while timer_running and time_remaining > 0:
        mins, secs = divmod(time_remaining, 60)
        timer_label.config(text=f"Tiempo restante: {mins:02}:{secs:02}")
        time.sleep(1)
        time_remaining -= 1

    if time_remaining <= 0:
        timer_label.config(text="¡Tiempo agotado!")
        Button(popup, text="Finalizar Partida", command=root.quit).pack()

def add_time():
    """Añade 5 segundos al temporizador."""
    global time_remaining
    time_remaining += 5

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
max_sequence_len = 20 

# Crear la ventana principal de la interfaz
root = Tk()
root.title("Sugerencia de Movimiento de Ajedrez")

# Variables de entrada y salida
input_var = StringVar()
current_moves = ""
board = chess.Board() 

# Mostrar el tablero al inicio
board_image = board_to_image(board)
popup = Toplevel(root)
popup.title("Tablero de Ajedrez")
img_label = Label(popup, image=board_image)
img_label.image = board_image
img_label.pack()

# Etiqueta para el temporizador
timer_label = Label(popup, text="Tiempo restante: 03:00", font=("Arial", 16), fg="red")
timer_label.pack()

# Iniciar el temporizador en un hilo separado
timer_thread = threading.Thread(target=update_timer, daemon=True)
timer_thread.start()

# Función para actualizar la imagen del tablero
def update_board_image():
    """Actualiza la imagen del tablero en la interfaz."""
    global board_image
    board_image = board_to_image(board)
    img_label.config(image=board_image)
    img_label.image = board_image

# Función para mostrar el tablero y sugerir el siguiente movimiento
def show_board():
    global current_moves, board, move_history
    user_move = input_var.get().strip()

    # Normalizar enroques
    if user_move.lower() == 'o-o':
        user_move = 'O-O'
    elif user_move.lower() == 'o-o-o':
        user_move = 'O-O-O'

    if user_move.lower() == 'exit':
        global timer_running
        timer_running = False  # Detener el temporizador
        popup.destroy()
        root.quit()
        return

    # Verificar si el movimiento del usuario es legal
    try:
        legal_moves_san = [board.san(m) for m in board.legal_moves]
        if user_move not in legal_moves_san:
            Label(popup, text=f"Movimiento inválido: {user_move}", fg="red").pack()
            input_var.set("")  # Limpiar el campo de entrada
            entry.focus_set()  # Volver a enfocar el campo de entrada
            return
    except Exception as e:
        Label(popup, text=f"Error inesperado: {str(e)}", fg="red").pack()
        input_var.set("")
        entry.focus_set()
        return

    # Aplicar el movimiento del usuario si es válido
    try:
        board.push_san(user_move)
        current_moves += (" " + user_move if current_moves else user_move)
        add_time()  # Añadir tiempo por el movimiento del usuario

        # Añadir el movimiento del usuario al historial
        move_history.append((user_move, None))  # Máquina aún no ha movido
    except Exception as e:
        Label(popup, text=f"Error al aplicar el movimiento: {str(e)}", fg="red").pack()
        input_var.set("")
        entry.focus_set()
        return

    # Actualizar el tablero y mostrarlo
    update_board_image()

    # Predecir el siguiente movimiento
    try:
        next_move = predict_next_move(move_suggester, tokenizer, current_moves, max_sequence_len)

        # Validar el movimiento sugerido
        legal_moves_san = [board.san(m) for m in board.legal_moves]
        if next_move not in legal_moves_san:
            next_move = next((move for move in legal_moves_san if move.startswith(next_move)), None)

        if next_move:
            board.push_san(next_move)
            current_moves += " " + next_move
            # Actualizar el historial con el movimiento de la máquina
            move_history[-1] = (move_history[-1][0], next_move) 
            add_time() 
        else:
            Label(popup, text=f"El movimiento sugerido no es legal: {next_move}", fg="red").pack()
    except Exception as e:
        Label(popup, text=f"Error al predecir el movimiento: {str(e)}", fg="red").pack()

    # Actualizar el tablero con el movimiento sugerido
    update_board_image()
    input_var.set("") 
    entry.focus_set()

# Crear un popup con la interfaz
Label(popup, text="Introduce movimientos en formato PGN (o 'exit' para salir):").pack()
entry = Entry(popup, textvariable=input_var, width=50)
entry.pack()
entry.bind('<Return>', lambda event: show_board())

# Botón para mostrar el tablero
Button(popup, text="Enviar Movimiento", command=show_board).pack()

# Botón para abrir el historial de movimientos
Button(popup, text="Ver Historial", command=create_history_popup).pack()


root.mainloop()

print("Programa finalizado.")