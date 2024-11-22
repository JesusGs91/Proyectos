import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import os

# Configuración de rutas
data_path = 'DATA/raw/games.csv'  # Ruta al dataset
model_dir = 'models/'
move_suggester_path = model_dir + 'move_suggester.h5'
tokenizer_path = model_dir + 'tokenizer.pkl'

# Cargar el modelo y el tokenizador
print("Cargando modelo...")
move_suggester = load_model(move_suggester_path)
print("Cargando tokenizador...")
with open(tokenizer_path, 'rb') as f:
    tokenizer = joblib.load(f)

# Definir longitud máxima de secuencia
max_sequence_len = 20 

# Función para preparar datos de validación
def prepare_validation_data(data_path):
    """Carga y prepara datos de validación desde el archivo."""
    print("Cargando datos de validación...")
    data = pd.read_csv(data_path)

    # Seleccionar partidas donde el ganador está definido y hay movimientos
    data = data.dropna(subset=['winner', 'moves'])
    moves = data['moves'].str.split()

    inputs, targets = [], []

    for game_moves in moves:
        for i in range(1, len(game_moves)):
            inputs.append(" ".join(game_moves[:i])) 
            targets.append(game_moves[i]) 
    return pd.Series(inputs), pd.Series(targets)

# Obtener datos de validación
validation_moves, validation_targets = prepare_validation_data(data_path)

# Tokenizar y preparar las secuencias
validation_sequences = tokenizer.texts_to_sequences(validation_moves)
validation_sequences_padded = pad_sequences(validation_sequences, maxlen=max_sequence_len, padding='pre')

# Realizar predicciones
print("Realizando predicciones...")
predictions = move_suggester.predict(validation_sequences_padded)
predicted_moves = [tokenizer.index_word.get(np.argmax(pred), "(Desconocido)") for pred in predictions]

# Mostrar ejemplos de datos para diagnóstico
print("\nEjemplo de movimientos reales y predichos:")
for i in range(5):
    print(f"Real: {validation_targets.iloc[i]} | Predicho: {predicted_moves[i]}")

# Evaluar el rendimiento
accuracy = accuracy_score(validation_targets, predicted_moves)
print(f"\nPrecisión en el conjunto de validación: {accuracy * 100:.2f}%")

# Generar matriz de confusión
valid_labels = list(set(validation_targets) & set(predicted_moves))

if not valid_labels:
    print("\nNo hay etiquetas válidas comunes entre los movimientos reales y predichos.")
else:
    conf_matrix = confusion_matrix(validation_targets, predicted_moves, labels=valid_labels)
    print("\nMatriz de Confusión:")
    print(conf_matrix)

    # Gráfico de calor mejorado
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=valid_labels, yticklabels=valid_labels, linewidths=.5)
    plt.title("Matriz de Confusión - Movimientos Sugeridos vs Reales", fontsize=16)
    plt.xlabel("Movimientos Predichos", fontsize=14)
    plt.ylabel("Movimientos Reales", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Gráfico de barras para la proporción de aciertos y errores
    correct = sum(np.array(predicted_moves) == np.array(validation_targets))
    incorrect = sum(np.array(predicted_moves) != np.array(validation_targets))

    plt.figure(figsize=(8, 6))
    plt.bar(["Correctos", "Incorrectos"], [correct, incorrect], color=['green', 'red'], alpha=0.75, edgecolor="black")
    plt.title("Distribución de Predicciones Correctas e Incorrectas", fontsize=16)
    plt.ylabel("Cantidad", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()
