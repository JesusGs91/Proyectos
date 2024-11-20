
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm
import chess

# Configuración de rutas
data_path = '/content/drive/MyDrive/ML_CHESS/data/raw/games.csv'  # Ruta en Google Drive
model_dir = '/content/drive/MyDrive/ML_CHESS/models/'
move_suggester_path = model_dir + 'move_suggester.h5'
winner_predictor_path = model_dir + 'winner_predictor.pkl'

# Funciones de preparación de datos
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    data['rating_advantage_white'] = data['white_rating'] - data['black_rating']
    data['game_duration'] = (pd.to_datetime(data['last_move_at'], unit='ms') - pd.to_datetime(data['created_at'], unit='ms')).dt.total_seconds() / 60
    return data

def prepare_sequence_data(moves):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(moves)
    sequences = tokenizer.texts_to_sequences(moves)
    vocab_size = len(tokenizer.word_index) + 1

    input_sequences, next_moves = [], []
    for sequence in sequences:
        for i in range(1, len(sequence)):
            input_sequences.append(sequence[:i])
            next_moves.append(sequence[i])

    max_sequence_len = max(len(seq) for seq in input_sequences)
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
    next_moves = np.array(next_moves)

    return input_sequences, next_moves, vocab_size, max_sequence_len, tokenizer

# Funciones de entrenamiento de modelos
def train_winner_predictor(X, y):
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    param_grids = {
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [10]},
        'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.1]},
        'SVM': {'C': [1], 'kernel': ['linear']}
    }
    best_models = {}
    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=1, verbose=1)
        grid_search.fit(X, y)
        best_models[name] = grid_search.best_estimator_
    best_model_name = max(best_models, key=lambda x: accuracy_score(y, best_models[x].predict(X)))
    best_model = best_models[best_model_name]
    joblib.dump(best_model, winner_predictor_path)
    return best_model

def train_move_suggester(X, y, vocab_size, max_sequence_len):
    model = Sequential([
        Embedding(vocab_size, 50, input_length=max_sequence_len),
        LSTM(100),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=64, verbose=1)
    model.save(move_suggester_path)
    return model

# Cargar y preparar datos
print("Cargando y preparando datos...")
data = load_and_prepare_data(data_path)
moves = data['moves'].values

# Preparar secuencias de movimientos
print("Preparando secuencias de movimientos...")
input_sequences, next_moves, vocab_size, max_sequence_len, tokenizer = prepare_sequence_data(moves)

# Entrenar modelo de predicción de ganador
print("\nEntrenando modelo de predicción de ganador...")
features = ['turns', 'white_rating', 'black_rating', 'rating_advantage_white', 'game_duration', 'opening_ply']
X = data[features]
y = data['winner'].apply(lambda x: 1 if x == 'white' else (0 if x == 'black' else -1))
X = X[y != -1]
y = y[y != -1]
winner_predictor = train_winner_predictor(X, y)
print(f"Modelo de predicción de ganador guardado en {winner_predictor_path}")

# Entrenar modelo de sugerencia de movimientos
print("\nEntrenando modelo de sugerencia de movimientos...")
move_suggester = train_move_suggester(input_sequences, next_moves, vocab_size, max_sequence_len)
print(f"Modelo de sugerencia de movimientos guardado en {move_suggester_path}")

# Ejemplo de uso: Predicción de movimiento
def predict_next_move(model, tokenizer, input_text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')
    predicted = model.predict(sequence)
    next_move = tokenizer.index_word[predicted.argmax()]
    return next_move

print("\nEjemplo de sugerencia de movimiento:")
input_text = "e4 e5 Nf3 Nc6"
print("Movimientos iniciales:", input_text)
move_suggester = load_model(move_suggester_path)
next_move = predict_next_move(move_suggester, tokenizer, input_text, max_sequence_len)
print("Siguiente movimiento sugerido:", next_move)

# Ejemplo de uso: Predicción de ganador
def predict_winner(model, features):
    prediction = model.predict([features])[0]
    return "white" if prediction == 1 else "black"

print("\nEjemplo de predicción de ganador:")
winner_predictor = joblib.load(winner_predictor_path)
example_features = [20, 1500, 1400, 100, 30, 3]  # [turns, white_rating, black_rating, rating_diff, game_duration, opening_ply]
predicted_winner = predict_winner(winner_predictor, example_features)
print("Ganador previsto:", predicted_winner)