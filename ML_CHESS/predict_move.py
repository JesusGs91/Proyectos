import chess
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils.data_prep import load_and_prepare_data

model = load_model('models/move_suggester.h5')
tokenizer = joblib.load('models/tokenizer.pkl')

def predict_next_move(input_text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')
    predicted = model.predict(sequence)
    next_move = tokenizer.index_word[np.argmax(predicted)]
    return next_move

def show_board(moves):
    board = chess.Board()
    for move in moves.split():
        board.push_san(move)
    print(board)

def main():
    print("Introduce movimientos en formato PGN (ej: 'e4 e5 Nf3 Nc6'):")
    input_text = input().strip()
    next_move = predict_next_move(input_text, max_sequence_len=20)
    print("Tablero actual:")
    show_board(input_text)
    print("\nSiguiente movimiento sugerido:", next_move)

if __name__ == "__main__":
    main()
