import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_and_prepare_data(data_path):
    data = pd.read_csv(data_path)
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