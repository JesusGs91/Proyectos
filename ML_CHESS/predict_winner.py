import joblib
from data_prep import load_and_prepare_data

def predict_winner(features):
    model = joblib.load('./models/winner_predictor.pkl')
    prediction = model.predict([features])[0]
    return "white" if prediction == 1 else "black"

def main():
    # Ejemplo de características de entrada
    features = [20, 1500, 1400, 100, 30, 3]  # [turns, white_rating, black_rating, rating_diff, game_duration, opening_ply]
    winner = predict_winner(features)
    print("Ganador previsto:", winner)

if __name__ == "__main__":
    main()