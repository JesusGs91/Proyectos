# Proyecto de Predicción y Sugerencia de Movimientos en Ajedrez

Este proyecto tiene dos objetivos principales:
1. **Predecir el ganador de una partida de ajedrez** usando características como el rating de los jugadores, el número de movimientos, y otros detalles del juego.
2. **Sugerir el mejor movimiento en función de los movimientos previos** y la apertura en curso, mediante un modelo de secuencia.

## Estructura del Proyecto

project-root/
├── src/
│   ├── utils/
│   │   ├── data_prep.py       # Carga y preparación de datos
│   │   ├── model_train.py     # Entrenamiento de modelos y GridSearch
│   ├── models/
│   │   ├── move_suggester.h5  # Modelo de sugerencia de movimientos entrenado
│   │   ├── winner_predictor.pkl # Modelo de predicción de ganador entrenado
│   ├── predict_move.py        # Predicción del próximo movimiento y visualización del tablero
│   ├── predict_winner.py      # Predicción del ganador y comparación de modelos
└── README.md

## Descripción de Scripts

### 1. `data_prep.py`

Este script contiene funciones para cargar y preparar los datos:

- `load_and_prepare_data`: Carga el archivo `games.csv`, crea características adicionales como `rating_advantage_white` y `game_duration`, y prepara los datos para el modelo de predicción de ganador.
- `prepare_sequence_data`: Prepara las secuencias de movimientos para el modelo de sugerencia de movimientos, utilizando `Tokenizer` para convertir los movimientos en secuencias numéricas y realizando padding.

### 2. `model_train.py`

Este script entrena ambos modelos:
  
- **Modelo de Predicción de Ganador**: Compara tres modelos (Random Forest, Gradient Boosting y SVM) usando `GridSearchCV` para encontrar los mejores hiperparámetros. El mejor modelo se guarda en `winner_predictor.pkl`.
  
- **Modelo de Sugerencia de Movimientos**: Entrena un modelo de red neuronal (LSTM) para predecir el siguiente movimiento. El modelo final se guarda en `move_suggester.h5`.

### 3. `predict_move.py`

Este script predice el mejor movimiento a partir de una secuencia de movimientos en formato PGN:

- **Input**: Introduce movimientos en formato PGN desde la consola (ej. `e4 e5 Nf3 Nc6`).
- **Output**: El script sugiere el mejor movimiento y muestra el tablero actual en la consola.

### 4. `predict_winner.py`

Este script utiliza el modelo entrenado para predecir el ganador de una partida de ajedrez en función de las características de entrada:

- **Input**: Introduce características de partida como `turns`, `white_rating`, `black_rating`, `rating_advantage_white`, `game_duration`, y `opening_ply`.
- **Output**: Predice el ganador de la partida (blancas o negras).

## Instalación

Asegúrate de tener las siguientes bibliotecas instaladas:
```bash
pip install pandas numpy tensorflow scikit-learn joblib python-chess