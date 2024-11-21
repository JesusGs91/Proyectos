# Proyecto de Predicción y Sugerencia de Movimientos en Ajedrez

Este proyecto tiene varios objetivos principales:

1. **Predecir el ganador de una partida de ajedrez** usando características como el rating de los jugadores, el número de movimientos, y otros detalles del juego.
2. **Predecir la duración de una partida de ajedrez** basándose en características como el número de movimientos, los ratings de los jugadores, y la apertura.
3. **Realizar clustering de partidas de ajedrez** para identificar patrones en partidas con características similares.
4. **Sugerir el mejor movimiento en función de los movimientos previos** y la apertura en curso, mediante un modelo de secuencia.

## Estructura del Proyecto

project-root/
│   ├── Data/
│   │   ├── raw
│   │       ├── data.csv           # Datos
│   ├── memoria/
│   │   ├── memoria.pdf            # Memoria PDF
│   ├── images                     # Imagenes Utilziadas
│   ├── predict_move.py            # Predicción del próximo movimiento y visualización del tablero
│   ├── predict_winner.py          # Predicción del ganador y comparación de modelos
│   ├── Googlecolab_run.py         # Entrenamiento de modelos en Google Colab
│   ├── main_Interfaz_juego.py     # Interfaz de usuario para el juego
│   ├── main_Modelos.py            # Entrenamiento de modelos de predicción de duración y número de turnos
│   ├── main_predicciones_juego.py # Evaluación del modelo de predicción de movimientos
│   └── README.md                  # Documentación del proyecto

## Descripción de Scripts

### 1. `Googlecolab_run.py`

Este script fue utilizado para entrenar los modelos en Google Colab. Incluye la carga y preparación de los datos, así como el entrenamiento de dos modelos:

- **Modelo de Predicción de Ganador**: Entrena y guarda un modelo utilizando Random Forest, Gradient Boosting, XGBoost y SVM, y selecciona el mejor modelo basado en el rendimiento.
- **Modelo de Sugerencia de Movimientos**: Entrena un modelo LSTM para predecir el siguiente movimiento basado en secuencias de movimientos anteriores. Guarda el modelo entrenado como `move_suggester.h5`.


### 2. `predict_move.py`

Este script predice el mejor movimiento a partir de una secuencia de movimientos en formato PGN:

- **Input**: Introduce movimientos en formato PGN desde la consola (ej. `e4 e5 Nf3 Nc6`).
- **Output**: El script sugiere el mejor movimiento y muestra el tablero actual en la consola.

### 3. `predict_winner.py`

Este script utiliza el modelo entrenado para predecir el ganador de una partida de ajedrez en función de las características de entrada:

- **Input**: Introduce características de partida como `turns`, `white_rating`, `black_rating`, `rating_advantage_white`, `game_duration`, y `opening_ply`.
- **Output**: Predice el ganador de la partida (blancas o negras).

### 4. `main_Interfaz_juego.py`

Este script gestiona la interfaz gráfica donde el usuario interactúa con el juego. Utiliza `Tkinter` para mostrar el tablero y permitir la introducción de movimientos en formato PGN.

- **Input**: El usuario ingresa los movimientos y recibe la sugerencia del próximo movimiento por parte del modelo.
- **Output**: Muestra el tablero actualizado después de cada movimiento, junto con un historial de los movimientos.

### 5. `main_Modelos.py`

Este script entrena modelos de predicción para la duración de la partida y el número de turnos, y evalúa el rendimiento mediante métricas como la MAE y RMSE. Utiliza modelos de regresión como Random Forest y Gradient Boosting.

- **Input**: Características de las partidas como el número de turnos, rating, etc.
- **Output**: Predicciones de la duración de la partida y el número de turnos.

### 6. `main_predicciones_juego.py`

Este script evalúa el modelo de predicción de movimientos utilizando datos de validación. Calcula la precisión y genera matrices de confusión para evaluar el rendimiento del modelo.

- **Input**: Movimientos históricos de partidas.
- **Output**: Precisión del modelo y visualización de la matriz de confusión.

## Instalación

Asegúrate de tener las siguientes bibliotecas instaladas:
```bash
pip install pandas numpy tensorflow scikit-learn joblib python-chess matplotlib seaborn pillow wand
