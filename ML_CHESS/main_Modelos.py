import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, root_mean_squared_error,accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de rutas
data_path = 'DATA/raw/games.csv'  # Ruta al dataset
model_dir = 'models/'
winner_predictor_path = model_dir + 'winner_predictor.pkl'

# Cargar y preparar datos
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    data['rating_advantage_white'] = data['white_rating'] - data['black_rating']
    data['game_duration'] = (pd.to_datetime(data['last_move_at'], unit='ms') - pd.to_datetime(data['created_at'], unit='ms')).dt.total_seconds() / 60
    features = ['turns', 'white_rating', 'black_rating', 'rating_advantage_white', 'game_duration', 'opening_ply']
    X = data[features]
    y = data['winner'].apply(lambda x: 1 if x == 'white' else (0 if x == 'black' else -1))
    X = X[y != -1]
    y = y[y != -1]
    return X, y

# Predicción de duración de la partida (Regresión)
def predict_game_duration(X, y):
    print("\nEntrenando modelo de predicción de duración de partida...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")

    return model

# Predicción del número de turnos (Regresión)
def predict_turns(X, y):
    print("\nEntrenando modelo de predicción del número de turnos...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test, y_pred, squared=False):.2f}")

    return model

# Evaluar clustering (Aprendizaje no supervisado)
def evaluate_clustering(data, n_clusters=3):
    print("\nEvaluando modelo de clustering...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)

    data['Cluster'] = kmeans.labels_
    print(f"Centroides de los clusters: {kmeans.cluster_centers_}")
    print(f"Distribución de los clusters: {data['Cluster'].value_counts()}")

    return kmeans, data

# Función para evaluar modelos Machine Learning
def create_and_evaluate_models(X, y):
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'KNeighbors': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': GradientBoostingClassifier(random_state=42)
    }
    param_grids = {
        'RandomForest': {'n_estimators': [10, 20, 30, 40, 50, 100], 'max_depth': [5, 10, 20, 30]},
        'GradientBoosting': {'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 4, 5], 'subsample': [0.8, 0.9, 1.0]},
        'KNeighbors': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
        'LogisticRegression': {'C': [0.01, 0.1, 1, 10, 100]},
        'XGBoost': {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 10], 'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.8, 0.9]}
    }

    best_models = {}
    for name, model in models.items():
        print(f"Evaluando {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Mejor modelo para {name}: {grid_search.best_params_}")
        print(f"Cross-validation score: {cross_val_score(grid_search.best_estimator_, X, y, cv=5).mean():.4f}")
        
        # Mostrar matriz de confusión y reporte de clasificación
        y_pred = grid_search.best_estimator_.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión para {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        print(classification_report(y_test, y_pred))
    
    return best_models

# Predicción de duración de la partida (Regresión)
def predict_game_duration(X, y):
    print("\nEntrenando modelo de predicción de duración de partida...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Cálculo de métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Visualización de predicciones vs reales
    plt.figure(figsize=(12, 6))
    
    # Gráfico de dispersión
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicciones')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Línea de Identidad')
    plt.title('Predicción vs Real - Duración de la Partida')
    plt.xlabel('Duración Real')
    plt.ylabel('Duración Predicha')
    plt.legend()
    
    # Histograma de residuos
    plt.subplot(1, 2, 2)
    residuos = y_test - y_pred
    plt.hist(residuos, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.title('Distribución de Residuos')
    plt.xlabel('Residuos (Real - Predicho)')
    plt.ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.show()

    return model

# Predicción del número de turnos (Regresión)
def predict_turns(X, y):
    print("\nEntrenando modelo de predicción del número de turnos...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

    # Visualizar predicciones vs reales
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('Predicción vs Real - Número de Turnos')
    plt.xlabel('Real')
    plt.ylabel('Predicción')
    plt.show()

    return model

# Evaluar clustering (Aprendizaje no supervisado)
def evaluate_clustering(data, n_clusters=3):
    print("\nEvaluando modelo de clustering...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)

    data['Cluster'] = kmeans.labels_
    print(f"Centroides de los clusters: {kmeans.cluster_centers_}")
    print(f"Distribución de los clusters: {data['Cluster'].value_counts()}")

    # Visualizar los clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(data['white_rating'], data['black_rating'], c=data['Cluster'], cmap='viridis', alpha=0.7)
    plt.title('Clusters de Partidas')
    plt.xlabel('White Rating')
    plt.ylabel('Black Rating')
    plt.colorbar(label='Cluster')
    plt.show()

    return kmeans, data

# Cargar y preparar datos
print("Cargando y preparando datos...")
X, y = load_and_prepare_data(data_path)

# Evaluar predicción de duración de partida
print("\nEvaluando predicción de duración de partida...")
game_duration_model = predict_game_duration(X[['white_rating', 'black_rating', 'turns', 'opening_ply']], X['game_duration'])

# Evaluar clustering
print("\nEvaluando clustering en las partidas...")
kmeans_model, clustered_data = evaluate_clustering(X[['white_rating', 'black_rating']])

# Evaluar predicción del número de turnos
print("\nEvaluando predicción del número de turnos...")
turns_model = predict_turns(X[['white_rating', 'black_rating', 'opening_ply']], X['turns'])

print("\nEntrenamiento y evaluación completados.")
# Cargar y preparar datos
print("Cargando y preparando datos...")
X, y = load_and_prepare_data(data_path)

# Evaluar predicción de duración de partida
print("\nEvaluando predicción de duración de partida...")
game_duration_model = predict_game_duration(X[['white_rating', 'black_rating', 'turns', 'opening_ply']], X['game_duration'])

# Evaluar clustering
print("\nEvaluando clustering en las partidas...")
kmeans_model, clustered_data = evaluate_clustering(X[['white_rating', 'black_rating']])

# Evaluar predicción del número de turnos
print("\nEvaluando predicción del número de turnos...")
turns_model = predict_turns(X[['white_rating', 'black_rating', 'opening_ply']], X['turns'])

# Crear y evaluar modelos de predicción de ganador
print("\nEntrenando y evaluando modelos de predicción de ganador...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
best_models = create_and_evaluate_models(X, y)

# Guardar el mejor modelo (por ejemplo, GradientBoosting)
best_model = best_models['GradientBoosting']
joblib.dump(best_model, winner_predictor_path)
print(f"Mejor modelo de predicción de ganador guardado en {winner_predictor_path}")

print("Entrenamiento y evaluación completados.")
