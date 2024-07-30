# Proyectos

# Análisis Exploratorio de Datos en Juegos de Ajedrez

## Descripción del Proyecto

Este proyecto tiene como objetivo realizar un análisis exploratorio de datos (EDA) en un conjunto de partidas de ajedrez obtenidas de Kaggle. A través de este análisis, buscamos identificar patrones y tendencias que puedan ofrecer insights valiosos sobre el juego de ajedrez.

## Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal utilizado para el análisis de datos.
- **Pandas**: Biblioteca para la manipulación y análisis de datos.
- **Matplotlib y Seaborn**: Bibliotecas para la visualización de datos.
- **Scipy**: Biblioteca para realizar pruebas estadísticas.
- **Jupyter Notebook**: Entorno interactivo para desarrollar y compartir el análisis.

## Hipótesis Analizadas

### Hipótesis 1: Los jugadores con una calificación más alta tienden a ganar más partidas

#### Metodología
- Se realizó una prueba t de dos muestras independientes para comparar las calificaciones de los jugadores ganadores y perdedores.

#### Resultados
- Los resultados indican que los jugadores ganadores tienen calificaciones significativamente más altas que los perdedores.


### Hipótesis 2: Influencia de las Aperturas en los Resultados de las Partidas

#### Metodología
- Se realizó una prueba de chi-cuadrado para comparar las tasas de victorias entre diferentes aperturas y determinar si el resultado de la partida depende del tipo de apertura utilizado.

#### Resultados
- Las aperturas tienen una influencia significativa en los resultados de las partidas.

### Hipótesis 3: Las partidas con un mayor número de turnos tienden a terminar en tablas

#### Metodología
- Se realizó un análisis de varianza (ANOVA) para comparar el número de turnos entre las partidas que terminan en tablas y las que terminan en jaque mate, resignación o tiempo agotado.

#### Resultados
- Las partidas que terminan en tablas tienden a tener un mayor número de turnos.