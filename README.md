# CinemaMatch PCA

## Overview / Visión general / Visão geral

CinemaMatch PCA is an interactive movie recommendation app built with **Streamlit**, **PCA**, and the **MovieLens 100k** dataset.

El proyecto muestra cómo PCA puede reconstruir una matriz usuario-item para predecir ratings faltantes.

O projeto demonstra um sistema que usa PCA para estimar avaliações ausentes e recomendar filmes.

## What it does / Qué hace / O que faz

- Loads the MovieLens 100k dataset from `u.data` and `u.item`
- Builds a user-item rating matrix
- Hides 10% of observed ratings for evaluation
- Centers ratings by user mean
- Applies PCA to reconstruct the rating matrix
- Estimates missing user ratings and generates recommendations
- Displays model metrics, heatmaps and recommended movies

## How it works / Cómo funciona / Como funciona

1. Carga los archivos `u.data` y `u.item`.
2. Construye la matriz usuario-item.
3. Oculta aleatoriamente 10% de los ratings observados.
4. Centra los ratings según la media de cada usuario.
5. Rellena valores faltantes con cero en la matriz centrada.
6. Aplica PCA para reducir dimensionalidad y reconstruir la matriz.
7. Calcula el error con RMSE usando los valores ocultos.
8. Recomienda películas no vistas según las puntuaciones estimadas.

## Technologies / Tecnologías / Tecnologias

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Dataset

The app uses the **MovieLens 100k** dataset.

- `u.data`: user rating records
- `u.item`: movie metadata and genres

## Usage / Uso / Uso

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

3. Open the browser at the address shown by Streamlit.

## Project structure / Estructura del proyecto / Estrutura do projeto

```text
.
|-- app.py
|-- u.data
|-- u.item
|-- requirements.txt
|-- README.md
```

## Expected result / Resultado esperado / Resultado esperado

When the app runs, you can explore the dataset, see how PCA reconstructs the rating matrix, and view personalized movie recommendations in a clean interface.

## Notes / Notas / Observações

- `app.py`, `u.data` and `u.item` remain unchanged.
- The project is focused on the Streamlit app and the MovieLens dataset.

## Author / Autor

Academic project developed for recommendation systems coursework.
