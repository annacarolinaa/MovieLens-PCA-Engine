from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import TruncatedSVD

st.set_page_config(
    page_title="CinemaMatch Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parent
GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Childrens",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            color: #e5e7eb;
            background-color: #0b1120;
        }
        .stApp {
            background: linear-gradient(180deg, #0b1120 0%, #111827 100%);
            color: #e5e7eb;
        }
        .block-container {
            padding: 1.2rem 2rem 2rem 2rem;
            max-width: 1100px;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: Inter, system-ui, sans-serif;
            color: #ffffff;
        }
        .movie-card {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 16px;
        }
        .movie-title {
            color: #f8fafc;
            font-size: 1.1rem;
            margin: 0;
        }
        .movie-meta {
            color: #94a3b8;
            font-size: 0.92rem;
            margin: 6px 0 0;
        }
        .stButton > button {
            background-color: #0ea5e9 !important;
            color: #ffffff !important;
            border: none !important;
        }
        .subtitle {
            color: #94a3b8;
            margin-top: -10px;
            margin-bottom: 24px;
        }
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data():
    ratings = pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    movie_columns = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        *GENRE_COLUMNS,
    ]
    movies = pd.read_csv(
        DATA_DIR / "u.item",
        sep="|",
        names=movie_columns,
        encoding="latin-1",
        engine="python",
    )
    movies["genre_list"] = movies.apply(
        lambda row: ", ".join([genre for genre in GENRE_COLUMNS if int(row[genre]) == 1]),
        axis=1,
    )
    movies["release_year"] = movies["release_date"].str[-4:]
    movies["release_year"] = movies["release_year"].where(
        movies["release_year"].str.fullmatch(r"\d{4}"), "",
    )
    return ratings, movies[["item_id", "title", "genre_list", "release_year"]]


@st.cache_data(show_spinner=False)
def get_popular_movies(ratings: pd.DataFrame, min_ratings: int = 50) -> pd.DataFrame:
    popular_ids = ratings["item_id"].value_counts()
    popular_ids = popular_ids[popular_ids >= min_ratings].index
    return popular_ids


@st.cache_data(show_spinner=False)
def build_user_item_matrix(ratings: pd.DataFrame, popular_ids: pd.Index) -> pd.DataFrame:
    matrix = ratings[ratings["item_id"].isin(popular_ids)].pivot(
        index="user_id", columns="item_id", values="rating"
    )
    return matrix


def get_recommendations(
    user_ratings: dict[int, int],
    matrix: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
    n_components: int = 20,
) -> pd.DataFrame:
    user_id = int(matrix.index.max() + 1)
    new_user = pd.Series(user_ratings, name=user_id)
    combined = pd.concat([matrix, new_user.to_frame().T], sort=False)

    user_means = combined.mean(axis=1)
    centered = combined.sub(user_means, axis=0).fillna(0.0)

    n_components = min(n_components, centered.shape[0] - 1, centered.shape[1] - 1)
    n_components = max(1, n_components)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    factors = svd.fit_transform(centered)
    reconstructed = svd.inverse_transform(factors)

    predictions = (
        pd.DataFrame(reconstructed, index=combined.index, columns=combined.columns)
        .add(user_means, axis=0)
        .clip(1, 5)
    )

    user_pred = predictions.loc[user_id].drop(index=list(user_ratings.keys()), errors="ignore")
    top_recs = user_pred.sort_values(ascending=False).head(top_n).reset_index()
    top_recs.columns = ["item_id", "predicted_rating"]
    top_recs = top_recs.merge(movies, on="item_id", how="left")
    return top_recs


def reset_session():
    for key in [
        "view",
        "started",
        "current_idx",
        "my_ratings",
        "name",
        "sample_movies",
    ]:
        if key in st.session_state:
            del st.session_state[key]


inject_styles()
ratings, movies = load_data()
popular_ids = get_popular_movies(ratings, min_ratings=60)

if "view" not in st.session_state:
    st.session_state.view = "home"
if "started" not in st.session_state:
    st.session_state.started = False
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "my_ratings" not in st.session_state:
    st.session_state.my_ratings = {}
if "name" not in st.session_state:
    st.session_state.name = ""
if "sample_movies" not in st.session_state:
    st.session_state.sample_movies = (
        movies[movies["item_id"].isin(popular_ids)]
        .sample(20, random_state=42)
        .reset_index(drop=True)
    )

st.sidebar.title("CinemaMatch")
st.sidebar.markdown("Escolha a forma de ver recomendações e quantas você quer receber.")
selected_model = st.sidebar.selectbox("Modelo", ["SVD"], help="Use a collaborative filtering model based on SVD.")
top_n = st.sidebar.slider("Número de recomendações", 5, 15, 10)

if st.session_state.view == "home":
    st.title("CinemaMatch")
    st.markdown("<p class='subtitle'>Avalie 20 filmes para receber recomendações personalizadas.</p>", unsafe_allow_html=True)

    if not st.session_state.started:
        name_input = st.text_input("Seu nome", value=st.session_state.name)
        if st.button("Começar"):
            if name_input.strip():
                st.session_state.name = name_input.strip()
                st.session_state.started = True
                st.experimental_rerun()
            else:
                st.warning("Por favor, insira seu nome para continuar.")
    else:
        current_idx = st.session_state.current_idx
        movie = st.session_state.sample_movies.iloc[current_idx]

        st.markdown(
            f"""
            <div class='movie-card'>
                <p class='movie-title'>{movie['title']}</p>
                <p class='movie-meta'>🎭 {movie['genre_list']} • 📅 {movie['release_year']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rating = st.radio(
            "Como você avalia este filme?",
            ["1", "2", "3", "4", "5"],
            index=0,
            key=f"rating_{current_idx}",
            horizontal=True,
        )
        st.session_state.my_ratings[int(movie["item_id"])] = int(rating)

        st.progress((current_idx + 1) / len(st.session_state.sample_movies))
        st.write(f"Filme {current_idx + 1} de {len(st.session_state.sample_movies)}")

        col1, col2, col3 = st.columns([1, 1, 1])
        if col1.button("⬅️ Anterior") and current_idx > 0:
            st.session_state.current_idx -= 1
            st.experimental_rerun()
        if col3.button("Próximo ➡️"):
            if current_idx < len(st.session_state.sample_movies) - 1:
                st.session_state.current_idx += 1
                st.experimental_rerun()
        if current_idx == len(st.session_state.sample_movies) - 1:
            if col2.button("Ver recomendações"):
                st.session_state.view = "results"
                st.experimental_rerun()

elif st.session_state.view == "results":
    st.title("Suas Recomendações")
    st.markdown(f"<p class='subtitle'>Obrigado, {st.session_state.name}! Aqui estão filmes que você provavelmente vai gostar.</p>", unsafe_allow_html=True)

    with st.spinner("Gerando recomendações..."):
        user_item = build_user_item_matrix(ratings, popular_ids)
        recommendations = get_recommendations(
            st.session_state.my_ratings,
            user_item,
            movies,
            top_n=top_n,
            n_components=25,
        )

    if recommendations.empty:
        st.warning("Não foi possível gerar recomendações com os dados fornecidos.")
    else:
        for _, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div class='movie-card'>
                    <p class='movie-title'>{row['title']}</p>
                    <p class='movie-meta'>🎭 {row['genre_list']} • 📅 {row['release_year']} • ⭐ {row['predicted_rating']:.1f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button("Avaliar novamente", type="primary"):
        reset_session()
        st.experimental_rerun()
