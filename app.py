from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

st.set_page_config(page_title="CinemaMatch PCA", layout="wide")

DATA_DIR = Path(__file__).resolve().parent

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

st.markdown(
    """
<style>
.stApp { background: #f5f5f5; color: #222; }

.block-container {
    max-width: 980px;
    padding-top: 5.5rem;
    padding-bottom: 5.5rem;
}

.card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
}

.hero {
    background: white;
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
}

.movie-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 6px;
}

.movie-meta {
    color: #666;
    font-size: 0.94rem;
    line-height: 1.5;
}

.score {
    display: inline-block;
    margin-top: 10px;
    padding: 4px 10px;
    border-radius: 999px;
    background: #f0f0f0;
    font-weight: 600;
}

.small-note {
    color: #666;
    font-size: 0.92rem;
}

.nav {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(255,255,255,0.96);
    border-top: 1px solid #ddd;
    padding: 10px 18px;
    z-index: 999;
}

div.stButton > button {
    background: #222 !important;
    color: white !important;
    border: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    ratings = pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t",
        names=["user", "item", "rating", "time"],
    )

    movies = pd.read_csv(
        DATA_DIR / "u.item",
        sep="|",
        names=["item", "title", "date", "v", "url", *GENRES],
        encoding="latin-1",
    )

    movies["genres"] = movies.apply(
        lambda row: ", ".join([genre for genre in GENRES if str(row[genre]) == "1"]),
        axis=1,
    )
    movies["year"] = movies["date"].fillna("").str[-4:]
    movies["year"] = movies["year"].where(movies["year"].str.fullmatch(r"\d{4}"), "")
    return ratings, movies



ratings, movies = load_data()
matrix = ratings.pivot(index="user", columns="item", values="rating")

mask = ~matrix.isna()
random_mask = np.random.rand(*matrix.shape) < 0.1
holdout = mask & random_mask

train = matrix.mask(holdout)

means = train.mean(axis=1)
centered = train.sub(means, axis=0).fillna(0)

pca = PCA(n_components=20)
latent = pca.fit_transform(centered)
recon = pca.inverse_transform(latent)

final = pd.DataFrame(recon, index=matrix.index, columns=matrix.columns).add(means, axis=0).clip(1, 5)

if "i" not in st.session_state:
    st.session_state.i = 0
if "ratings" not in st.session_state:
    st.session_state.ratings = {}
if "sample" not in st.session_state:
    st.session_state.sample = movies.sample(20, random_state=42).reset_index(drop=True)

sample = st.session_state.sample


st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0;">CinemaMatch PCA</h1>
        <div class="small-note">Califica algunas peliculas y obten tu Top 10 personalizado.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")
st.subheader("Como funciona el sistema")
st.markdown(
    """
    Este sistema construye una **matriz usuario-item**, centra los ratings por usuario y aplica
    **PCA** para reconstruir preferencias. A partir de esa reconstruccion, estima los ratings
    faltantes y genera un **Top 10** de peliculas no calificadas.
    """
)

with st.expander("Ver logica principal en Python"):
    st.code(
        """
matrix = ratings.pivot(index="user", columns="item", values="rating")

means = matrix.mean(axis=1)
centered = matrix.sub(means, axis=0).fillna(0)

pca = PCA(n_components=20)
latent = pca.fit_transform(centered)
recon = pca.inverse_transform(latent)

final = pd.DataFrame(recon, index=matrix.index, columns=matrix.columns).add(means, axis=0)
        """,
        language="python",
    )

st.markdown("<br>", unsafe_allow_html=True)

progress = min(st.session_state.i / len(sample), 1.0)
st.progress(progress)
st.caption(f"Peliculas calificadas: {min(st.session_state.i, len(sample))} de {len(sample)}")

if st.session_state.i < len(sample):
    movie = sample.iloc[st.session_state.i]
    movie_id = int(movie["item"])

    saved_rating = st.session_state.ratings.get(movie_id, 3)

    st.markdown(
        f"""
        <div class="card">
            <div class="movie-title">{movie['title']}</div>
            <div class="movie-meta">Generos: {movie['genres'] or 'No disponible'}</div>
            <div class="movie-meta">Ano: {movie['year'] or 'No disponible'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Tu calificacion")
    st.caption("1 = muy mala, 5 = excelente")

    rating = st.radio(
        "Selecciona una nota",
        ["1", "2", "3", "4", "5"],
        index=int(saved_rating) - 1,
        horizontal=True,
        label_visibility="collapsed",
    )

    st.session_state.ratings[movie_id] = int(rating)

    st.markdown('<div class="nav">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    if col1.button("Anterior"):
        if st.session_state.i > 0:
            st.session_state.i -= 1
            st.rerun()

    if col2.button("Siguiente"):
        st.session_state.i += 1
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

else:
    st.subheader("Top 10 recomendado")
    st.caption("Peliculas no vistas con mayor puntuacion predicha por el modelo")

    user_vector = pd.Series(np.nan, index=matrix.columns)
    for item_id, score in st.session_state.ratings.items():
        user_vector[item_id] = score

    user_mean = user_vector.mean()
    centered_user = user_vector.sub(user_mean).fillna(0)

    latent_user = pca.transform([centered_user])
    recon_user = pca.inverse_transform(latent_user)[0]

    pred = pd.Series(np.clip(recon_user + user_mean, 1, 5), index=matrix.columns)
    recs = pred[user_vector.isna()].sort_values(ascending=False).head(10)

    result = pd.DataFrame({"item": recs.index, "score": recs.values}).merge(movies, on="item", how="left")

    for pos, (_, row) in enumerate(result.iterrows(), start=1):
        st.markdown(
            f"""
            <div class="card">
                <div class="movie-title">{pos}. {row['title']}</div>
                <div class="movie-meta">Generos: {row['genres'] or 'No disponible'}</div>
                <div class="movie-meta">Ano: {row['year'] or 'No disponible'}</div>
                <div class="score">Score estimado: {row['score']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Reiniciar"):
        st.session_state.i = 0
        st.session_state.ratings = {}
        st.session_state.sample = movies.sample(20, random_state=42).reset_index(drop=True)
        st.rerun()

# ---------- EVALUACIÓN DEL MODELO ----------

true = matrix.where(holdout).stack()
predicted = final.where(holdout).stack()

rmse = np.sqrt(((true - predicted)**2).mean())

col1, col2 = st.columns([1, 3])

with col1:
    st.metric("RMSE", f"{rmse:.3f}")


