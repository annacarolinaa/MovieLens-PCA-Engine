from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


st.set_page_config(
    page_title="CinemaMatch PCA",
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
            --bg: #fffaf2;
            --panel: rgba(255, 255, 255, 0.78);
            --ink: #3f2a1f;
            --muted: #6d5649;
            --line: #ecc9aa;
            --accent: #c45a11;
            --accent-2: #874014;
            --shadow: 0 18px 40px rgba(124, 68, 25, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, #fff2dc 0%, #fff8ee 35%, #f8ead7 100%);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fff7eb 0%, #f4e0c7 100%);
            border-right: 1px solid rgba(135, 64, 20, 0.12);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }

        h1, h2, h3 {
            font-family: Georgia, Cambria, "Times New Roman", serif;
            color: var(--ink);
            letter-spacing: 0.01em;
        }

        .hero-panel {
            background: linear-gradient(135deg, rgba(255, 249, 239, 0.98), rgba(245, 220, 189, 0.92));
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.7rem;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
            margin-bottom: 1rem;
        }

        .hero-panel::after {
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            right: -90px;
            top: -80px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(196, 90, 17, 0.22), rgba(196, 90, 17, 0));
        }

        .eyebrow {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(196, 90, 17, 0.12);
            color: var(--accent-2);
            border: 1px solid rgba(196, 90, 17, 0.18);
            font-size: 0.82rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .hero-title {
            font-size: 2.3rem;
            line-height: 1.05;
            margin: 0 0 0.7rem 0;
        }

        .hero-copy {
            max-width: 760px;
            font-size: 1rem;
            line-height: 1.6;
            color: var(--muted);
            margin-bottom: 0.95rem;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 0.7rem;
        }

        .pill {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            color: var(--accent-2);
            font-size: 0.88rem;
        }

        .metric-grid,
        .workflow-grid,
        .rec-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.9rem;
            margin: 1rem 0 1.25rem 0;
        }

        .workflow-grid {
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        }

        .metric-card,
        .workflow-card,
        .rec-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 28px rgba(90, 49, 18, 0.06);
            backdrop-filter: blur(6px);
        }

        .metric-label,
        .workflow-step,
        .rec-label {
            color: var(--muted);
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-family: Georgia, Cambria, "Times New Roman", serif;
            font-size: 1.9rem;
            line-height: 1.1;
            color: var(--ink);
        }

        .metric-note,
        .workflow-copy,
        .rec-meta {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .workflow-card {
            background: linear-gradient(180deg, rgba(255, 250, 242, 0.98), rgba(255, 244, 227, 0.92));
        }

        .workflow-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: var(--accent);
            color: #fff8ef;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .section-copy {
            color: var(--muted);
            margin-top: -0.3rem;
            margin-bottom: 1rem;
        }

        .rec-card {
            background: linear-gradient(180deg, rgba(255, 252, 246, 0.98), rgba(252, 239, 220, 0.95));
        }

        .rec-title {
            font-size: 1.12rem;
            color: var(--ink);
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.35rem;
        }

        .score-chip {
            display: inline-block;
            margin-top: 0.7rem;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            background: rgba(196, 90, 17, 0.14);
            color: var(--accent-2);
            font-weight: 700;
        }

        [data-testid="stTabs"] button {
            font-weight: 600;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid var(--line);
            padding: 0.8rem;
            border-radius: 18px;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero-panel {
                padding: 1.2rem;
                border-radius: 20px;
            }

            .hero-title {
                font-size: 1.7rem;
            }

            .metric-card,
            .workflow-card,
            .rec-card {
                border-radius: 18px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_genre_text(row: pd.Series) -> str:
    genres = [genre for genre in GENRE_COLUMNS if row.get(genre, 0) == 1]
    return ", ".join(genres) if genres else "Unknown"


@st.cache_data(show_spinner=False)
def load_data():
    ratings = pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    item_columns = [
        "item_id",
        "movie_title",
        "release_date",
        "video_release_date",
        "IMDb_URL",
        *GENRE_COLUMNS,
    ]
    movies = pd.read_csv(
        DATA_DIR / "u.item",
        sep="|",
        names=item_columns,
        encoding="latin-1",
    )
    movies["release_year"] = movies["release_date"].str[-4:]
    movies["release_year"] = movies["release_year"].where(
        movies["release_year"].str.fullmatch(r"\d{4}"),
        "",
    )
    movies["genres"] = movies.apply(build_genre_text, axis=1)
    return ratings, movies


@st.cache_data(show_spinner=False)
def prepare_recommender(
    ratings: pd.DataFrame,
    n_components: int,
    holdout_fraction: float = 0.10,
    seed: int = 42,
):
    full_matrix = (
        ratings.pivot_table(index="user_id", columns="item_id", values="rating")
        .sort_index()
        .sort_index(axis=1)
    )

    observed_mask = ~full_matrix.isna()
    rng = np.random.default_rng(seed)
    holdout_mask = observed_mask & (rng.random(full_matrix.shape) < holdout_fraction)

    if not holdout_mask.any().any():
        first_row, first_col = np.argwhere(observed_mask.to_numpy())[0]
        holdout_mask.iat[first_row, first_col] = True

    train_matrix = full_matrix.mask(holdout_mask)
    global_mean = float(ratings["rating"].mean())
    user_means = train_matrix.mean(axis=1).fillna(global_mean)

    matrix_centered = train_matrix.sub(user_means, axis=0)
    matrix_imputed = matrix_centered.fillna(0.0)

    max_components = min(n_components, matrix_imputed.shape[0], matrix_imputed.shape[1])
    pca = PCA(n_components=max_components, random_state=seed)
    components = pca.fit_transform(matrix_imputed)
    reconstructed = pca.inverse_transform(components)

    reconstructed_df = pd.DataFrame(
        reconstructed,
        index=full_matrix.index,
        columns=full_matrix.columns,
    )
    final_matrix = reconstructed_df.add(user_means, axis=0).clip(1, 5)

    hidden_actual = full_matrix.where(holdout_mask).stack()
    hidden_predicted = final_matrix.where(holdout_mask).stack()
    rmse = float(np.sqrt(mean_squared_error(hidden_actual.values, hidden_predicted.values)))

    explained_variance = float(pca.explained_variance_ratio_.sum())
    sparsity = float(full_matrix.isna().sum().sum() / full_matrix.size)

    return {
        "full_matrix": full_matrix,
        "train_matrix": train_matrix,
        "final_matrix": final_matrix,
        "holdout_mask": holdout_mask,
        "user_means": user_means,
        "pca": pca,
        "rmse": rmse,
        "explained_variance": explained_variance,
        "sparsity": sparsity,
    }


@st.cache_data(show_spinner=False)
def build_movie_stats(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    stats = ratings.groupby("item_id", as_index=False).agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
    )
    stats = stats.merge(
        movies[["item_id", "movie_title", "genres", "release_year"]],
        on="item_id",
        how="left",
    )
    return stats.sort_values(["rating_count", "avg_rating"], ascending=[False, False])


@st.cache_data(show_spinner=False)
def build_similarity_matrix(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    min_ratings: int = 100,
    top_movies: int = 12,
):
    popular = ratings.groupby("item_id").filter(lambda rows: len(rows) >= min_ratings)
    merged = popular.merge(movies[["item_id", "movie_title"]], on="item_id", how="left")
    top_titles = merged["movie_title"].value_counts().head(top_movies).index
    filtered = merged[merged["movie_title"].isin(top_titles)]
    utility_matrix = filtered.pivot_table(
        index="user_id",
        columns="movie_title",
        values="rating",
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = utility_matrix.corr()
    return corr_matrix.fillna(0)


def recommend_for_existing_user(
    user_id: int,
    full_matrix: pd.DataFrame,
    final_matrix: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    original = full_matrix.loc[user_id]
    predicted = final_matrix.loc[user_id]
    unseen_items = original[original.isna()].index
    recommendations = (
        predicted.loc[unseen_items]
        .sort_values(ascending=False)
        .head(top_n)
        .rename("predicted_rating")
        .reset_index()
    )
    return recommendations.merge(movies, on="item_id", how="left")


def recommend_for_custom_user(
    custom_ratings: dict[int, float],
    model: dict,
    movies: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    item_ids = model["full_matrix"].columns
    profile = pd.Series(np.nan, index=item_ids, dtype=float)
    for item_id, rating in custom_ratings.items():
        if item_id in profile.index:
            profile.loc[item_id] = rating

    if profile.notna().sum() == 0:
        return pd.DataFrame()

    profile_mean = float(profile.dropna().mean())
    centered = profile.sub(profile_mean).fillna(0.0)

    latent = model["pca"].transform(pd.DataFrame([centered], columns=item_ids))
    reconstructed = model["pca"].inverse_transform(latent)[0]
    predicted = pd.Series(np.clip(reconstructed + profile_mean, 1, 5), index=item_ids)

    recommendations = (
        predicted[profile.isna()]
        .sort_values(ascending=False)
        .head(top_n)
        .rename("predicted_rating")
        .reset_index()
    )
    return recommendations.merge(movies, on="item_id", how="left")


def render_metric_grid(cards: list[dict[str, str]]) -> None:
    html = ['<div class="metric-grid">']
    for card in cards:
        html.append(
            f"""
            <div class="metric-card">
                <div class="metric-label">{card["label"]}</div>
                <div class="metric-value">{card["value"]}</div>
                <div class="metric-note">{card["note"]}</div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_workflow_cards() -> None:
    steps = [
        ("01", "Escondemos ratings", "Reservamos 10% de los ratings observados para evaluar la reconstruccion."),
        ("02", "Centramos por usuario", "Cada usuario se ajusta alrededor de su propia media para reducir sesgos de severidad."),
        ("03", "Aplicamos PCA", "Reducimos dimensionalidad y capturamos tendencias latentes del gusto cinematografico."),
        ("04", "Reconstruimos", "Completamos la matriz usuario-item y generamos ratings estimados entre 1 y 5."),
    ]

    html = ['<div class="workflow-grid">']
    for number, title, copy in steps:
        html.append(
            f"""
            <div class="workflow-card">
                <div class="workflow-number">{number}</div>
                <div class="workflow-step">{title}</div>
                <div class="workflow-copy">{copy}</div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_recommendation_cards(recommendations: pd.DataFrame) -> None:
    html = ['<div class="rec-grid">']
    for _, row in recommendations.iterrows():
        year = row["release_year"] if row["release_year"] else "s/f"
        html.append(
            f"""
            <div class="rec-card">
                <div class="rec-label">Recomendacion</div>
                <div class="rec-title">{row["movie_title"]}</div>
                <div class="rec-meta">{row["genres"]}</div>
                <div class="rec-meta">Ano: {year} | Item ID: {int(row["item_id"])} </div>
                <div class="score-chip">Score estimado: {row["predicted_rating"]:.2f}</div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def create_missing_heatmap(full_matrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sample = full_matrix.iloc[:50, :50]
    sns.heatmap(
        sample.isna(),
        yticklabels=False,
        xticklabels=False,
        cbar=False,
        cmap=["#f2b56b", "#a64512"],
        ax=ax,
    )
    ax.set_title("Mapa de datos faltantes (muestra 50 x 50)", color="#8b3d14", fontsize=13)
    ax.set_xlabel("Peliculas")
    ax.set_ylabel("Usuarios")
    fig.tight_layout()
    return fig


def create_reconstructed_heatmap(final_matrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.heatmap(
        final_matrix.iloc[:20, :20],
        cmap="YlOrBr",
        linewidths=0.35,
        annot=True,
        fmt=".1f",
        ax=ax,
    )
    ax.set_title("Matriz reconstruida (muestra 20 x 20)", color="#8b3d14", fontsize=13)
    ax.set_xlabel("Peliculas")
    ax.set_ylabel("Usuarios")
    fig.tight_layout()
    return fig


def create_variance_chart(model: dict):
    variance = np.cumsum(model["pca"].explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(variance) + 1), variance, marker="o", color="#c45a11", linewidth=2.2)
    ax.fill_between(range(1, len(variance) + 1), variance, color="#f6c487", alpha=0.35)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Numero de componentes")
    ax.set_ylabel("Varianza acumulada")
    ax.set_title("Varianza explicada acumulada", color="#8b3d14", fontsize=13)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    return fig


def create_scatter_chart(movie_stats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    scatter = ax.scatter(
        movie_stats["rating_count"],
        movie_stats["avg_rating"],
        c=movie_stats["avg_rating"],
        s=np.clip(movie_stats["rating_count"], 20, 250),
        cmap="YlOrBr",
        alpha=0.72,
        edgecolors="#8b3d14",
        linewidths=0.25,
    )
    ax.set_title("Popularidad vs rating promedio", color="#8b3d14", fontsize=13)
    ax.set_xlabel("Numero de evaluaciones")
    ax.set_ylabel("Rating promedio")
    ax.grid(alpha=0.25, linestyle="--")
    fig.colorbar(scatter, ax=ax, label="Rating promedio")
    fig.tight_layout()
    return fig


def create_similarity_heatmap(corr_matrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        cmap="YlOrBr",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        square=True,
        cbar_kws={"label": "Correlacion"},
        ax=ax,
    )
    ax.set_title("Similaridad entre peliculas populares", color="#8b3d14", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


def create_recommendation_chart(recommendations: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = recommendations.sort_values("predicted_rating", ascending=True)
    ax.barh(
        ordered["movie_title"],
        ordered["predicted_rating"],
        color=sns.color_palette("YlOrBr", n_colors=len(ordered)),
        edgecolor="#8b3d14",
    )
    ax.set_xlim(1, 5)
    ax.set_xlabel("Rating estimado")
    ax.set_ylabel("Pelicula")
    ax.set_title(title, color="#8b3d14", fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def get_user_summary(
    user_id: int,
    full_matrix: pd.DataFrame,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> dict[str, str]:
    user_ratings = full_matrix.loc[user_id].dropna()
    seen_count = int(user_ratings.shape[0])
    mean_rating = float(user_ratings.mean())

    rated_movies = ratings.loc[ratings["user_id"] == user_id, ["item_id", "rating"]].merge(
        movies[["item_id", *GENRE_COLUMNS]],
        on="item_id",
        how="left",
    )
    genre_scores = {}
    for genre in GENRE_COLUMNS:
        filtered = rated_movies.loc[rated_movies[genre] == 1, "rating"]
        if not filtered.empty:
            genre_scores[genre] = filtered.mean()
    favorite_genre = max(genre_scores, key=genre_scores.get) if genre_scores else "Unknown"

    return {
        "seen_count": f"{seen_count}",
        "mean_rating": f"{mean_rating:.2f}",
        "favorite_genre": favorite_genre,
    }


inject_styles()
ratings, movies = load_data()

st.sidebar.title("Controles")
st.sidebar.markdown(
    "Ajusta el modelo del taller y explora recomendaciones sobre MovieLens 100k."
)

max_pca = min(60, ratings["user_id"].nunique(), ratings["item_id"].nunique())
n_components = st.sidebar.slider(
    "Componentes PCA",
    min_value=5,
    max_value=max_pca,
    value=20,
    help="Mas componentes capturan mas varianza, pero pueden perder simplicidad.",
)
top_n = st.sidebar.slider(
    "Numero de recomendaciones",
    min_value=5,
    max_value=15,
    value=8,
)
selected_user = st.sidebar.selectbox(
    "Usuario del dataset",
    options=sorted(ratings["user_id"].unique()),
    index=0,
)
min_popularity = st.sidebar.slider(
    "Minimo de ratings para similaridad",
    min_value=50,
    max_value=200,
    value=100,
    step=10,
)

with st.spinner("Entrenando modelo PCA y preparando visualizaciones..."):
    model = prepare_recommender(ratings, n_components=n_components)
    movie_stats = build_movie_stats(ratings, movies)
    similarity_matrix = build_similarity_matrix(
        ratings,
        movies,
        min_ratings=min_popularity,
        top_movies=12,
    )

st.markdown(
    """
    <section class="hero-panel">
        <div class="eyebrow">Proyecto final de sistemas de recomendacion</div>
        <div class="hero-title">CinemaMatch PCA</div>
        <div class="hero-copy">
            Tablero interactivo construido a partir del flujo visto en clase:
            matriz usuario-item, centrado por usuario, matrix completion con PCA,
            evaluacion con ratings ocultos y recomendacion top-N con titulos y generos.
        </div>
        <div class="pill-row">
            <span class="pill">MovieLens 100k</span>
            <span class="pill">Matrix completion</span>
            <span class="pill">PCA como motor</span>
            <span class="pill">Diseno responsivo</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

render_metric_grid(
    [
        {
            "label": "Usuarios",
            "value": f"{ratings['user_id'].nunique()}",
            "note": "Perfiles disponibles en la base.",
        },
        {
            "label": "Peliculas",
            "value": f"{ratings['item_id'].nunique()}",
            "note": "Catalogo modelado por el recomendador.",
        },
        {
            "label": "Ratings",
            "value": f"{len(ratings):,}",
            "note": "Interacciones historicas observadas.",
        },
        {
            "label": "Sparsity",
            "value": f"{model['sparsity']:.1%}",
            "note": "Huecos reales de la matriz usuario-item.",
        },
        {
            "label": "RMSE",
            "value": f"{model['rmse']:.3f}",
            "note": "Error en el 10% de ratings escondidos.",
        },
        {
            "label": "Varianza",
            "value": f"{model['explained_variance']:.1%}",
            "note": "Varianza retenida por el PCA actual.",
        },
    ]
)

st.markdown("### Flujo del modelo")
st.markdown(
    '<div class="section-copy">El tablero sigue la misma logica presentada en los slides, solo que empaquetada en una experiencia lista para entregar.</div>',
    unsafe_allow_html=True,
)
render_workflow_cards()

tab_overview, tab_analysis, tab_recommender = st.tabs(
    ["Resumen del modelo", "Exploracion y similaridad", "Recomendador interactivo"]
)

with tab_overview:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Datos faltantes")
        st.pyplot(create_missing_heatmap(model["full_matrix"]), use_container_width=True)

    with col_b:
        st.markdown("#### Matrix completion")
        st.pyplot(create_reconstructed_heatmap(model["final_matrix"]), use_container_width=True)

    st.markdown("#### Varianza explicada")
    st.pyplot(create_variance_chart(model), use_container_width=True)

    st.info(
        "La evaluacion se hace ocultando aleatoriamente el 10% de los ratings observados, tal como pide el taller. "
        "Luego se entrena el PCA sobre la matriz centrada por usuario y se mide el error de reconstruccion."
    )

with tab_analysis:
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### Popularidad vs rating promedio")
        st.pyplot(create_scatter_chart(movie_stats), use_container_width=True)

    with right:
        st.markdown("#### Top peliculas mejor valoradas")
        st.dataframe(
            movie_stats[["movie_title", "avg_rating", "rating_count", "genres"]]
            .head(10)
            .rename(
                columns={
                    "movie_title": "Pelicula",
                    "avg_rating": "Promedio",
                    "rating_count": "Ratings",
                    "genres": "Generos",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Similaridad entre peliculas populares")
    st.pyplot(create_similarity_heatmap(similarity_matrix), use_container_width=True)

    reference_movie = st.selectbox(
        "Pelicula de referencia para buscar similares",
        options=movie_stats["movie_title"].head(80).tolist(),
        index=0,
    )
    reference_item = movies.loc[movies["movie_title"] == reference_movie, "item_id"].iloc[0]
    reference_ratings = model["full_matrix"][reference_item]
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity_scores = model["full_matrix"].corrwith(reference_ratings)
    similar_movies = (
        similarity_scores
        .dropna()
        .rename("correlation")
        .reset_index()
        .merge(movie_stats, on="item_id", how="left")
    )
    similar_movies = similar_movies[similar_movies["movie_title"] != reference_movie]
    similar_movies = similar_movies[similar_movies["rating_count"] >= min_popularity]
    similar_movies = similar_movies.sort_values("correlation", ascending=False).head(8)

    st.dataframe(
        similar_movies[
            ["movie_title", "correlation", "avg_rating", "rating_count", "genres"]
        ].rename(
            columns={
                "movie_title": "Pelicula similar",
                "correlation": "Correlacion",
                "avg_rating": "Promedio",
                "rating_count": "Ratings",
                "genres": "Generos",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_recommender:
    mode = st.radio(
        "Modo de recomendacion",
        options=["Usuario del dataset", "Usuario invitado"],
        horizontal=True,
    )

    if mode == "Usuario del dataset":
        user_summary = get_user_summary(selected_user, model["full_matrix"], ratings, movies)
        st.markdown("#### Perfil del usuario")

        summary_cols = st.columns(3)
        summary_cols[0].metric("Peliculas vistas", user_summary["seen_count"])
        summary_cols[1].metric("Rating medio", user_summary["mean_rating"])
        summary_cols[2].metric("Genero favorito", user_summary["favorite_genre"])

        recommendations = recommend_for_existing_user(
            selected_user,
            model["full_matrix"],
            model["final_matrix"],
            movies,
            top_n=top_n,
        )

        st.markdown("#### Top recomendaciones")
        render_recommendation_cards(recommendations)
        st.pyplot(
            create_recommendation_chart(
                recommendations,
                f"Top {top_n} para el usuario {selected_user}",
            ),
            use_container_width=True,
        )
        st.dataframe(
            recommendations[
                ["movie_title", "predicted_rating", "genres", "release_year"]
            ].rename(
                columns={
                    "movie_title": "Pelicula",
                    "predicted_rating": "Score estimado",
                    "genres": "Generos",
                    "release_year": "Ano",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.markdown(
            "#### Califica algunas peliculas y deja que el PCA te recomiende el resto"
        )
        candidate_movies = movie_stats.head(20)[
            ["item_id", "movie_title", "rating_count", "genres"]
        ].copy()
        selected_titles = st.multiselect(
            "Selecciona peliculas que ya viste",
            options=candidate_movies["movie_title"].tolist(),
            default=candidate_movies["movie_title"].head(4).tolist(),
        )

        custom_ratings: dict[int, float] = {}
        if selected_titles:
            slider_cols = st.columns(2)
            for index, title in enumerate(selected_titles):
                movie_row = candidate_movies.loc[candidate_movies["movie_title"] == title].iloc[0]
                with slider_cols[index % 2]:
                    rating_value = st.slider(
                        f"{title}",
                        min_value=1.0,
                        max_value=5.0,
                        value=4.0,
                        step=0.5,
                        help=f"{int(movie_row['rating_count'])} ratings historicos | {movie_row['genres']}",
                    )
                    custom_ratings[int(movie_row["item_id"])] = rating_value

        if len(custom_ratings) < 3:
            st.warning("Califica al menos 3 peliculas para generar recomendaciones mas estables.")
        else:
            guest_recommendations = recommend_for_custom_user(
                custom_ratings=custom_ratings,
                model=model,
                movies=movies,
                top_n=top_n,
            )
            st.markdown("#### Recomendaciones para el usuario invitado")
            render_recommendation_cards(guest_recommendations)
            st.pyplot(
                create_recommendation_chart(
                    guest_recommendations,
                    f"Top {top_n} para el usuario invitado",
                ),
                use_container_width=True,
            )
            st.dataframe(
                guest_recommendations[
                    ["movie_title", "predicted_rating", "genres", "release_year"]
                ].rename(
                    columns={
                        "movie_title": "Pelicula",
                        "predicted_rating": "Score estimado",
                        "genres": "Generos",
                        "release_year": "Ano",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

st.caption(
    "Base metodologica tomada del taller: MovieLens 100k, matriz usuario-item, ocultamiento del 10%, "
    "centrado por usuario, PCA para matrix completion y recomendacion top-N."
)
