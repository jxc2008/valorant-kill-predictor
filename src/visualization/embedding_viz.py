"""PCA projection and interactive scatter plots of player embedding space."""


def pca_project(embeddings, n_components=2):
    """Project high-dimensional embeddings to 2D/3D via PCA."""
    raise NotImplementedError


def plot_embedding_space(projected, labels, roles):
    """Scatter plot of player embeddings colored by role/archetype."""
    raise NotImplementedError
