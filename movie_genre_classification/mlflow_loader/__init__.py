from ..genre_classifier import GenreClassifierOnnx


def _load_pyfunc(path: str) -> GenreClassifierOnnx:
    genre_classifier = GenreClassifierOnnx(path)
    return genre_classifier


__all__ = ["_load_pyfunc"]
