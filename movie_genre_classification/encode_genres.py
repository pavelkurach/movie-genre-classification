import functools
from collections import Counter
from pathlib import Path

from datasets import Dataset, formatting, load_from_disk


class GenreEncoder:
    def __init__(
        self, train_preprocessed: Dataset | None, n_most_freq_genres: int = 15
    ):
        self.path_to_data = (Path('..') / 'data').resolve()
        self.dataset = (
            train_preprocessed
            if train_preprocessed is not None
            else self._load_train_preprocessed()
        )
        self.n_most_freq_genres = n_most_freq_genres
        self._most_freq_genres: list[str] = []
        self._encoded = False

    def encode(self) -> Dataset:
        self._most_freq_genres = self._get_most_frequent_genres()
        self.dataset = self.dataset.map(
            functools.partial(
                self._encode_movie_genre,
                most_freq_genres=self._most_freq_genres,
            )
        )
        self._encoded = True
        return self.dataset

    def get_id2label(self) -> dict[int, str]:
        if not self._encoded:
            raise RuntimeError('Run .encode() first')
        return {idx: label for idx, label in enumerate(self._most_freq_genres)}

    def get_label2id(self) -> dict[str, int]:
        if not self._encoded:
            raise RuntimeError('Run .encode() first')
        return {label: idx for idx, label in enumerate(self._most_freq_genres)}

    def _load_train_preprocessed(self) -> Dataset:
        path_to_train_preprocessed = (
            self.path_to_data / 'train_preprocessed.hf'
        )
        return load_from_disk(str(path_to_train_preprocessed))

    def _get_most_frequent_genres(self) -> list[str]:
        dataset_len = len(self.dataset)
        genre_counts: Counter[str] = Counter()
        for i in range(dataset_len):
            genres = self.dataset[i]['genres']
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        most_freq = list(
            map(
                list,
                zip(*genre_counts.most_common(self.n_most_freq_genres + 1)),
            )
        )[0]
        if 'unknown' in most_freq:
            most_freq.remove('unknown')
        else:
            most_freq = most_freq[: self.n_most_freq_genres]
        return most_freq

    @staticmethod
    def _encode_movie_genre(
        movie: formatting.formatting.LazyRow, most_freq_genres: list[str]
    ) -> dict[str, list[float]]:
        genre_enc = []
        for genre in most_freq_genres:
            genre_enc.append(float(genre in movie['genres']))
        return {'genres_enc': genre_enc}
