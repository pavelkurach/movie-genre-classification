import functools
from collections import Counter

from datasets import Dataset, DatasetDict, formatting


class GenreEncoder:
    def __init__(
        self,
        n_most_freq_genres: int = 15,
    ):
        self.n_most_freq_genres = n_most_freq_genres
        self._most_freq_genres: list[str] | None = None

    def transform(self, dataset_with_genres_split: DatasetDict) -> DatasetDict:
        if self._most_freq_genres is None:
            raise RuntimeError("Run .fit() first.")
        return (
            dataset_with_genres_split.map(
                functools.partial(
                    self._clean_genres,
                    most_freq_genres=self._most_freq_genres,
                ),
                desc="Clean genres",
            )
            .filter(lambda movie: len(movie["genres"]) > 0)
            .map(
                functools.partial(
                    self._encode_movie_genre,
                    most_freq_genres=self._most_freq_genres,
                ),
                desc="Encode genres",
            )
        )

    def fit(self, dataset_with_genres_split: Dataset) -> None:
        self._most_freq_genres = self._get_most_frequent_genres(
            dataset_with_genres_split
        )

    def get_num_labels(self) -> int:
        if self._most_freq_genres is None:
            raise RuntimeError("Run .fit() first.")
        return len(self._most_freq_genres)

    def get_id2label(self) -> dict[int, str]:
        if self._most_freq_genres is None:
            raise RuntimeError("Run .fit() first.")
        return {idx: label for idx, label in enumerate(self._most_freq_genres)}

    def get_label2id(self) -> dict[str, int]:
        if self._most_freq_genres is None:
            raise RuntimeError("Run .fit() first.")
        return {label: idx for idx, label in enumerate(self._most_freq_genres)}

    def _get_most_frequent_genres(self, train_dataset: Dataset) -> list[str]:
        train_len = len(train_dataset)
        genre_counts: Counter[str] = Counter()
        for i in range(train_len):
            genres = train_dataset[i]["genres"]
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        most_freq = list(
            map(
                list,
                zip(*genre_counts.most_common(self.n_most_freq_genres + 1)),
            )
        )[0]
        if "unknown" in most_freq:
            most_freq.remove("unknown")
        else:
            most_freq = most_freq[: self.n_most_freq_genres]
        return most_freq

    @staticmethod
    def _clean_genres(
        movie: formatting.formatting.LazyRow, most_freq_genres: list[str]
    ) -> dict[str, list[float]]:
        genres_clean = []
        for genre in movie["genres"]:
            if genre in most_freq_genres:
                genres_clean.append(genre)
        return {"genres": genres_clean}

    @staticmethod
    def _encode_movie_genre(
        movie: formatting.formatting.LazyRow, most_freq_genres: list[str]
    ) -> dict[str, list[float]]:
        genre_enc = []
        for genre in most_freq_genres:
            genre_enc.append(float(genre in movie["genres"]))
        return {"labels": genre_enc}
