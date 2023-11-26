import functools
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, formatting, load_from_disk


class GenreEncoder:
    def __init__(
        self,
        preprocessed_dataset_dict: DatasetDict | None,
        path_to_preprocessed: Path | None,
        n_most_freq_genres: int = 15,
    ):
        self.path_to_dataset = path_to_preprocessed
        self.path_to_data = (Path('..') / 'data').resolve()
        self.dataset_dict = (
            preprocessed_dataset_dict
            if preprocessed_dataset_dict is not None
            else self._load_train_preprocessed()
        )
        self.n_most_freq_genres = n_most_freq_genres
        self._most_freq_genres: list[str] = []
        self._trained = False

    def encode(self) -> DatasetDict:
        self._most_freq_genres = self._get_most_frequent_genres()
        self.dataset_dict = self.dataset_dict.map(
            functools.partial(
                self._encode_movie_genre,
                most_freq_genres=self._most_freq_genres,
            )
        )
        return self.dataset_dict

    def save(self) -> None:
        if not self._trained:
            raise RuntimeError('Run .train() first')
        self.dataset_dict.save_to_disk(str(self.path_to_data / 'encoded'))

    def train(self) -> None:
        self._most_freq_genres = self._get_most_frequent_genres()
        self._trained = True

    def get_num_labels(self) -> int:
        if not self._trained:
            raise RuntimeError('Run .train() first')
        return len(self._most_freq_genres)

    def get_id2label(self) -> dict[int, str]:
        if not self._trained:
            raise RuntimeError('Run .train() first')
        return {idx: label for idx, label in enumerate(self._most_freq_genres)}

    def get_label2id(self) -> dict[str, int]:
        if not self._trained:
            raise RuntimeError('Run .train() first')
        return {label: idx for idx, label in enumerate(self._most_freq_genres)}

    def _load_train_preprocessed(self) -> DatasetDict:
        return load_from_disk(str(self.path_to_dataset))

    def _get_most_frequent_genres(self) -> list[str]:
        train_dataset = self.dataset_dict['train']
        train_len = len(train_dataset)
        genre_counts: Counter[str] = Counter()
        for i in range(train_len):
            genres = train_dataset[i]['genres']
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
        return {'labels': genre_enc}
