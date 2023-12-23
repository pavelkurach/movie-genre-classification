from datasets import DatasetDict, formatting

from .genre_encoder import GenreEncoder


class Preprocessor:
    def __init__(
        self,
        n_most_freq_genres: int = 15,
    ):
        self.genre_encoder = GenreEncoder(n_most_freq_genres)

    def transform(
        self, split_dataset: DatasetDict
    ) -> tuple[DatasetDict, dict[int, str], dict[str, int]]:
        dataset_with_genres_split = (
            split_dataset.map(
                self._lower_and_strip_plot, desc="Preprocess plot"
            )
            .map(
                self._split_genres,
                desc="Split genres str",
            )
            .remove_columns(["genre"])
        )
        self.genre_encoder.fit(dataset_with_genres_split["train"])
        return (
            self.genre_encoder.transform(dataset_with_genres_split),
            self.genre_encoder.get_id2label(),
            self.genre_encoder.get_label2id(),
        )

    @staticmethod
    def _lower_and_strip_plot(
        movie: formatting.formatting.LazyRow,
    ) -> dict[str, str]:
        return {
            "plot": movie["plot"].strip().lower(),
        }

    @staticmethod
    def _split_genres(
        movie: formatting.formatting.LazyRow,
    ) -> dict[str, list[str]]:
        delimiters = ["/", ",", "-", "[", "]"]
        genres = movie["genre"]
        for delimiter in delimiters:
            genres = " ".join(genres.split(delimiter))
        genres = list(map(lambda s: s.strip(), genres.split()))
        return {"genres": genres}
