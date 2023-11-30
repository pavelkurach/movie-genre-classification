import functools

from datasets import DatasetDict, formatting
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast


class Preprocessor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
    ):
        self.tokenizer = tokenizer

    def transform(self, split_dataset: DatasetDict) -> DatasetDict:
        return (
            split_dataset.map(
                self._lower_and_strip_plot, desc="Preprocess plot"
            )
            .map(
                self._split_genres,
                desc="Split genres str",
            )
            .remove_columns(["genre"])
            .map(
                functools.partial(self._tokenize, tokenizer=self.tokenizer),
                desc="Tokenize",
            )
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
        delimeters = ["/", ",", "-", "[", "]"]
        genres = movie["genre"]
        for delimeter in delimeters:
            genres = " ".join(genres.split(delimeter))
        genres = list(map(lambda s: s.strip(), genres.split()))
        return {"genres": genres}

    @staticmethod
    def _tokenize(
        movies: formatting.formatting.LazyRow,
        tokenizer: PreTrainedTokenizerFast,
    ) -> BatchEncoding:
        return tokenizer(movies["plot"], truncation=True, max_length=512)
