import functools
from pathlib import Path

from datasets import DatasetDict, formatting, load_from_disk
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast


class Preprocessor:
    def __init__(
        self,
        split_dataset_dict: DatasetDict | None,
        path_to_split: str | None,
        tokenizer: AutoTokenizer,
    ):
        self.path_to_dataset = path_to_split
        self.path_to_data = (Path('..') / 'data').resolve()
        self.dataset_dict = (
            split_dataset_dict
            if split_dataset_dict is not None
            else self._load_dataset()
        )
        self.tokenizer = tokenizer

    def preprocess(self) -> DatasetDict:
        if self.dataset_dict is None:
            raise RuntimeError('Dataset not loaded or provided')
        self.dataset_dict = (
            self.dataset_dict.map(self._lower_and_strip_plot)
            .map(self._split_genres)
            .remove_columns(['genre'])
            .map(functools.partial(self._tokenize, tokenizer=self.tokenizer))
        )
        return self.dataset_dict

    def _load_dataset(self) -> DatasetDict:
        if self.path_to_dataset is None:
            raise RuntimeError('Cannot load dataset: path not indicated')
        return load_from_disk(str(self.path_to_dataset))

    def save(self) -> None:
        self.dataset_dict.save_to_disk(str(self.path_to_data / 'preprocessed'))

    def get_dataset_dict(self) -> DatasetDict:
        return self.dataset_dict

    @staticmethod
    def _lower_and_strip_plot(
        movie: formatting.formatting.LazyRow,
    ) -> dict[str, str]:
        return {
            'plot': movie['plot'].strip().lower(),
        }

    @staticmethod
    def _split_genres(
        movie: formatting.formatting.LazyRow,
    ) -> dict[str, list[str]]:
        delimeters = ['/', ',']
        genres = movie['genre']
        for delimeter in delimeters:
            genres = ' '.join(genres.split(delimeter))
        genres = list(map(lambda s: s.strip(), genres.split()))
        return {'genres': genres}

    @staticmethod
    def _tokenize(
        movies: formatting.formatting.LazyRow,
        tokenizer: PreTrainedTokenizerFast,
    ) -> BatchEncoding:
        return tokenizer(movies['plot'], truncation=True, max_length=512)
