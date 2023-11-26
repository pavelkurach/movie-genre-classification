import functools
from pathlib import Path

from datasets import Dataset, formatting, load_from_disk
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast


class Preprocessor:
    def __init__(
        self, split_dataset: Dataset | None, split: str | None, tokenizer: str
    ):
        self.split = split
        self.path_to_data = (Path('..') / 'data').resolve()
        self.dataset = (
            split_dataset
            if split_dataset is not None
            else self._load_dataset()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def preprocess(self) -> Dataset:
        if self.dataset is None:
            raise RuntimeError('Dataset not loaded or provided')
        self.dataset = (
            self.dataset.map(self._lower_and_strip_plot)
            .map(self._split_genres)
            .remove_columns(['genre'])
            .map(functools.partial(self._tokenize, tokenizer=self.tokenizer))
        )
        return self.dataset

    def _load_dataset(self) -> Dataset:
        if self.split is None:
            raise RuntimeError('Cannot load dataset: split not indicated')
        path_to_split = self.path_to_data / f'{self.split}.hf'
        return load_from_disk(str(path_to_split))

    def save(self) -> None:
        if self.split is None:
            raise RuntimeError('Cannot save dataset: split not indicated')
        self.dataset.save_to_disk(
            str(self.path_to_data / f'{self.split}_preprocessed.hf')
        )

    def get_dataset(self) -> Dataset:
        return self.dataset

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
        return tokenizer(movies['plot'], truncation=True)
