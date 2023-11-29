from pathlib import Path

from datasets import Dataset, DatasetDict, formatting, load_dataset

from .lib import csv_helpers


class MoviePlotsDataset:
    def __init__(self) -> None:
        self.path_to_data = Path('data').resolve()
        self.path_to_csv = self.path_to_data / 'wiki_movie_plots_deduped.csv'
        self.columns = csv_helpers.get_columns_in_csv(self.path_to_csv)
        self.seed = 42
        self.dataset_dict: DatasetDict | None = None
        self._loaded = False

    def load(self) -> DatasetDict:
        self.dataset_dict = self._load_dataset()
        self._loaded = True
        return self.dataset_dict

    def save(self) -> None:
        if self.dataset_dict is None:
            raise RuntimeError('Run .load() first')
        self.dataset_dict.save_to_disk(self.path_to_data / 'split')

    def get_train_dataset(self) -> Dataset:
        if self.dataset_dict is None:
            raise RuntimeError('Run .load() first')
        return self.dataset_dict['train']

    def get_test_dataset(self) -> Dataset:
        if self.dataset_dict is None:
            raise RuntimeError('Run .load() first')
        return self.dataset_dict['test']

    def _load_dataset(self) -> DatasetDict:
        return (
            load_dataset(
                'csv', data_files=[str(self.path_to_csv)], split="train"
            )
            .map(MoviePlotsDataset._select_genre_and_plot)
            .remove_columns(self.columns)
            .train_test_split(0.2, seed=self.seed)
        )

    @staticmethod
    def _select_genre_and_plot(
        entry: formatting.formatting.LazyRow,
    ) -> dict[str, str]:
        return {
            'genre': entry['Genre'],
            'plot': entry['Plot'],
        }
