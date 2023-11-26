from pathlib import Path

from datasets import Dataset, DatasetDict, formatting, load_dataset
from lib import csv_helpers


class MoviePlotsDataset:
    def __init__(self) -> None:
        self.path_to_data = (Path('..') / 'data').resolve()
        self.path_to_csv = self.path_to_data / 'wiki_movie_plots_deduped.csv'
        self.columns = csv_helpers.get_columns_in_csv(self.path_to_csv)
        self.seed = 42
        self.dataset_dict = self._load_dataset()

    @staticmethod
    def select_genre_and_plot(
        entry: formatting.formatting.LazyRow,
    ) -> dict[str, str]:
        return {
            'genre': entry['Genre'],
            'plot': entry['Plot'],
        }

    def _load_dataset(self) -> DatasetDict:
        return (
            load_dataset(
                'csv', data_files=[str(self.path_to_csv)], split="train"
            )
            .map(MoviePlotsDataset.select_genre_and_plot)
            .remove_columns(self.columns)
            .train_test_split(0.2, seed=self.seed)
        )

    def save_train_test_split(self) -> None:
        for split, dataset in self.dataset_dict.items():
            dataset.save_to_disk(str(self.path_to_data / f'{split}.hf'))

    def get_train_dataset(self) -> Dataset:
        return self.dataset_dict['train']

    def get_test_dataset(self) -> Dataset:
        return self.dataset_dict['test']


if __name__ == '__main__':
    movie_plots_dataset = MoviePlotsDataset()
    print(movie_plots_dataset.get_test_dataset()[0])
