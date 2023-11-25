from pathlib import Path
from typing import Self

from datasets import formatting, load_dataset
from lib import csv_helpers


class MoviePlotsDataset:
    def __init__(self) -> None:
        self.path_to_data = (Path('..') / 'data').resolve()
        self.path_to_csv = self.path_to_data / 'wiki_movie_plots_deduped.csv'
        self.columns = csv_helpers.get_columns_in_csv(self.path_to_csv)
        self.seed = 42
        self.dataset_dict = None

    @staticmethod
    def select_genre_and_plot(
        entry: formatting.formatting.LazyRow,
    ) -> dict[str, str]:
        return {
            'genre': entry['Genre'],
            'plot': entry['Plot'],
        }

    def load(self) -> Self:
        self.dataset_dict = (
            load_dataset(
                'csv', data_files=[str(self.path_to_csv)], split="train"
            )
            .map(MoviePlotsDataset.select_genre_and_plot)
            .remove_columns(self.columns)
            .train_test_split(0.2, seed=self.seed)
        )
        return self

    def save_train_test_split(self) -> None:
        if self.dataset_dict is None:
            raise RuntimeError('Dataset not loaded')
        for split, dataset in self.dataset_dict.items():
            dataset.save_to_disk(str(self.path_to_data / f'{split}.hf'))


if __name__ == '__main__':
    (MoviePlotsDataset().load().save_train_test_split())
