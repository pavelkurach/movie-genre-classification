from pathlib import Path

import fire
from hydra import compose, initialize
from movie_genre_classification.data_loader import MoviePlotsDataset
from movie_genre_classification.genre_classifier import GenreClassifier
from transformers import TrainingArguments


def train(cloud: bool = False) -> None:
    overrides = ["training_arguments=cloud"] if cloud else []
    cfg = compose(config_name="config", overrides=overrides)

    path_to_models = (Path("") / "models").resolve()
    pretrained_model_name = cfg.pretrained_model_name

    args = TrainingArguments(
        str(path_to_models / f"{pretrained_model_name}-finetuned"),
        **cfg.training_arguments.hf_trainer,
    )

    split_dataset = MoviePlotsDataset().load()

    genre_classifier = GenreClassifier(pretrained_model_name, 5)
    genre_classifier.train(
        split_dataset, args, cfg.training_arguments.train_classifier_layer_only
    )


if __name__ == "__main__":
    initialize(version_base=None, config_path="conf")
    fire.Fire()
