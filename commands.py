from pathlib import Path

import fire
import mlflow
from hydra import compose, initialize
from movie_genre_classification.data_loader import MoviePlotsDataset
from movie_genre_classification.genre_classifier import GenreClassifier
from omegaconf import OmegaConf
from transformers import TrainingArguments


def train(cloud: bool = False) -> None:
    overrides = ["training_arguments=cloud"] if cloud else []
    cfg = compose(config_name="config", overrides=overrides)

    pretrained_model_name = cfg.pretrained_model_name
    path_to_model = str(
        (Path(".") / "models" / f"{pretrained_model_name}-finetuned").resolve()
    )

    args = TrainingArguments(
        output_dir=path_to_model,
        **cfg.training_arguments.hf_trainer,
    )

    split_dataset = MoviePlotsDataset(seed=cfg.seed).load()

    genre_classifier = GenreClassifier(
        pretrained_model_name, path_to_model, cfg.num_genres
    )

    experiment = mlflow.set_experiment(
        experiment_name="Movie genre classification (wiki)"
    )
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_dict(
            OmegaConf.to_container(cfg, throw_on_missing=True), "config.yaml"
        )
        genre_classifier.train(
            split_dataset,
            args,
            cfg.training_arguments.train_classifier_layer_only,
            cfg.training_arguments.limit_train_dataset,
            cfg.training_arguments.limit_test_dataset,
        )

    # using onnx is required, but it doesn't make sense here
    path_to_onnx_model = genre_classifier.save_onnx()

    mlflow.pyfunc.save_model(
        cfg.mlflow_model_path,
        data_path=path_to_onnx_model,
        loader_module=cfg.mlflow_loader_path,
    )


def predict(plot: str) -> None:
    overrides: list[str] = []
    cfg = compose(config_name="config", overrides=overrides)

    pretrained_model_name = cfg.pretrained_model_name
    path_to_model = str(
        (Path(".") / "models" / f"{pretrained_model_name}-finetuned").resolve()
    )

    genre_classifier = GenreClassifier(
        pretrained_model_name, str(path_to_model), cfg.num_genres
    )
    genre_classifier.load(path_to_model)
    print(genre_classifier.predict(plot))


def run_server() -> None:
    overrides: list[str] = []
    cfg = compose(config_name="config", overrides=overrides)
    try:
        model = mlflow.pyfunc.load_model(cfg.mlflow_model_path)
    except OSError:
        print("Run `python commands.py train` first")
        return
    while True:
        print("Summarize the plot:")
        plot = input()
        print("Most likely genres:", ", ".join(model.predict([plot])), "\n\n")


if __name__ == "__main__":
    initialize(version_base=None, config_path="conf")
    experiment_name = "Movie genre classification (wiki)"
    fire.Fire()
