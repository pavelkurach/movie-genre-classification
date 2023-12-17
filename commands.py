from pathlib import Path

import fire
import mlflow
import numpy as np
import onnx
import onnxruntime as ort
from hydra import compose, initialize
from movie_genre_classification.data_loader import MoviePlotsDataset
from movie_genre_classification.genre_classifier import GenreClassifier
from omegaconf import OmegaConf
from transformers import AutoTokenizer, TrainingArguments


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

    onnx_model_path = genre_classifier.save_onnx()

    onnx_model = onnx.load_model(str(onnx_model_path))
    mlflow.onnx.save_model(onnx_model, cfg.mlflow_model_path)


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
        model = mlflow.onnx.load_model(cfg.mlflow_model_path)
    except OSError:
        print("Run `python commands.py train` first")
        return
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name)
    ort_session = ort.InferenceSession(model.SerializeToString())
    print("Summarize the plot:")
    plot = input()

    def prepare_for_session(input_to_tokenize: str) -> dict[str, np.ndarray]:
        tokens = tokenizer(input_to_tokenize, return_tensors="pt")
        return {k: v.cpu().detach().numpy() for k, v in tokens.items()}

    print(ort_session.run(None, prepare_for_session(plot)))


if __name__ == "__main__":
    initialize(version_base=None, config_path="conf")
    experiment_name = "Movie genre classification (wiki)"
    fire.Fire()
