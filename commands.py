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


def train(cloud: bool = False, log_model: bool = False) -> None:
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
        experiment_name=cfg.mlflow_experiment_name,
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
        onnx_model = onnx.load_model(str(f"{onnx_model_path}/model.onnx"))
        mlflow.onnx.log_model(onnx_model, cfg.mlflow_model_path)
        mlflow.log_artifact(
            f"{onnx_model_path}/config.json", cfg.mlflow_model_path
        )
        mlflow.log_artifact(
            f"{onnx_model_path}/tokenizer",
            f"{cfg.mlflow_model_path}/tokenizer",
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
    path_to_model = path_to_model = str(
        (
            Path(".") / "models" / f"{cfg.pretrained_model_name}-finetuned"
        ).resolve()
        / "onnx"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        path_to_model + "/tokenizer/", local_files_only=True
    )

    ort_session = ort.InferenceSession(path_to_model + "/model.onnx")
    print("Summarize the plot:")
    plot = input()

    def prepare_for_session(input_to_tokenize: str) -> dict[str, np.ndarray]:
        tokens = tokenizer(input_to_tokenize, return_tensors="np")
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

    print(ort_session.run(None, prepare_for_session(plot)))


if __name__ == "__main__":
    initialize(version_base=None, config_path="conf")
    fire.Fire()
