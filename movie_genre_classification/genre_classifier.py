import os
import typing
from typing import Any, List

import mlflow
from datasets import DatasetDict
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    pipeline,
)

from .genre_encoder import GenreEncoder
from .lib.metrics.multi_label_metrics import compute_metrics
from .preprocessor import Preprocessor


os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


class GenreClassifier:
    def __init__(
        self,
        pretrained_model_name: str,
        path_to_model: str,
        n_most_frequent_genres: int,
    ):
        self.pretrained_model_name = pretrained_model_name
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self._preprocessor = Preprocessor(self._tokenizer)
        self._genre_encoder = GenreEncoder(n_most_frequent_genres)
        self._model: PreTrainedModel | None = None
        self.path_to_model = path_to_model
        self._trainer: Trainer | None = None

    def train(
        self,
        split_dataset: DatasetDict,
        args: TrainingArguments,
        train_classifier_layer_only: bool = True,
        limit_train_dataset: float = 1.0,
        limit_test_dataset: float = 1.0,
    ) -> None:
        preprocessed_dataset = self._preprocessor.transform(split_dataset)

        self._genre_encoder.fit(preprocessed_dataset["train"])
        encoded_dataset = self._genre_encoder.transform(preprocessed_dataset)

        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_model_name,
                problem_type="multi_label_classification",
                num_labels=self._genre_encoder.get_num_labels(),
                id2label=self._genre_encoder.get_id2label(),
                label2id=self._genre_encoder.get_label2id(),
            )

        if train_classifier_layer_only:
            for name, param in self._model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)
        train_len = len(encoded_dataset["train"])
        test_len = len(encoded_dataset["test"])

        self._trainer = Trainer(
            self._model,
            args,
            train_dataset=encoded_dataset["train"].select(
                range(int(train_len * limit_train_dataset))
            ),
            eval_dataset=encoded_dataset["test"].select(
                range(int(test_len * limit_test_dataset))
            ),
            tokenizer=self._tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        self._trainer.train()
        self._trainer.save_model(f"{self.path_to_model}/best")

    def evaluate(self) -> Any:
        if self._trainer is None:
            raise RuntimeError("Train the model first")
        return self._trainer.evaluate()

    def load(self, path: str) -> None:
        self._model = AutoModelForSequenceClassification.from_pretrained(path)

    def get_config(self) -> PretrainedConfig:
        if self._model is None:
            raise RuntimeError("Train the model first")
        return self._model.config

    def save_onnx(self) -> str:
        if self._trainer is None:
            raise RuntimeError("Train the model first")
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            f"{self.path_to_model}/best", export=True
        )
        onnx_model.save_pretrained(f"{self.path_to_model}/onnx")
        return f"{self.path_to_model}/onnx"

    def predict(self, plot: str) -> Any:
        if self._model is None:
            raise RuntimeError("Train the model first")
        if len(plot) > 512:
            raise ValueError(
                "Model accept plots not longer than 512 characters"
            )
        self.static_predict(
            plot,
            self._model,
            self._tokenizer,
        )

    @staticmethod
    def static_predict(
        plot: str, model: Any, tokenizer: PreTrainedTokenizerFast
    ) -> Any:
        classifier = pipeline(
            "text-classification",
            model=model.to("cpu"),
            tokenizer=tokenizer,
            top_k=None,
        )
        return GenreClassifier._process_pipeline_output(classifier(plot))

    @staticmethod
    def _process_pipeline_output(
        output: list[list[dict[str, Any]]], threshold: float = 0.5
    ) -> list[str]:
        preds = output[0]
        min_score, max_score = 1, 0
        for pred in preds:
            if pred["score"] > max_score:
                max_score = pred["score"]
            if pred["score"] < min_score:
                min_score = pred["score"]

        genres = []
        for pred in preds:
            score_normalized = (pred["score"] - min_score) / (
                max_score - min_score
            )
            if score_normalized > threshold:
                genres.append(pred["label"])

        return genres


class GenreClassifierOnnx(mlflow.pyfunc.PythonModel):
    def __init__(self, path: str):
        super().__init__()
        self._model = ORTModelForSequenceClassification.from_pretrained(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)

    def load_context(
        self, context: mlflow.pyfunc.model.PythonModelContext
    ) -> None:
        print(context.artifacts)

    @typing.no_type_check
    def predict(self, model_inputs: List[str]) -> List[str]:
        return GenreClassifier.static_predict(
            model_inputs[0], self._model, self._tokenizer
        )
