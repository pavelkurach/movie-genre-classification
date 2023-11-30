import os
from typing import Any

from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
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
        self, pretrained_model_name: str, n_most_frequent_genres: int
    ):
        self.pretrained_model_name = pretrained_model_name
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self._preprocessor = Preprocessor(self._tokenizer)
        self._genre_encoder = GenreEncoder(n_most_frequent_genres)
        self._model: Any | None = None
        self._trainer: Trainer | None = None

    def train(
        self,
        split_dataset: DatasetDict,
        args: TrainingArguments,
        train_classifier_layer_only: bool = True,
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
        self._trainer = Trainer(
            self._model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["test"],
            tokenizer=self._tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        self._trainer.train()

    def evaluate(self) -> Any:
        if self._trainer is None:
            raise RuntimeError("Train the model first")
        return self._trainer.evaluate()

    def predict(self, plot: str) -> list[Any] | Any:
        if self._model is None:
            raise RuntimeError("Train the model first")
        if len(plot) > 512:
            raise ValueError(
                "Model accept plots not longer than 512 characters"
            )
        classifier = pipeline(
            "text-classification",
            model=self._model.to("cpu"),
            tokenizer=self._tokenizer,
            top_k=None,
        )
        return classifier(plot)
