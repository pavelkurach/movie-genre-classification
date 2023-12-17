import os
from typing import Any

from datasets import DatasetDict
from optimum.onnxruntime import ORTModelForSequenceClassification
from psutil import cpu_count
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

from .lib.metrics.multi_label_metrics import compute_metrics
from .preprocessor import Preprocessor


os.environ["OMP_NUM_THREADS"] = f"{cpu_count() - 1}"
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class GenreClassifier:
    def __init__(
        self,
        pretrained_model_name: str,
        path_to_model: str,
        n_most_freq_genres: int = 15,
    ):
        self.pretrained_model_name = pretrained_model_name
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self._preprocessor = Preprocessor(n_most_freq_genres)
        self._model: PreTrainedModel | None = None
        self.path_to_model = path_to_model
        self._trainer: Trainer | None = None
        self._num_labels = n_most_freq_genres

    def train(
        self,
        split_dataset: DatasetDict,
        args: TrainingArguments,
        train_classifier_layer_only: bool = True,
        limit_train_dataset: float = 1.0,
        limit_test_dataset: float = 1.0,
    ) -> None:
        (
            preprocessed_dataset,
            id2label,
            label2id,
        ) = self._preprocessor.transform(split_dataset)

        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_model_name,
                problem_type="multi_label_classification",
                num_labels=self._num_labels,
                id2label=id2label,
                label2id=label2id,
            )

        if train_classifier_layer_only:
            for name, param in self._model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        preprocessed_dataset = preprocessed_dataset.map(
            lambda example: self._tokenizer(example["plot"], truncation=True),
            desc="Tokenize",
        )
        data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)
        train_len = len(preprocessed_dataset["train"])
        test_len = len(preprocessed_dataset["test"])

        self._trainer = Trainer(
            self._model,
            args,
            train_dataset=preprocessed_dataset["train"].select(
                range(int(train_len * limit_train_dataset))
            ),
            eval_dataset=preprocessed_dataset["test"].select(
                range(int(test_len * limit_test_dataset))
            ),
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
        self._static_predict(
            plot,
            self._model,
            self._tokenizer,
        )

    @staticmethod
    def _static_predict(
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
