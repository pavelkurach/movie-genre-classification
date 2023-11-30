from pathlib import Path

from movie_genre_classification.data_loader import MoviePlotsDataset
from movie_genre_classification.genre_classifier import GenreClassifier
from transformers import TrainingArguments


pretrained_model_name = "distilbert-base-uncased"

path_to_models = (Path("") / "models").resolve()

args = TrainingArguments(
    str(path_to_models / f"{pretrained_model_name}-finetuned"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
)


def train() -> None:
    split_dataset = MoviePlotsDataset().load()

    genre_classifier = GenreClassifier(pretrained_model_name, 15)
    genre_classifier.train(split_dataset, args)

    example = split_dataset["train"][0]
    print(example["plot"])
    print(example["genre"])
    print(genre_classifier.predict(example["plot"][:512]))


if __name__ == "__main__":
    train()
