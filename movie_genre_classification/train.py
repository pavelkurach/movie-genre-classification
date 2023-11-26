# type: ignore

from pathlib import Path

import numpy as np
import torch
from encode_genres import GenreEncoder
from load_and_split import MoviePlotsDataset
from preprocess import Preprocessor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    pipeline,
)


batch_size = 32
metric_name = "f1"

path_to_models = (Path('..') / 'models').resolve()

args = TrainingArguments(
    str(path_to_models / 'distilbert-finetuned'),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)

pretrained_model_name = 'distilbert-base-uncased'


def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    return {'f1': f1_micro_average, 'roc_auc': roc_auc, 'accuracy': accuracy}


def compute_metrics(p: EvalPrediction):
    preds = (
        p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    )
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


if __name__ == '__main__':
    split = MoviePlotsDataset().load()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    preprocessor = Preprocessor(
        split_dataset_dict=split, path_to_split=None, tokenizer=tokenizer
    )
    preprocessed = preprocessor.preprocess()

    genre_encoder = GenreEncoder(
        preprocessed_dataset_dict=preprocessed,
        path_to_preprocessed=None,
        n_most_freq_genres=20,
    )
    genre_encoder.train()
    encoded = genre_encoder.encode()
    print(encoded)

    data_collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        problem_type='multi_label_classification',
        num_labels=genre_encoder.get_num_labels(),
        id2label=genre_encoder.get_id2label(),
        label2id=genre_encoder.get_label2id(),
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded['train'].select(range(320)),
        eval_dataset=encoded['test'].select(range(32)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

    classifier = pipeline(
        'text-classification',
        model=model.to('cpu'),
        tokenizer=tokenizer,
        top_k=None,
    )

    example = encoded['train'][0]
    print(example['plot'])
    print(example['genres'])
    print(classifier(example['plot'][:512]))
