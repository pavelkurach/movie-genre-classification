from pathlib import Path

from datasets import load_from_disk
from preprocess import Preprocessor


if __name__ == '__main__':
    path_to_data = (Path('..') / 'data').resolve()
    path_to_train = path_to_data / 'train.hf'
    train_dataset = load_from_disk(str(path_to_train))
    train_preprocessed = Preprocessor(
        split_dataset=train_dataset,
        split='train',
        tokenizer='distilbert-base-uncased',
    ).preprocess()
    print(train_preprocessed[1])
