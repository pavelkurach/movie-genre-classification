from pathlib import Path

from datasets import load_from_disk
from encode_genres import GenreEncoder
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
    genre_encoder = GenreEncoder(train_preprocessed, n_most_freq_genres=20)
    train_encoded = genre_encoder.encode()
    print(train_encoded[0]['genres'])
    print(
        [
            genre_encoder.get_id2label()[idx]
            for idx, label in enumerate(train_encoded[0]['genres_enc'])
            if label == 1.0
        ]
    )
