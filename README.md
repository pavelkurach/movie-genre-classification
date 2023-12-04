# Movie genre classification

## Overview

Movie genre classification project is aimed at developing a model that predicts movie genre from its plot summary.
It is based on [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) dataset.

## Installation

This project requires Python 3.11.

1. Clone project:

```shell
git clone https://github.com/pavelkurach/movie-genre-classification.git &&
cd movie-genre-classification
```

2. Create python virtual environment:

```shell
python -m venv venv
```

3. If you don't have `poetry` installed, you can install it in previously created virtual environment

```shell
pip install poetry
```

4. Install project dependencies:

```shell
poetry install
```

## Data 

Pull data from DVC repository:

```shell
dvc pull
```

If you did not succeed to pull data using DVC, you can download it from Kaggle:

```shell
pip install kaggle &&
cd data &&
kaggle datasets download -d jrobischon/wikipedia-movie-plots &&
unzip wikipedia-movie-plots.zip &&
rm wikipedia-movie-plots.zip &&
cd ../
```

## Usage

### Train

To use the model, it should be train first, for that run:

```shell
python commands.py train
```

By default, the model trains in debug mode, on a small part data, with small number of epochs, and only on only a small part of the dataset.
To fully train the model, and `--cloud=true` to the previous command.

### Inference

To use the model for inference, run:

```shell
python commands.py run_server
```

You will be prompted to enter movie plot summary. It should not exceed 512 characters.
