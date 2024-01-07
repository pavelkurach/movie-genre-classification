# type: ignore

from functools import lru_cache

import numpy as np
import torch
from transformers import AutoTokenizer
from tritonclient.http import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
)
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_ensemble(text: str):
    triton_client = get_client()
    text = np.array([text.encode("utf-8")], dtype=object)

    input_text = InferInput(
        name="texts", shape=text.shape, datatype=np_to_triton_dtype(text.dtype)
    )
    input_text.set_data_from_numpy(text, binary_data=True)

    infer_output = InferRequestedOutput("logits", binary_data=True)
    query_response = triton_client.infer(
        "ensemble", [input_text], outputs=[infer_output]
    )
    logits = query_response.as_numpy("logits")[0]
    return logits


def call_triton_tokenizer(text: str):
    triton_client = get_client()
    text = np.array([text.encode("utf-8")], dtype=object)

    input_text = InferInput(
        name="texts", shape=text.shape, datatype=np_to_triton_dtype(text.dtype)
    )
    input_text.set_data_from_numpy(text, binary_data=True)

    query_response = triton_client.infer(
        "tokenizer",
        [input_text],
        outputs=[
            InferRequestedOutput("input_ids", binary_data=True),
            InferRequestedOutput("attention_mask", binary_data=True),
        ],
    )
    input_ids = query_response.as_numpy("input_ids")[0]
    attention_massk = query_response.as_numpy("attention_mask")[0]
    return input_ids, attention_massk


def main():
    texts = [
        "Just a test text",
        "Another test text",
    ]
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-base")
    encoded = tokenizer(
        texts[0],
        padding="max_length",
        max_length=512,
        truncation=True,
    )
    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
    _input_ids, _attention_mask = call_triton_tokenizer(texts[0])
    assert (input_ids == _input_ids).all() and (
        attention_mask == _attention_mask
    ).all()

    logits = torch.tensor(
        [call_triton_ensemble(row).tolist() for row in texts]
    )
    print(logits)


if __name__ == "__main__":
    main()
