import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import EvalPrediction


def multi_label_metrics(
    predictions: np.ndarray | tuple[np.ndarray],
    labels: np.ndarray | tuple[np.ndarray],
    threshold: float = 0.5,
) -> dict[str, float]:
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    max_ = torch.max(probs, dim=-1).values.unsqueeze(-1)
    min_ = torch.min(probs, dim=-1).values.unsqueeze(-1)
    probs = (probs - min_) / (max_ - min_)
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    return {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}


def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    preds = (
        p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    )
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result
