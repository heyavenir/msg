import re
import string
from collections import Counter


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    gt_tokens = normalize(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize(prediction) == normalize(ground_truth) else 0.0
