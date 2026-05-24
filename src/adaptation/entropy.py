import torch
import torch.nn.functional as F


def softmax_entropy(logits):
    """
    Computes prediction entropy from model logits.

    Args:
        logits:
            Model outputs before softmax.
            Shape: (batch_size, num_classes)

    Returns:
        Entropy for each sample.
        Shape: (batch_size,)
    """

    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)

    entropy = -torch.sum(
        probabilities * log_probabilities,
        dim=1
    )

    return entropy