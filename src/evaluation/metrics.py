import torch


def accuracy(outputs, targets):
    """
    Computes classification accuracy.

    Args:
        outputs:
            Model logits with shape (batch_size, num_classes)

        targets:
            Ground-truth labels with shape (batch_size)

    Returns:
        Accuracy score between 0 and 1.
    """

    # Get predicted class index.
    _, predictions = torch.max(outputs, dim=1)

    # Count correctly classified samples.
    correct = (predictions == targets).sum().item()

    # Total number of samples.
    total = targets.size(0)

    # Compute accuracy.
    acc = correct / total

    return acc