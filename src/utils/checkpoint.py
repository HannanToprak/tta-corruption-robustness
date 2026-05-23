import torch


def load_checkpoint(checkpoint_path, model, device):
    """
    Loads a saved model checkpoint.

    Args:
        checkpoint_path: path to checkpoint file
        model: initialized model architecture
        device: cuda or cpu

    Returns:
        model with loaded weights
        checkpoint dictionary
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint