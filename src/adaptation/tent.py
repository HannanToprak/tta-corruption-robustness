import torch
import torch.nn as nn
import torch.optim as optim

from src.adaptation.entropy import softmax_entropy
from src.adaptation.bn_utils import (
    configure_model_for_tent,
    collect_bn_params
)


class Tent:
    """
    TENT:
    Fully Test-Time Adaptation by Entropy Minimization.

    This implementation:
        - updates only BatchNorm affine parameters
        - minimizes prediction entropy at test time
        - performs online adaptation
    """

    def __init__(
        self,
        model,
        lr=1e-3
    ):
        """
        Args:
            model: neural network model
            lr: adaptation learning rate
        """

        self.model = configure_model_for_tent(model)

        params, param_names = collect_bn_params(self.model)

        self.optimizer = optim.Adam(
            params,
            lr=lr
        )

        print("\nTENT initialized.")
        print(f"Number of BN parameters: {len(params)}")

    def forward_and_adapt(self, images):
        """
        Performs one TENT adaptation step.

        Args:
            images:
                test batch

        Returns:
            outputs:
                model predictions after adaptation

            entropy_loss:
                mean entropy loss
        """

        outputs = self.model(images)

        entropy = softmax_entropy(outputs)

        loss = entropy.mean()

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return outputs, loss

    def predict(self, images):
        """
        Performs prediction with adaptation.

        Args:
            images:
                input batch

        Returns:
            outputs:
                adapted predictions
        """

        outputs, _ = self.forward_and_adapt(images)

        return outputs