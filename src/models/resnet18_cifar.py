import torch.nn as nn
import torchvision.models as models


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 classification.
    """

    def __init__(self, num_classes=10):
        super(ResNet18CIFAR, self).__init__()

        # Load standard ResNet-18 architecture.
        self.model = models.resnet18(weights=None)

        # CIFAR-10 images are 32x32,
        # so we replace the large ImageNet stem.
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3, #it was 7
            stride=1, #was 2 for imagenet dataset
            padding=1,
            bias=False
        )

        # Remove aggressive downsampling.
        self.model.maxpool = nn.Identity()

        # Replace classifier head.
        self.model.fc = nn.Linear(
            in_features=512,
            out_features=num_classes
        )

    def forward(self, x):
        return self.model(x)