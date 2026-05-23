import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network for CIFAR-10 classification.

    Input:
        images with shape (batch_size, 3, 32, 32)

    Output:
        logits with shape (batch_size, 10)
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Feature extractor:
        # Learns spatial image features using convolution, normalization,
        # non-linearity, and downsampling.
        self.features = nn.Sequential(
            # Input: 3 x 32 x 32
            # Output: 32 x 32 x 32
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Output: 32 x 16 x 16
            nn.MaxPool2d(kernel_size=2),

            # Input: 32 x 16 x 16
            # Output: 64 x 16 x 16
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Output: 64 x 8 x 8
            nn.MaxPool2d(kernel_size=2),

            # Input: 64 x 8 x 8
            # Output: 128 x 8 x 8
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Output: 128 x 4 x 4
            nn.MaxPool2d(kernel_size=2)
        )

        # Classifier:
        # Converts learned feature maps into class predictions.
        self.classifier = nn.Sequential(
            # 128 channels * 4 height * 4 width = 2048 features
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x: input image batch with shape (batch_size, 3, 32, 32)

        Returns:
            logits with shape (batch_size, num_classes)
        """

        # Extract convolutional feature maps.
        x = self.features(x)

        # Flatten feature maps:
        # From (batch_size, 128, 4, 4)
        # To   (batch_size, 2048)
        x = torch.flatten(x, start_dim=1)

        # Produce class logits.
        x = self.classifier(x)

        return x