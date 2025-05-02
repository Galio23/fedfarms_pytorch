import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_shape, output_shape,
                 conv1_filters=64, conv2_filters=64, kernel_size=5, use_pooling=False):
        """
        A flexible 1D CNN for multi-output regression.

        Args:
            input_shape (tuple): (timesteps, channels)
            output_shape (int): Number of regression outputs.
        """
        super(CNN1D, self).__init__()
        self.input_length = input_shape[0]
        self.input_channels = input_shape[1]
        self.use_pooling = use_pooling

        # Compute padding to maintain output length == input length
        pad = (kernel_size - 1) // 2

        # First convolutional block with 'same' padding
        self.conv1 = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=conv1_filters,
            kernel_size=kernel_size,
            padding=pad
        )
        if self.use_pooling:
            self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second convolutional block with 'same' padding
        self.conv2 = nn.Conv1d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=kernel_size,
            padding=pad
        )
        if self.use_pooling:
            self.pool2 = nn.MaxPool1d(kernel_size=2)

        # After two 'same' convs, length remains input_length
        length_after_conv = self.input_length
        # If pooling, each pool halves the length
        if self.use_pooling:
            length_after_conv = length_after_conv // 2  # after pool1
            length_after_conv = length_after_conv // 2  # after pool2

        flattened_size = conv2_filters * length_after_conv

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, output_shape)

    def forward(self, x):
        # Accept x as (batch, timesteps) or (batch, timesteps, channels)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, timesteps, 1)
        # Rearrange to (batch, channels, timesteps)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        if self.use_pooling:
            x = self.pool1(x)
        x = F.relu(self.conv2(x))
        if self.use_pooling:
            x = self.pool2(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(input_shape, output_shape):
    """
    Instantiate the CNN1D model.

    Args:
        input_shape (tuple): (timesteps, channels)
        output_shape (int): Number of regression outputs.
    """
    return CNN1D(input_shape, output_shape)
