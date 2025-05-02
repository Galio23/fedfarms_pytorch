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

        # First convolutional block
        self.conv1 = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=conv1_filters,
            kernel_size=kernel_size
        )
        if self.use_pooling:
            self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second convolutional block
        self.conv2 = nn.Conv1d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=kernel_size
        )
        if self.use_pooling:
            self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Compute flattened dimension
        if self.use_pooling:
            def conv_pool_length(l, k, p):
                return (l - k + 1) // p
            l1 = conv_pool_length(self.input_length, kernel_size, 2)
            l2 = conv_pool_length(l1, kernel_size, 2)
        else:
            l1 = self.input_length - kernel_size + 1
            l2 = l1 - kernel_size + 1
        flattened_size = conv2_filters * l2

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, output_shape)

    def forward(self, x):
        # Accept x as (batch, timesteps) or (batch, timesteps, channels)
        if x.dim() == 2:
            # add channel dimension
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
    model = CNN1D(input_shape, output_shape)
    return model

    


# # Example usage:
# if __name__ == "__main__":
#     # Input: 10 timesteps, 1 channel; Output: 2 targets (Clay and Carbon)
#     input_shape = (10, 1)
#     output_shape = 2
#     model = create_model(input_shape, output_shape)
#     dummy_input = torch.randn(8, *input_shape)  # Batch of 8 samples
#     output = model(dummy_input)
#     print("Output shape:", output.shape)  # Expect (8, 2)

