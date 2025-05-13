import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with a single hidden layer of ReLU features.

    Args:
        input_dim (int): Dimension D of input activations.
        feature_dim (int): Number F of sparse features.

    Architecture:
        Encoder: Linear(in_features=D, out_features=F) + ReLU
        Decoder: Linear(in_features=F, out_features=D)
    """
    def __init__(self, input_dim: int, feature_dim: int):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim  # D
        self.feature_dim = feature_dim  # F
        # Encoder: maps from D -> F
        self.encoder = nn.Linear(input_dim, feature_dim, bias=True)
        # Decoder: maps from F -> D
        self.decoder = nn.Linear(feature_dim, input_dim, bias=True)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the sparse autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, D)

        Returns:
            x_hat (torch.Tensor): Reconstructed tensor of shape (batch_size, D)
            features (torch.Tensor): Feature activations of shape (batch_size, F)
        """
        # Encode: linear transform to features
        # x:      (batch_size, D)
        # features_pre_relu: (batch_size, F)
        features_pre_relu = self.encoder(x)
        # Apply ReLU nonlinearity
        # features: (batch_size, F)
        features = F.relu(features_pre_relu)

        # Decode: linear transform back to input space
        # x_hat: (batch_size, D)
        x_hat = self.decoder(features)

        return x_hat, features 