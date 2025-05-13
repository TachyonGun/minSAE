import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAECriterion(nn.Module):
    """
    Sparse Autoencoder loss:
      L = recon_weight * E[||x - x_hat||_2^2]
        + lambda_sparse * E[sum_i f_i(x) * ||W_dec[:, i]||_2]

    Args:
        lambda_sparse (float): coefficient for the sparsity penalty (λ).
        recon_weight (float): coefficient for the reconstruction term.
        reduction (str): 'mean', 'sum', or 'none' for how to reduce losses over the batch.
        p_norm (int): norm degree for decoder weight columns (default 2).
    """
    def __init__(self,
                 lambda_sparse: float = 1e-2,
                 recon_weight: float = 1.0,
                 reduction: str = 'mean',
                 p_norm: int = 2):
        super(SparseAECriterion, self).__init__()
        self.lambda_sparse = lambda_sparse
        self.recon_weight = recon_weight
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.p_norm = p_norm

    def forward(self,
                x: torch.Tensor,
                x_hat: torch.Tensor,
                features: torch.Tensor,
                decoder_weight: torch.Tensor):
        """
        Compute the sparse AE loss.

        Args:
            x (Tensor): original inputs, shape (batch_size, D)
            x_hat (Tensor): reconstructions, shape (batch_size, D)
            features (Tensor): activations post-ReLU, shape (batch_size, F)
            decoder_weight (Tensor): weight matrix of decoder, shape (D, F)

        Returns:
            total_loss (Tensor): scalar or per-sample loss depending on reduction.
            recon_loss (Tensor): reconstruction term.
            sparsity_loss (Tensor): sparsity penalty term.
        """
        # Reconstruction loss: MSE ||x - x_hat||_2^2
        recon_loss = F.mse_loss(x_hat, x, reduction=self.reduction)

        # Compute L2 norm of each decoder weight column: ||W_dec[:, i]||_p_norm
        # decoder_weight: (D, F) --> weight_norms: (F,)
        weight_norms = decoder_weight.norm(p=self.p_norm, dim=0)

        # Sparsity penalty per sample: sum_i features[n,i] * weight_norms[i]
        # features: (batch_size, F)
        per_sample_penalty = (features * weight_norms).sum(dim=1)

        # Reduce over batch
        if self.reduction == 'mean':
            sparsity_loss = per_sample_penalty.mean()
        elif self.reduction == 'sum':
            sparsity_loss = per_sample_penalty.sum()
        else:  # 'none'
            sparsity_loss = per_sample_penalty

        sparsity_loss = self.lambda_sparse * sparsity_loss

        # Weighted sum of both terms
        total_loss = self.recon_weight * recon_loss + sparsity_loss

        return total_loss, recon_loss, sparsity_loss 