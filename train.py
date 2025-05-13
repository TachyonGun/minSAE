import argparse
import csv
import time  # for timing eval windows
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception as e:
    print(f"Warning: wandb import failed: {e}")
    _WANDB_AVAILABLE = False
import torch
from embeddings_dataset import EmbeddingDataset
from sparse_autoencoder import SparseAutoencoder
from losses import SparseAECriterion

def parse_args():
    parser = argparse.ArgumentParser(description="Train sparse autoencoder on stored embeddings.")
    parser.add_argument("--num_steps", type=int, default=1072 * 10,
                        help="Number of training steps (batches) to run.")
    parser.add_argument("--number_of_arrays", type=int, default=1,
                        help="Number of .npy embedding arrays to load per sample.")
    parser.add_argument("--feature_dim_factor", type=float, default=8,
                        help="Factor to multiply input dimension to get feature_dim (feature_dim = factor * input_dim)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer.")
    parser.add_argument("--lambda_sparsity", type=float, default=0.01,
                        help="Coefficient for the sparsity penalty (lambda).")
    parser.add_argument("--recon_weight", type=float, default=1.0,
                        help="Weight for the reconstruction loss term.")
    parser.add_argument("--eval_window", type=int, default=50,
                        help="Number of steps between evaluations.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Initialize Weights & Biases (if available)
    if _WANDB_AVAILABLE:
        wandb.init(
            project="minSAE",
            name=f"ns{args.num_steps}_arr{args.number_of_arrays}_fdf{args.feature_dim_factor}_lr{args.lr}_lam{args.lambda_sparsity}_rw{args.recon_weight}",
            config=vars(args),
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Dataset setup
    dataset = EmbeddingDataset('embeddings/store/l-1', number_of_arrays=args.number_of_arrays)
    print(f"Dataset length (#samples): {len(dataset)}")

    # 2. Infer dimensions from first sample
    sample0 = dataset[0]
    tokens_per_step, input_dim = sample0.shape
    print(f"Tokens per step: {tokens_per_step}, input_dim: {input_dim}")

    # 3. Compute feature dimension
    feature_dim = int(args.feature_dim_factor * input_dim)
    print(f"Feature dimension: {feature_dim}")

    # 4. Model, loss, optimizer
    model = SparseAutoencoder(input_dim, feature_dim).to(device)
    criterion = SparseAECriterion(lambda_sparse=args.lambda_sparsity,
                                  recon_weight=args.recon_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5. Metrics and histogram
    loss_history = []
    recon_history = []
    l1_history = []
    nonzero_history = []
    activation_hist = torch.zeros(feature_dim, dtype=torch.long)

    eval_window = args.eval_window
    # Prepare CSV for activation frequencies
    csvfile = open("act_freqs.csv", "w", newline="")
    writer = csv.writer(csvfile)
    header = ["step"] + [f"feat_{i}" for i in range(feature_dim)]
    writer.writerow(header)
    # Track timing for evaluation windows
    last_eval_time = time.time()

    # 6. Training loop
    for step in range(args.num_steps):
        x = dataset[step  % len(dataset)].to(device)
        x_hat, features = model(x)

        total_loss, recon_loss, sparsity_loss = criterion(
            x, x_hat, features, model.decoder.weight
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Record metrics
        loss_history.append(total_loss.item())
        recon_history.append(recon_loss.item())
        l1_history.append(sparsity_loss.item())
        # Average number of active features per sample
        avg_active_per_sample = (features > 0).sum(dim=1).float().mean().item()
        nonzero_history.append(avg_active_per_sample)
        # Update per-feature activation histogram
        batch_nonzero_per_feature = (features > 0).sum(dim=0).detach().cpu()
        activation_hist += batch_nonzero_per_feature

        # Evaluation at intervals
        if (step + 1) % eval_window == 0:
            # Compute timing and ETA
            now = time.time()
            window_time = now - last_eval_time
            last_eval_time = now
            steps_remaining = args.num_steps - (step + 1)
            time_per_step = window_time / eval_window
            eta_seconds = time_per_step * steps_remaining
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            # Rolling averages over last eval_window steps
            recent_recons = recon_history[-eval_window:]
            avg_mse = sum(recent_recons) / len(recent_recons)
            recent_l1s = l1_history[-eval_window:]
            avg_l1 = sum(recent_l1s) / len(recent_l1s)
            recent_nonzeros = nonzero_history[-eval_window:]
            avg_nonzero = sum(recent_nonzeros) / len(recent_nonzeros)
            processed_tokens = (step + 1) * tokens_per_step
            print(f"[Step {step+1}/{args.num_steps}] "
                  f"Processed {processed_tokens} tokens; "
                  f"Avg MSE (last {eval_window}): {avg_mse:.6f}; "
                  f"Avg L1 (last {eval_window}): {avg_l1:.6f}; "
                  f"Avg active features/sample: {avg_nonzero:.2f}; "
                  f"Window time: {window_time:.2f}s; ETA: {eta_str}")
            # Log to W&B
            if _WANDB_AVAILABLE:
                wandb.log({
                    "avg_mse": avg_mse,
                    "avg_l1": avg_l1,
                    "avg_active_features": avg_nonzero
                }, step=step+1)
            # Write activation histogram to CSV and reset
            writer.writerow([processed_tokens] + activation_hist.tolist())
            csvfile.flush()
            activation_hist.zero_()

    # 7. Final histogram summary
    print("Final activation histogram per feature index:")
    print(activation_hist.tolist())
    # 7. Save model checkpoint
    save_path = "sparse_autoencoder_final.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model checkpoint to {save_path}")
    # 8. Close CSV and finish W&B
    csvfile.close()
    if _WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main() 