import argparse
import os
import yaml
import time
import numpy as np
import torch

from gpatchtst.data import get_tuh_dataloaders
from gpatchtst.models.patchtst.patchTST import PatchTST
from gpatchtst.models.patchtst.layers.revin import RevIN

# Global cache for activations
ACT_CACHE = None

def hook_fn(module, input, output):
    """Hook function to capture the layer output."""
    global ACT_CACHE
    try:
        ACT_CACHE = output[0]
    except Exception:
        ACT_CACHE = output

def create_patches(xb, patch_len, stride):
    """
    Split input sequence into overlapping patches.
    Args:
        xb: Tensor of shape (batch_size, seq_len, n_vars)
        patch_len: length of each patch
        stride: step size between patches

    Returns:
        patches: Tensor of shape (batch_size, num_patches, n_vars, patch_len)
        num_patches: int
    """
    seq_len = xb.shape[1]
    num_patches = (seq_len - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patches - 1)
    s_begin = seq_len - tgt_len
    xb = xb[:, s_begin:, :]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)
    return xb, num_patches

def parse_args():
    parser = argparse.ArgumentParser(description="Compute embeddings via pretrained PatchTST model.")
    parser.add_argument("--root_path", type=str, required=True,
                        help="Path to preprocessed data directory.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint .pth file.")
    parser.add_argument("--cache_layer", type=int, default=-1,
                        help="Layer index to hook for embedding extraction.")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Maximum number of batches to process (default: all).")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    model_cfg = cfg['model']

    # Override root and CSV paths
    data_cfg['root_path'] = args.root_path
    data_cfg['csv_path'] = os.path.join(args.root_path, 'file_lengths_map.csv')

    # Instantiate data loaders
    train_loader, _, _ = get_tuh_dataloaders(
        data_cfg['root_path'],
        data_cfg.get('data_path'),
        data_cfg['csv_path'],
        batch_size=data_cfg.get('batch_size', 128),
        num_workers=data_cfg.get('num_workers', 1),
        prefetch_factor=data_cfg.get('prefetch_factor', 2),
        pin_memory=data_cfg.get('pin_memory', False),
        drop_last=data_cfg.get('drop_last', False),
        size=[model_cfg['seq_len'], model_cfg['target_dim'], model_cfg['patch_length']]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Compute patch parameters
    seq_len = model_cfg['seq_len']
    patch_length = model_cfg['patch_length']
    stride = model_cfg['stride']
    num_patches = seq_len // patch_length

    # Initialize model and RevIN
    model = PatchTST(
        c_in=data_cfg['n_vars'],
        target_dim=model_cfg['target_dim'],
        patch_len=patch_length,
        stride=stride,
        num_patch=num_patches,
        n_layers=model_cfg['num_layers'],
        d_model=model_cfg['d_model'],
        n_heads=model_cfg['num_heads'],
        shared_embedding=model_cfg['shared_embedding'],
        d_ff=model_cfg['d_ff'],
        norm=model_cfg['norm'],
        attn_dropout=model_cfg['attn_dropout'],
        dropout=model_cfg['dropout'],
        act=model_cfg['activation'],
        res_attention=model_cfg['res_attention'],
        pe=model_cfg['pe'],
        learn_pe=model_cfg['learn_pe'],
        head_dropout=model_cfg['head_dropout'],
        head_type=model_cfg['head_type'],
        use_cls_token=True
    ).to(device)

    revin = RevIN(
        data_cfg['n_vars'],
        float(model_cfg.get('revin_eps', 1e-5)),
        bool(model_cfg.get('revin_affine', False))
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if 'revin_state_dict' in ckpt and ckpt['revin_state_dict'] is not None:
        revin.load_state_dict(ckpt['revin_state_dict'])
    else:
        print("RevIN state dict missing or None; skipping RevIN load.")
    print(f"Loaded checkpoint from {args.checkpoint_path}")

    # Register forward hook for embedding extraction
    model.backbone.encoder.layers[args.cache_layer].register_forward_hook(hook_fn)

    # Prepare output directory
    out_dir = os.path.join('embeddings', 'store', f'l{args.cache_layer}')
    os.makedirs(out_dir, exist_ok=True)

    total_batches = args.max_batches if args.max_batches is not None else len(train_loader)
    print(f"Processing {total_batches} batches...")

    # Iterate and extract embeddings
    for i, batch in enumerate(train_loader):
        if args.max_batches is not None and i >= args.max_batches:
            break
        t0 = time.time()

        # Normalize inputs
        data = batch['past_values'].to(device)
        data = revin(data, mode='norm')
        print(f"Batch {i} of {total_batches}")

        # Create patches
        t1 = time.time()
        input_patches, _ = create_patches(data, patch_length, stride)
        print("Created patches")

        # Forward pass
        with torch.no_grad():
            _ = model(input_patches)
        print("Got prediction")

        # Retrieve and save embeddings
        embedding = ACT_CACHE
        print("Got embedding")
        embedding_np = embedding.detach().cpu().numpy()
        filename = f'b{i:04d}.npy'
        save_path = os.path.join(out_dir, filename)
        np.save(save_path, embedding_np)
        print(f"Saved embedding batch {i:04d} of {total_batches}")

        # Timing and ETA
        total_time = time.time() - t0
        eta = total_time * (total_batches - i)
        print(f"Total time: {total_time:.4f} seconds, ETA: {eta:.4f} seconds")

    print("Done.")

if __name__ == '__main__':
    main() 