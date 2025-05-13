import matplotlib.pyplot as plt
import torch
import os 

def create_mask(patches, mask_ratio, independent_channel_masking=False, 
                mask_type='random', forecasting_num_patches=None, fixed_position=None):
    bs, num_patches, n_vars, patch_len = patches.shape
    num_masked_patches = int(num_patches * mask_ratio)

    mask = torch.zeros((bs, num_patches, n_vars), dtype=torch.bool).to(patches.device, non_blocking=True) #CHANGES

    if mask_type == 'random':
        if independent_channel_masking:
            # Generate a mask for each channel
            for i in range(n_vars):
                mask_indices = torch.randperm(num_patches)[:num_masked_patches]
                mask[:, mask_indices, i] = True #CHANGES
        else:
            # Generate a mask for each patch
            mask_indices = torch.randperm(num_patches)[:num_masked_patches]
            mask[:, mask_indices, :] = True #CHANGES
    elif mask_type == 'forecasting': # or continous towards the end
        if forecasting_num_patches is None:
            raise ValueError("forecasting_num_patches must be provided for forecasting masking.")
        else:
            mask[:, -forecasting_num_patches:, :] = True #CHANGES
    elif mask_type == 'backcasting': # or continous towards the beginning
        if forecasting_num_patches is None:
            raise ValueError("forecasting_num_patches must be provided for backcasting masking.")
        else:
            mask[:, :forecasting_num_patches, :] = True
    elif mask_type == 'fixed_position':
        if fixed_position is None:
            raise ValueError("fixed_position must be provided for fixed position masking.")
        else:
            mask[:, fixed_position, :] = True #CHANGES
    else:
        raise ValueError("Invalid mask type. Choose 'random', 'forecasting', or 'fixed_position'.")

    return mask

def apply_mask(patches, mask, masked_value=0):
    # Apply the mask to the patches
    masked_patches = patches.clone()
    masked_patches[mask] = masked_value

    return masked_patches # masked_patches: [bs x num_patch x n_vars x patch_len], mask: [bs x num_patch x n_vars]

def apply_inv_mask(patches, mask, masked_value=0):
    # Apply the mask to the patches
    masked_patches = patches.clone()
    masked_patches[~mask] = masked_value

    return masked_patches # masked_patches: [bs x num_patch x n_vars x patch_len], mask: [bs x num_patch x n_vars]

def create_patches(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    # breakpoint()
    num_patches = (seq_len - patch_len) // stride + 1
    tgt_len = patch_len + stride*(num_patches - 1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patches


def plot_sample_reconstruction(model, revin, sample, mask_ratio, masked_value, mask_type, stride,
                               independent_channel_masking, patch_len, device, epoch, split='train', num_channels=3, 
                               forecasting_num_patches=None, fixed_position=None):
    """
    Plot a constant sample's reconstruction at the end of each epoch
    Shows multiple channels stacked vertically
    
    Args:
        model: The model to use for reconstruction
        revin: RevIN normalization module (or None)
        sample: The sample to reconstruct
        mask_ratio: Ratio of patches to mask
        masked_value: Value to use for masked patches
        mask_type: Type of masking ('random', 'forecasting', 'fixed_position')
        stride: Stride for patching
        independent_channel_masking: Whether to mask channels independently
        patch_len: Patch length
        device: Device to use
        epoch: Current epoch number
        split: Dataset split ('train' or 'val')
        num_channels: Number of channels to plot
        forecasting_num_patches: Number of patches to mask for forecasting
        fixed_position: Fixed position for masking
    """
    model.eval()
    
    # Prepare sample
    data = sample['past_values'].to(device, non_blocking=True)
    
    if revin:
        data = revin(data, mode='norm')
    
    input_patches, num_patches = create_patches(data, patch_len, stride)
    mask = create_mask(input_patches, mask_ratio, independent_channel_masking, mask_type=mask_type,
                       forecasting_num_patches=forecasting_num_patches, fixed_position=fixed_position)
    masked_patches = apply_mask(input_patches, mask, masked_value)
    target_patches = apply_inv_mask(input_patches, mask, masked_value)
    
    # Get model prediction
    with torch.no_grad():
        predicted_sequence = model(masked_patches)
    
    # Get total available channels and limit to actual number available
    n_vars = input_patches.shape[2]
    num_channels = min(num_channels, n_vars)
    
    # Create figure with subplots stacked vertically
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 5*num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]  # Make it iterable for the loop
    
    total_length = input_patches.shape[1] * patch_len
    
    # Plot each channel
    for ch_idx in range(num_channels):
        ax = axes[ch_idx]
        
        # Get data for current channel
        orig_signal = input_patches[0, :, ch_idx, :].detach().cpu().numpy().flatten()
        masked_signal = masked_patches[0, :, ch_idx, :].detach().cpu().numpy().flatten()
        recon_signal = predicted_sequence[0, :, ch_idx, :].detach().cpu().numpy().flatten()
        mask_regions = mask[0, :, ch_idx].detach().cpu().numpy() #CHANGES
        # breakpoint()
        # Plot signals
        ax.plot(orig_signal, label='Original', color='blue', alpha=0.7)
        ax.plot(recon_signal, label='Reconstruction', color='red')
        
        # Draw patch boundary grid
        for i in range(num_patches + 1):
            boundary_pos = i * patch_len
            if boundary_pos <= total_length:
                ax.axvline(x=boundary_pos, color='gray', linestyle='--', alpha=0.5)
        
        # Highlight masked regions
        for i in range(len(mask_regions)):
            if (mask_regions[i]):
                start_idx = i * patch_len
                end_idx = start_idx + patch_len
                ax.axvspan(start_idx, end_idx, color='yellow', alpha=0.2)
        
        # Add grid
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        
        # Add channel label
        ax.set_title(f'Channel {ch_idx+1} of {n_vars}', fontsize=14)
        
        # # Only add legend to the first subplot to avoid redundancy
        # if ch_idx == 0:
        ax.legend()
    
    # Add overall title
    fig.suptitle(f'Epoch {epoch}: {os.path.basename(split).capitalize()} Sample Reconstruction - {split} split, Mask Type: {mask_type}', fontsize=16)
    
    # Add x-ticks at patch boundaries
    tick_positions = [i * patch_len for i in range(num_patches + 1)]
    tick_labels = [str(i * stride) for i in range(num_patches + 1)]

    # Interleave tick labels (e.g., show every other label)
    interleave_factor = max(1, len(tick_positions) // 20)  # Adjust based on the number of patches
    tick_labels = [label if i % interleave_factor == 0 else '' for i, label in enumerate(tick_labels)]

    # Dynamically increase plot size horizontally based on the number of patches
    fig_width = max(15, num_patches // 10)  # Adjust width dynamically
    fig.set_size_inches(fig_width, fig.get_size_inches()[1])  # Keep the height unchanged

    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis='x', which='both', labelsize=10, labelrotation=90, labelbottom=True)
    
    axes[-1].set_xlabel('Timestamps')

    # Create directory if it doesn't exist
    os.makedirs(split, exist_ok=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95) # Adjust top to make room for the suptitle
    plt.savefig(os.path.join(split, f'reconstruction_epoch_{epoch+1}_mask_type_{mask_type}.png'))
    plt.close()
    
    return