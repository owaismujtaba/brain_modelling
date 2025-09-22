import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d


def gauss_smooth(
    inputs,
    device,
    smooth_kernel_std: float = 2.0,
    smooth_kernel_size: int = 100,
    padding: str = "same",
) -> torch.Tensor:
    """
    Apply 1D Gaussian smoothing along the time axis.

    Args:
        inputs (torch.Tensor): A tensor of shape [B, T, N] where
            B = batch size, T = time steps, N = features.
            Tensor should already be on the correct device.
        device (str): Device for computation ('cuda' or 'cpu').
        smooth_kernel_std (float, optional): Standard deviation of the
            Gaussian smoothing kernel. Default is 2.0.
        smooth_kernel_size (int, optional): Size of the Gaussian kernel.
            Default is 100.
        padding (str, optional): Padding mode for convolution. Options are
            'same' or 'valid'. Default is 'same'.

    Returns:
        torch.Tensor: Smoothed tensor of shape [B, T, N].
    """
    # Create Gaussian kernel in numpy
    orig_dtype = inputs.dtype
    inputs_fp32 = inputs.to(torch.float32)

    kernel = np.zeros(smooth_kernel_size, dtype=np.float32)
    kernel[smooth_kernel_size // 2] = 1
    kernel = gaussian_filter1d(kernel, smooth_kernel_std)
    valid_idx = np.argwhere(kernel > 0.01)
    kernel = kernel[valid_idx]
    kernel = np.squeeze(kernel / np.sum(kernel))
    kernel = torch.tensor(kernel, dtype=torch.float32, device=device)
    kernel = kernel.view(1, 1, -1)

    batch_size, time_steps, channels = inputs_fp32.shape
    inputs_fp32 = inputs_fp32.permute(0, 2, 1)
    kernel = kernel.repeat(channels, 1, 1)

    smoothed = F.conv1d(inputs_fp32, kernel, padding=padding, groups=channels)
    smoothed = smoothed.permute(0, 2, 1)

    # cast back to original dtype (bfloat16)
    return smoothed.to(orig_dtype)


def create_attention_mask(sequence_lengths: torch.Tensor) -> torch.Tensor:
    """
    Create an attention mask for variable-length sequences.

    Args:
        sequence_lengths (torch.Tensor): Tensor of shape [batch_size]
            containing the lengths of each sequence.

    Returns:
        torch.Tensor: Attention mask of shape [batch_size, 1, max_length, max_length],
        where valid positions are 0.0 and padding positions are -inf.
    """
    batch_size = sequence_lengths.size(0)
    max_length = sequence_lengths.max().item()

    # Create a mask for valid key positions: [batch_size, max_length]
    key_mask = torch.arange(max_length, device=sequence_lengths.device)
    key_mask = key_mask.expand(batch_size, max_length)
    key_mask = key_mask < sequence_lengths.unsqueeze(1)

    # Expand to [batch_size, 1, 1, max_length]
    key_mask = key_mask.unsqueeze(1).unsqueeze(1)

    # Broadcast across query positions: [batch_size, 1, max_length, max_length]
    attention_mask = key_mask.expand(batch_size, 1, max_length, max_length)

    # Convert boolean mask to float mask: valid → 0.0, padding → -inf
    attention_mask = attention_mask.float()
    attention_mask = attention_mask.masked_fill(~attention_mask.bool(), float("-inf"))

    return attention_mask


def transform_data(config, features, n_time_steps, device, mode="train"):
    """
    Apply various augmentations and smoothing to data.

    Performing augmentations is much faster on GPU than CPU.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing data transformation parameters.
    features : torch.Tensor
        Input data tensor of shape (batch_size, time_steps, channels).
    n_time_steps : int
        Number of time steps in the sequence.
    device : torch.device
        Torch device (e.g., "cuda" or "cpu").
    mode : str, optional
        Mode of operation ("train" or "eval"), by default "train".

    Returns
    -------
    features : torch.Tensor
        Transformed feature tensor.
    n_time_steps : int
        Possibly updated number of time steps after augmentation.
    """
    data_shape = features.shape
    batch_size = data_shape[0]
    channels = data_shape[-1]
    dtype = features.dtype  # ensure consistency

    transforms = config["dataset"]["data_transforms"]
    static_gain_std = transforms["static_gain_std"]
    white_noise_std = transforms["white_noise_std"]
    constant_offset_std = transforms["constant_offset_std"]
    random_walk_std = transforms["random_walk_std"]
    random_walk_axis = transforms["random_walk_axis"]
    random_cut = transforms["random_cut"]
    smooth_data = transforms["smooth_data"]
    smooth_kernel_size = transforms["smooth_kernel_size"]
    smooth_kernel_std = transforms["smooth_kernel_std"]

    if mode == "train":
        # Static gain noise
        if static_gain_std > 0:
            warp_mat = torch.tile(
                torch.unsqueeze(torch.eye(channels, device=device, dtype=dtype), dim=0),
                (batch_size, 1, 1),
            )
            warp_mat += torch.randn_like(warp_mat, device=device, dtype=dtype) * static_gain_std
            features = torch.matmul(features, warp_mat)

        # White noise
        if white_noise_std > 0:
            features += torch.randn(data_shape, device=device, dtype=dtype) * white_noise_std

        # Constant offset noise
        if constant_offset_std > 0:
            features += (
                torch.randn((batch_size, 1, channels), device=device, dtype=dtype)
                * constant_offset_std
            )

        # Random walk noise
        if random_walk_std > 0:
            features += torch.cumsum(
                torch.randn(data_shape, device=device, dtype=dtype) * random_walk_std,
                dim=random_walk_axis,
            )

        # Random cutoff
        if random_cut > 0:
            cut = np.random.randint(0, random_cut)
            features = features[:, cut:, :]
            n_time_steps -= cut

    # Gaussian smoothing (applied in both training and validation)
    if smooth_data:
        features = gauss_smooth(
            inputs=features,
            device=device,
            smooth_kernel_std=smooth_kernel_std,
            smooth_kernel_size=smooth_kernel_size,
        )

    return features, n_time_steps
