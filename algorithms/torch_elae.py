import os
import torch
import torchaudio
import torchaudio.functional as AF


# ----------------------------
# Utility Functions
# ----------------------------

def add_noise(signal):
    """Avoid divide-by-zero by adding tiny noise."""
    noise = torch.randn_like(signal) * 1e-10
    return signal + noise


def match_shape(original, modified):
    """Pad/crop to match shape."""
    diff = original.shape[-1] - modified.shape[-1]
    if diff > 0:
        return torch.nn.functional.pad(modified, (0, diff))
    elif diff < 0:
        return modified[..., :original.shape[-1]]
    return modified


def apply_fade_in(x, fade_len=128):
    """Applies a short fade-in to avoid pops. Works for mono or stereo."""
    fade_len = min(fade_len, x.shape[-1])
    fade = torch.linspace(0.0, 1.0, fade_len, device=x.device)

    if x.dim() == 1:
        x[:fade_len] *= fade
    elif x.dim() == 2:
        if x.shape[0] <= 4:
            x[:, :fade_len] *= fade.unsqueeze(0)
        else:
            x[:fade_len, :] *= fade.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected audio shape: {x.shape}")
    return x



def torch_stft(sig, n_fft, hop_length, win_length, window):
    """
    Returns complex STFT: shape (freq, time)
    """
    return torch.stft(
        sig,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect"
    )


def torch_istft(stft_matrix, n_fft, hop_length, win_length, window, length=None):
    return torch.istft(
        stft_matrix,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length
    )


def center_channel_decomposition(stereo, window_size, overlap):
    """
    Torchaudio version of the Vickers 2009 upmix model.
    """
    L = add_noise(stereo[:, 0])
    R = add_noise(stereo[:, 1])

    hop = window_size - overlap
    window = torch.hann_window(window_size, periodic=True)

    XL = torch_stft(L, window_size, hop, window_size, window)
    XR = torch_stft(R, window_size, hop, window_size, window)

    sum_vec = XL + XR
    diff_vec = XL - XR

    sum_mag = sum_vec.abs()
    diff_mag = diff_vec.abs()

    # geometric mean modification
    k = 0.5
    diff_mag = torch.sqrt(diff_mag * ((1 - k) * diff_mag + k * sum_mag))

    center_mag = torch.sqrt(torch.tensor(0.5)) * (sum_mag - diff_mag)

    unit_vec = sum_vec / (sum_mag + 1e-12)
    center_vec = unit_vec * center_mag

    left_vec = XL - torch.sqrt(torch.tensor(0.5)) * center_vec
    right_vec = XR - torch.sqrt(torch.tensor(0.5)) * center_vec

    L_out = torch_istft(left_vec, window_size, hop, window_size, window, length=L.shape[-1])
    C_out = torch_istft(center_vec, window_size, hop, window_size, window, length=L.shape[-1])
    R_out = torch_istft(right_vec, window_size, hop, window_size, window, length=L.shape[-1])

    return (
        match_shape(L, L_out),
        match_shape(L, C_out),
        match_shape(R, R_out),
    )


def auto_correlation(X, forgetting_factor):
    power = X.abs().pow(2)                      # (freq, time)

    out = torch.zeros_like(power)
    prev = torch.zeros(power.shape[0], dtype=X.dtype, device=X.device)

    for t in range(power.shape[1]):
        prev = forgetting_factor * prev + (1 - forgetting_factor) * power[:, t]
        out[:, t] = prev

    return out


def cross_correlation(X1, X2, forgetting_factor):
    cross = X1 * torch.conj(X2)

    out = torch.zeros_like(cross)
    prev = torch.zeros(cross.shape[0], dtype=cross.dtype, device=cross.device)

    for t in range(cross.shape[1]):
        prev = forgetting_factor * prev + (1 - forgetting_factor) * cross[:, t]
        out[:, t] = prev

    return out


def decorrelate_stereo_signal(stereo, window_size, overlap, lambda_val):
    L = add_noise(stereo[:, 0])
    R = add_noise(stereo[:, 1])

    hop = window_size - overlap
    window = torch.hann_window(window_size, periodic=True)

    XL = torch_stft(L, window_size, hop, window_size, window)
    XR = torch_stft(R, window_size, hop, window_size, window)

    # Compute correlations
    auto_L = auto_correlation(XL, lambda_val)
    auto_R = auto_correlation(XR, lambda_val)
    cross = cross_correlation(XL, XR, lambda_val)

    ambient_energy = torch.sqrt(
        0.5 * (
            auto_L + auto_R
            - torch.sqrt((auto_L - auto_R).pow(2) + 4 * cross.abs().pow(2))
        )
    )

    mask_L = ambient_energy / (auto_L.sqrt() + 1e-12)
    mask_R = ambient_energy / (auto_R.sqrt() + 1e-12)

    amb_L = XL * mask_L
    amb_R = XR * mask_R

    L_out = torch_istft(amb_L, window_size, hop, window_size, window, length=L.shape[-1])
    R_out = torch_istft(amb_R, window_size, hop, window_size, window, length=L.shape[-1])

    return (
        match_shape(L, L_out),
        match_shape(R, R_out),
    )


def extract_ambience(input_stereo, window_size=1024, overlap=1024//2, fs=44100, output_path="."):
    L, C, R = center_channel_decomposition(input_stereo, window_size, overlap)

    stereo_dec = torch.stack([L, R], dim=1)

    amb_L, amb_R = decorrelate_stereo_signal(stereo_dec, window_size, overlap, lambda_val=0.7)

    ambience = torch.stack([amb_L, amb_R], dim=1)
    ambience = torch.nan_to_num(ambience, nan=0.0, posinf=0.0, neginf=0.0)

    primary = input_stereo - ambience
    primary = torch.nan_to_num(primary, nan=0.0, posinf=0.0, neginf=0.0)

    ambience = apply_fade_in(ambience, fade_len=128)
    primary = apply_fade_in(primary, fade_len=128)

    torchaudio.save(os.path.join(output_path, "ambience.wav"), ambience.T, fs)
    torchaudio.save(os.path.join(output_path, "primary.wav"), primary.T, fs)

    return ambience, primary

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_path', default=".", type=str, required=False)
    args = parser.parse_args()

    input_stereo, fs = torchaudio.load(args.input, channels_first=False)

    window_size = 1024
    overlap = window_size // 2

    start_time = time.time()
    extract_ambience(input_stereo, fs=fs, output_path=args.output_path)
    print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))
    # 6m38s takes 2.63s