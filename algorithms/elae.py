import os

import numpy as np
import soundfile as sf
from scipy.signal import (
    stft, istft
)


def add_noise(signal):
    """Checks for the value 0 and adds an unnoticeable value."""
    # print("  Adding noise to the signal...")
    noise = np.random.normal(1e-100, 1e-50, np.size(signal))
    data = signal + noise
    return data


def match_shape(original, modified):
    """Function for reshaping a modified signal to the original signal."""
    original_len = len(original)
    modified_len = len(modified)

    if modified_len > original_len:
        adjusted = modified[:original_len]
    elif modified_len < original_len:
        padding = np.zeros(original_len - modified_len)
        adjusted = np.concatenate((modified, padding))
    else:
        adjusted = modified
    return adjusted


def center_channel_decomposition(stereo_input, fs, window, nperseg, noverlap):
    """
    Vickers, Earl.
    "Frequency-domain two-to three-channel upmix for center channel derivation and speech enhancement."
    Audio Engineering Society Convention 127. Audio Engineering Society, 2009.
    """
    stereo_left, stereo_right = stereo_input[:, 0], stereo_input[:, 1]
    # print("Beginning center channel decomposition...")

    # Add noise to avoid dividing by zero value
    stereo_left = add_noise(stereo_left)
    stereo_right = add_noise(stereo_right)

    # Perform STFT
    # print("  Performing STFT on stereo channels")
    f, t, XL = stft(stereo_left, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    _, _, XR = stft(stereo_right, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

    # Compute the sum and difference of XL and XR
    # print("  Computing the sum and different magnitudes")
    sum_vector = XL + XR
    difference_vector = XL - XR
    # Compute magnitudes
    sum_magnitude = np.abs(sum_vector)
    difference_magnitude = np.abs(difference_vector)

    # Optionally perform the "geometric mean"
    perform_geometric_mean = True
    if perform_geometric_mean:
        # print("  Performing optional geometric mean modification to the difference magnitude")
        k = 0.5  # controls the balance between the sum and difference magnitudes
        difference_magnitude = np.sqrt(
            difference_magnitude * ((1 - k) * difference_magnitude + k * sum_magnitude))

    # Estimate the magnitude of the desired center vector
    # print("  Estimating the magnitude of the center channel")
    center_magnitude = np.sqrt(0.5) * (sum_magnitude - difference_magnitude)

    # Take a unit vector in the XL+XR direction and scale it by the estimated center magnitude
    # print("  Scaling the unit vector by the center magnitude")
    unit_vector = sum_vector / np.abs(sum_vector)
    center_vector = unit_vector * center_magnitude

    # Compute the left and right outputs:
    # print("  Computing the Left and Right outputs")
    left_vector = XL - np.sqrt(0.5) * center_vector
    right_vector = XR - np.sqrt(0.5) * center_vector

    # Compute ISTFT on the derived vectors
    # print("  Performing ISTFT on the derived channels")
    _, left_out = istft(left_vector, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    _, center_out = istft(center_vector, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    _, right_out = istft(right_vector, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

    # Adjust shape to match the input array
    # print("  Adjusting the shape to match the input array")
    left_out = match_shape(stereo_input[:, 0], left_out)
    center_out = match_shape(stereo_input[:, 0], center_out)
    right_out = match_shape(stereo_input[:, 1], right_out)

    # print("Center channel decomposition complete.")
    return left_out, center_out, right_out


def decorrelate_stereo_signal(stereo_input, fs, window, nperseg, noverlap, lambda_val):
    """
    Correlation-Based Ambience Extraction Algorithm
    Based on: Merimaa, J., Goodwin, M. M., & Jot, J.-M. (2007).
    "Correlation-Based Ambience Extraction from Stereo Recordings." AES 123rd Convention.
    """
    stereo_left, stereo_right = stereo_input[:, 0], stereo_input[:, 1]
    # print("Beginning correlation-based ambience extraction...")

    # Add noise to avoid numerical issues
    stereo_left = add_noise(stereo_left)
    stereo_right = add_noise(stereo_right)

    # Perform STFT on both channels
    f, t, stft_left = stft(stereo_left, fs, window=window, nperseg=nperseg, noverlap=noverlap)
    _, _, stft_right = stft(stereo_right, fs, window=window, nperseg=nperseg, noverlap=noverlap)

    num_frequency_bins, num_time_frames = stft_left.shape

    # Calculate auto-correlation and cross-correlation
    def auto_correlation(signal, forgetting_factor=0.5):
        result = np.zeros((num_frequency_bins, num_time_frames), dtype='complex')
        for freq_bin in range(num_frequency_bins):
            for time_frame in range(num_time_frames):
                last_value = result[freq_bin, time_frame - 1] if time_frame > 0 else 0
                current_value = np.power(np.abs(signal[freq_bin, time_frame]), 2)
                result[freq_bin, time_frame] = (forgetting_factor * last_value) + (
                        (1 - forgetting_factor) * current_value)
        return result

    def cross_correlation(signal_1, signal_2, forgetting_factor=0.5):
        result = np.zeros((num_frequency_bins, num_time_frames), dtype='complex')
        for freq_bin in range(num_frequency_bins):
            for time_frame in range(num_time_frames):
                last_value = result[freq_bin, time_frame - 1] if time_frame > 0 else 0
                current_value = signal_1[freq_bin, time_frame] * np.conj(signal_2[freq_bin, time_frame])
                result[freq_bin, time_frame] = (forgetting_factor * last_value) + (
                        (1 - forgetting_factor) * current_value)
        return result

    autocorr_left = auto_correlation(stft_left, forgetting_factor=lambda_val)
    autocorr_right = auto_correlation(stft_right, forgetting_factor=lambda_val)
    cross_corr = cross_correlation(stft_left, stft_right, forgetting_factor=lambda_val)

    # Generate equal levels mask
    ambient_energy = np.sqrt(0.5 * (autocorr_left + autocorr_right - np.sqrt(
        np.power(autocorr_left - autocorr_right, 2) + (4 * np.power(np.abs(cross_corr), 2)))))
    mask_left = ambient_energy / np.sqrt(autocorr_left)
    mask_right = ambient_energy / np.sqrt(autocorr_right)

    # Apply the mask to the STFTs
    freq_ambience_left = np.multiply(stft_left, mask_left)
    freq_ambience_right = np.multiply(stft_right, mask_right)

    # Perform inverse STFT to obtain time-domain signals
    _, ambience_left = istft(freq_ambience_left, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    _, ambience_right = istft(freq_ambience_right, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

    # Adjust shape to match the input
    ambience_left = match_shape(stereo_input[:, 0], ambience_left)
    ambience_right = match_shape(stereo_input[:, 1], ambience_right)

    # print("Ambience extraction completed.")
    return ambience_left, ambience_right


def extract_ambience(input_stereo, stft_window_size=128, stft_overlap=96, fs=44100, output_path="."):
    left_data, center_data, right_data = center_channel_decomposition(
        stereo_input=input_stereo,
        fs=fs,
        window='hann',
        nperseg=stft_window_size,
        noverlap=stft_overlap
    )
    decomp_signal = np.column_stack((left_data, right_data))
    ambience_left, ambience_right = decorrelate_stereo_signal(
        stereo_input=decomp_signal,
        fs=fs,
        window='hann',
        nperseg=stft_window_size,
        noverlap=stft_overlap,
        lambda_val=0.7
    )
    ambient_data = np.column_stack((ambience_left, ambience_right))
    primary_data = input_stereo - ambient_data
    sf.write(os.path.join(output_path, f"ambience.wav"), ambient_data, fs)
    sf.write(os.path.join(output_path, f"primary.wav"), primary_data, fs)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    start_time = time.time()
    input_stereo, fs = sf.read(args.input)

    window_size = 1024
    overlap = 2
    extract_ambience(input_stereo, window_size, overlap, fs, args.output_path)
    print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))
