import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet

def get_files(base_dir="eeg-motor-movementimagery-dataset-1.0.0/files"):
    """
    Returns a list of full paths to the first trial (R01) EDF file for the first 32 subjects in the dataset.
    """
    edf_files = []
    for subject in sorted(os.listdir(base_dir)):
        subject_dir = os.path.join(base_dir, subject)
        if os.path.isdir(subject_dir):
            first_trial_file = f"{subject}R01.edf"
            first_trial_path = os.path.join(subject_dir, first_trial_file)
            if os.path.isfile(first_trial_path):
                edf_files.append(first_trial_path)
    return edf_files

def average_eeg_data(edf_files):
    """
    Reads EEG data from a list of EDF files and returns the average data across all files.
    Returns:
        avg_data: numpy array of shape (n_channels, n_times)
        info: MNE info object (from the last file, for channel locations, etc.)
        times: numpy array of time points
    """
    all_data = []
    for edf_file in edf_files:
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        data, times = raw.get_data(return_times=True)
        if data.shape[1] != 9760:
            continue
        all_data.append(data)
    all_data = np.stack(all_data, axis=0)  # shape: (n_subjects, n_channels, n_times)
    avg_data = np.mean(all_data, axis=0)   # shape: (n_channels, n_times)
    return avg_data, raw.info, times

def plot_ersp_topography(avg_data, info, times, freqs, figsize=(15, 10)):
    """
    Plot ERSP topographies using MNE's time-frequency analysis (same style as t_test.py).
    
    Parameters:
    -----------
    avg_data : array-like, shape (n_channels, n_times)
        Averaged EEG data
    info : MNE info object
        Channel information
    times : list of float
        Time points in seconds to plot topographies
    freqs : list of float
        Frequencies in Hz to plot
    figsize : tuple
        Size of the figure
    """
    # Create EpochsArray with one fake trial (same as t_test.py)
    epochs = mne.EpochsArray(avg_data[np.newaxis, :, :], info)

    # Clean up channel names by removing dots and converting to uppercase
    clean_names = {name: name.replace('.', '').upper() for name in info['ch_names']}
    epochs.rename_channels(clean_names)

    # Define Morlet wavelet params (same as t_test.py)
    all_freqs = np.arange(min(freqs), max(freqs) + 1, 1)
    n_cycles = all_freqs / 2.0
    power = tfr_morlet(epochs, freqs=all_freqs, n_cycles=n_cycles, return_itc=False)

    # No baseline (same as t_test.py)
    # power.apply_baseline(baseline=(0, 0.1), mode='logratio')  # Optional

    # Get available channels in standard_1020 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    available_channels = [ch for ch in power.ch_names if ch in montage.ch_names]
    
    # Pick only the channels that are in the montage
    power.pick_channels(available_channels)
    
    # Set montage
    power.info.set_montage(montage)

    # Create topomap grid (same as t_test.py)
    fig, axes = plt.subplots(len(freqs), len(times), figsize=figsize)
    fig.suptitle(f"ERSP Topographies for Resting State", fontsize=16)

    for i, freq in enumerate(freqs):
        for j, t in enumerate(times):
            ax = axes[i, j] if len(freqs) > 1 else axes[j]
            power.plot_topomap(
                tmin=t, tmax=t, fmin=freq, fmax=freq,
                axes=ax, show=False, colorbar=False, ch_type='eeg', cmap='RdBu_r'
            )
            ax.set_title(f"{freq} Hz, {int(t*1000)} ms", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    """
    Main function to create ERSP topography maps.
    """
    files = get_files()
    avg_data, info, times = average_eeg_data(files[:32])
    
    print(f"Average data shape: {avg_data.shape}")
    print(f"Sampling frequency: {info['sfreq']} Hz")
    print(f"Time duration: {times[-1]:.2f} seconds")
    
    # Define time points and frequencies for ERSP topography
    time_points = [0.1, 0.2, 0.3, 0.4]  # Time points in seconds
    freq_points = [8, 13, 20, 30]        # Frequency points in Hz
    
    
    print("Creating ERSP topographies...")
    plot_ersp_topography(avg_data, info, time_points, freq_points)

if __name__ == "__main__":
    main()