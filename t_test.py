import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import combine_evoked
from scipy.stats import ttest_rel
from scipy.signal import find_peaks
import glob
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from mne.time_frequency import tfr_morlet
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set MNE logging to only show errors
mne.set_log_level('ERROR')

def load_subject_data(subject_path, condition):
    """
    subject_path: path to the subject's .set file
    condition: which condition to analyze (e.g., 'Air1', 'Car1')
    """

    # print("SUBJECT ;sob;!!: ", subject_path)
    # Load the raw EEG data
    raw = mne.io.read_raw_eeglab(subject_path, preload=True)
    
    # Custom mapping based on MATLAB script
    label_map = {
        '33024': 'Car1', '33025': 'Car2', '33026': 'Car3', '33027': 'Car4',
        '33028': 'Air1', '33029': 'Air2', '33030': 'Air3', '33031': 'Air4',
        '33032': 'Vib1', '33033': 'Vib2', '33034': 'Vib3', '33035': 'Vib4'
    }

    # Update annotation descriptions
    new_descriptions = [label_map.get(desc, desc) for desc in raw.annotations.description]
    raw.set_annotations(mne.Annotations(
        onset=raw.annotations.onset,
        duration=raw.annotations.duration,
        description=new_descriptions
    ))

    events, event_id = mne.events_from_annotations(raw)
    # Add event_repeated='drop' to handle repeated events
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-1, tmax=3, 
                       baseline=(-1, 0), preload=True, event_repeated='drop')
    
    # Get epochs for the specific condition and average them
    epochs_condition = epochs[condition]
    # Average across trials (epochs) to get shape (n_channels, n_times)
    averaged_data = epochs_condition.average().data
    # print("Num channels: ", averaged_data.shape[0], " Num time stamps: ", averaged_data.shape[1])
    
    return averaged_data, raw.ch_names

def ttest(stim1, stim2, data_dir):
    """
    This function performs a t-test between two conditions.
    returns: t_stats, p_values, data_stim1, data_stim2, ch_names, n_subjects, num_time_points
    stim1: first condition to compare
    stim2: second condition to compare
    data_dir: directory containing all subject data
    """
        # Get all subject directories (sub-01, sub-02...)
    subject_dirs = glob.glob(os.path.join(data_dir, 'sub-*'))
    n_subjects = len(subject_dirs)

    
    print(f"\nProcessing {n_subjects} subjects...")
    
    # Initialize arrays to store data for all subjects
    # Shape: (n_subjects, n_channels, n_times)
    data_stim1 = np.zeros((n_subjects, 22, 1001))  
    data_stim2 = np.zeros((n_subjects, 22, 1001))
    
    # Load and average data for each subject
    for subj_idx, subj_dir in enumerate(subject_dirs):
        # Construct path to the .set file
        set_file = os.path.join(subj_dir, 'pp_validation', f'{os.path.basename(subj_dir)}_E_pp_validation.set')
        
        # Load and average data for each condition
        data1, ch_names = load_subject_data(set_file, stim1) 
        data2, _ = load_subject_data(set_file, stim2)
        if data1.shape[0] == 22 and data2.shape[0] == 22:
            data_stim1[subj_idx] = data1
            data_stim2[subj_idx] = data2
    
    print(f"\nData shape for {stim1}: {data_stim1.shape}")
    print(f"Data shape for {stim2}: {data_stim2.shape}\n")
    
    num_channels = data_stim1.shape[1]
    num_time_points = data_stim1.shape[2]
    
    # Initialize arrays for results
    t_stats = np.zeros((num_channels, num_time_points))
    p_values = np.zeros((num_channels, num_time_points))
    
    # Perform t-test for each channel and time point
    for ch in range(num_channels):
        for t in range(num_time_points):
            # Perform paired t-test across subjects
            t_stat, p_val = ttest_rel(data_stim1[:, ch, t], data_stim2[:, ch, t])
            t_stats[ch, t] = t_stat
            p_values[ch, t] = p_val

    return t_stats, p_values, data_stim1, data_stim2, ch_names, n_subjects, num_time_points
    
def ttest_significant_channels(stim1, stim2, data_dir):
    """
    This function performs a t-test on the most significant channels between two conditions.
    stim1: first condition to compare
    stim2: second condition to compare
    data_dir: directory containing all subject data
    """
    t_stats, p_values, data_stim1, data_stim2, ch_names, n_subjects, num_time_points = ttest(stim1, stim2, data_dir)
    
    # Create time vector for x-axis
    times = np.linspace(-0.5, 1, num_time_points)
    
    # Find channels and time points with significant differences
    significant_mask = p_values < 0.05
    
    if np.any(significant_mask):
        print("\nSignificant differences found! Plotting average amplitudes...")
        
        # Calculate average amplitudes across subjects for each condition
        avg_stim1 = np.mean(data_stim1, axis=0)  # Average across subjects.  shape: (n_channel, n_time)
        avg_stim2 = np.mean(data_stim2, axis=0)
        print("shape1: ", avg_stim1.shape)
        
        # Find the channel with the most significant differences (lowest p-value)
        min_p_values = np.min(p_values, axis=1)  # Get minimum p-value for each channel
        most_sig_channel = np.argmin(min_p_values)  # Get channel with lowest p-value
        
        for ch in range(most_sig_channel):
            for t in range(500,600):
                val = p_values[ch, t]
                if val < 0.05:
                    print("P = ", val, " for channel ", ch_names[ch], " time stamp ", t)

        # Plot results for the most significant channel
        plt.figure(figsize=(15, 10))
        
        # Plot average amplitudes for the most significant channel
        plt.plot(times, avg_stim1[most_sig_channel], label=f'{stim1}', linewidth=2)
        plt.plot(times, avg_stim2[most_sig_channel], label=f'{stim2}', linewidth=2)
        
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)  # Vertical line at t=0
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Horizontal line at y=0
        plt.xlabel('Time (s)')
        plt.ylabel('Average EEG Amplitude (µV)')
        plt.title(f'Average EEG Amplitudes for {stim1} vs {stim2}\nChannel {ch_names[most_sig_channel]} (Most Significant)')
        
        # Set x-axis ticks at 100ms increments
        plt.xticks(np.arange(-0.5, 1.1, 0.1))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo significant differences found between conditions.")
    
    return t_stats, p_values

def ttest_channel(stim1, stim2, channel, data_dir):
    """
    This function performs a t-test on a specific channel between two conditions.
    stim1: first condition to compare
    stim2: second condition to compare
    channel: channel to analyze (string, e.g., 'Fp1', 'Fp2', etc.)
    data_dir: directory containing all subject data
    """
    t_stats, p_values, data_stim1, data_stim2, ch_names, n_subjects, num_time_points = ttest(stim1, stim2, data_dir)
    channel = ch_names.index(channel) #gets the actual index of the channel
    
    # Create time vector for x-axis
    times = np.linspace(-0.5, 1, num_time_points)
    
    # Calculate average amplitudes across subjects for each condition
    avg_stim1 = np.mean(data_stim1, axis=0)  # Average across subjects.  shape: (n_channel, n_time)
    avg_stim2 = np.mean(data_stim2, axis=0)
    
    # Find indices for 0-100ms window
    start_idx = np.where(times >= 0)[0][0]  # First index where time >= 0
    end_idx = np.where(times <= 0.1)[0][-1]  # Last index where time <= 100ms
    
    # Get data for the 0-100ms window
    window_times = times[start_idx:end_idx+1]
    window_stim1 = avg_stim1[channel, start_idx:end_idx+1]
    window_stim2 = avg_stim2[channel, start_idx:end_idx+1]
    
    # Find positive and negative peaks in the window
    # For stim1
    pos_peak_idx1 = np.argmax(window_stim1)  # Highest positive peak
    neg_peak_idx1 = np.argmin(window_stim1)  # Lowest negative peak
    
    # For stim2
    pos_peak_idx2 = np.argmax(window_stim2)  # Highest positive peak
    neg_peak_idx2 = np.argmin(window_stim2)  # Lowest negative peak
    
    # Get corresponding times and amplitudes
    pos_time1 = window_times[pos_peak_idx1]
    neg_time1 = window_times[neg_peak_idx1]
    pos_amp1 = window_stim1[pos_peak_idx1]
    neg_amp1 = window_stim1[neg_peak_idx1]
    
    pos_time2 = window_times[pos_peak_idx2]
    neg_time2 = window_times[neg_peak_idx2]
    pos_amp2 = window_stim2[pos_peak_idx2]
    neg_amp2 = window_stim2[neg_peak_idx2]
    
    print(f"\nPeak points for {ch_names[channel]} (0-100ms window):")
    print(f"{stim1}:")
    print(f"  Positive peak: Time = {pos_time1:.3f}s, Amplitude = {pos_amp1}µV")
    print(f"  Negative peak: Time = {neg_time1:.3f}s, Amplitude = {neg_amp1}µV")
    print(f"{stim2}:")
    print(f"  Positive peak: Time = {pos_time2:.3f}s, Amplitude = {pos_amp2}µV")
    print(f"  Negative peak: Time = {neg_time2:.3f}s, Amplitude = {neg_amp2}µV")
    
    # Plot results for the specified channel
    plt.figure(figsize=(15, 10))
    
    # Plot average amplitudes for the specified channel
    plt.plot(times, avg_stim1[channel], label=f'{stim1}', linewidth=2)
    plt.plot(times, avg_stim2[channel], label=f'{stim2}', linewidth=2)
    
    # Mark the 0-100ms window
    plt.axvspan(0, 0.1, color='gray', alpha=0.2, label='0-100ms window')
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)  # Vertical line at t=0
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Horizontal line at y=0
    plt.xlabel('Time (s)')
    plt.ylabel('Average EEG Amplitude (µV)')
    plt.title(f'Average EEG Amplitudes for {stim1} vs {stim2}\nChannel {ch_names[channel]}')
    
    # Set x-axis ticks at 100ms increments
    plt.xticks(np.arange(-0.5, 1.1, 0.1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return t_stats, p_values

def plot_stim_channel_data(stim, channel, data_dir):
    """
    This function plots the data for a specific stimulus and channel.
    stim: stimulus to plot
    channel: channel to plot
    data_dir: directory containing all subject data
    """
    _, _, data_stim1, _, ch_names, _, num_time_points = ttest(stim, stim, data_dir)
    channel = ch_names.index(channel) #gets the actual index of the channel
    
    # Create time vector for x-axis
    times = np.linspace(-0.5, 1, num_time_points)
    
    # Calculate average amplitudes across subjects
    avg_stim = np.mean(data_stim1, axis=0)  # Average across subjects. shape: (n_channel, n_time)
    
    # Find indices for 0-100ms window
    start_idx_early = np.where(times >= 0)[0][0]  # First index where time >= 0
    end_idx_early = np.where(times <= 0.1)[0][-1]  # Last index where time <= 100ms
    
    # Find indices for 300-500ms window
    start_idx_late = np.where(times >= 0.3)[0][0]  # First index where time >= 300ms
    end_idx_late = np.where(times <= 0.5)[0][-1]  # Last index where time <= 500ms
    
    # Get data for both windows
    window_times_early = times[start_idx_early:end_idx_early+1]
    window_stim_early = avg_stim[channel, start_idx_early:end_idx_early+1]
    
    window_times_late = times[start_idx_late:end_idx_late+1]
    window_stim_late = avg_stim[channel, start_idx_late:end_idx_late+1]
    
    # Find positive and negative peaks in the early window (0-100ms)
    pos_peak_idx_early = np.argmax(window_stim_early)
    neg_peak_idx_early = np.argmin(window_stim_early)
    
    # Find positive and negative peaks in the late window (300-500ms)
    pos_peak_idx_late = np.argmax(window_stim_late)
    neg_peak_idx_late = np.argmin(window_stim_late)
    
    # Get corresponding times and amplitudes for early window
    pos_time_early = window_times_early[pos_peak_idx_early]
    neg_time_early = window_times_early[neg_peak_idx_early]
    pos_amp_early = window_stim_early[pos_peak_idx_early]
    neg_amp_early = window_stim_early[neg_peak_idx_early]
    
    # Get corresponding times and amplitudes for late window
    pos_time_late = window_times_late[pos_peak_idx_late]
    neg_time_late = window_times_late[neg_peak_idx_late]
    pos_amp_late = window_stim_late[pos_peak_idx_late]
    neg_amp_late = window_stim_late[neg_peak_idx_late]
    
    print(f"\nPeak points for {ch_names[channel]}:")
    print(f"{stim} (0-100ms window):")
    print(f"  Positive peak: Time = {pos_time_early:.3f}s, Amplitude = {pos_amp_early}µV")
    print(f"  Negative peak: Time = {neg_time_early:.3f}s, Amplitude = {neg_amp_early}µV")
    print(f"{stim} (300-500ms window):")
    print(f"  Positive peak: Time = {pos_time_late:.3f}s, Amplitude = {pos_amp_late}µV")
    print(f"  Negative peak: Time = {neg_time_late:.3f}s, Amplitude = {neg_amp_late}µV")
    
    # Plot results for the specified channel
    plt.figure(figsize=(15, 10))
    
    # Plot average amplitudes for the specified channel
    plt.plot(times, avg_stim[channel], label=f'{stim}', linewidth=2)
    
    # Mark both windows
    plt.axvspan(0, 0.1, color='gray', alpha=0.2, label='0-100ms window')
    plt.axvspan(0.3, 0.5, color='lightgray', alpha=0.2, label='300-500ms window')
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)  # Vertical line at t=0
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Horizontal line at y=0
    plt.xlabel('Time (s)')
    plt.ylabel('Average EEG Amplitude (µV)')
    plt.title(f'Average EEG Amplitudes for {stim}\nChannel {ch_names[channel]}')
    
    # Set x-axis ticks at 100ms increments
    plt.xticks(np.arange(-0.5, 1.1, 0.1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_stim_channel_freq(stim, channel, data_dir):
    """
    This function plots the frequency domain data for a specific stimulus and channel.
    stim: stimulus to plot
    channel: channel to plot
    data_dir: directory containing all subject data
    """
    _, _, data_stim1, _, ch_names, _, num_time_points = ttest(stim, stim, data_dir)
    channel = ch_names.index(channel) #gets the actual index of the channel
    
    # Calculate averages across subjects
    avg_stim = np.mean(data_stim1, axis=0)  # Average across subjects. shape: (n_channel, n_time)
    
    # Get the sampling rate (assuming 1000 Hz based on the time points)
    sampling_rate = 1000  # Hz
    
    # Apply FFT to the time domain data
    # We'll use the entire time window for frequency analysis
    signal = avg_stim[channel] #shape: (n_time) just a 1d array
    n = len(signal)
    
    # Apply FFT
    fft_result = np.fft.fft(signal)
    
    # Calculate frequency bins
    freq_bins = np.fft.fftfreq(n, 1/sampling_rate)
    
    # Calculate magnitude spectrum (absolute value of FFT)
    magnitude_spectrum = np.abs(fft_result)
    
    # Only plot positive frequencies up to Nyquist frequency
    positive_freq_mask = freq_bins >= 0
    frequencies = freq_bins[positive_freq_mask]
    magnitudes = magnitude_spectrum[positive_freq_mask]

    # Find peaks in the frequency spectrum
    peaks, _ = find_peaks(magnitudes, height=np.max(magnitudes)*0.1)  # Only consider peaks above 10% of max
    
    print(f"\nPeak frequencies for {ch_names[channel]}:")
    for peak in peaks:
        if frequencies[peak] > 50:
            break;
        print(f"Frequency: {frequencies[peak]:.2f} Hz, Magnitude: {magnitudes[peak]}")
    
    # Plot the frequency spectrum
    plt.figure(figsize=(15, 10))
    
    # Plot magnitude spectrum
    plt.plot(frequencies, magnitudes, linewidth=2)
    
    # Add labels and title
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Frequency Spectrum for {stim}\nChannel {ch_names[channel]}')
    
    # Set x-axis limit to show frequencies up to 50 Hz (typical EEG range)
    plt.xlim(0, 50)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for common EEG frequency bands
    plt.axvline(x=4, color='r', linestyle='--', alpha=0.3, label='Delta (0.5-4 Hz)')
    plt.axvline(x=8, color='g', linestyle='--', alpha=0.3, label='Theta (4-8 Hz)')
    plt.axvline(x=13, color='b', linestyle='--', alpha=0.3, label='Alpha (8-13 Hz)')
    plt.axvline(x=30, color='m', linestyle='--', alpha=0.3, label='Beta (13-30 Hz)')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_stim_type_channels(stim_type, channels, data_dir):
    """
    This function plots data for all levels of a stimulus type across multiple channels.
    stim_type: type of stimulus ('Car', 'Air', or 'Vib')
    channels: list of channels to analyze (e.g., ['C3', 'C4', 'F3'])
    data_dir: directory containing all subject data
    """
    # Define all possible levels for each stimulus type
    stim_levels = {
        'Car': ['Car1', 'Car2', 'Car3', 'Car4'],
        'Air': ['Air1', 'Air2', 'Air3', 'Air4'],
        'Vib': ['Vib1', 'Vib2', 'Vib3', 'Vib4']
    }
    
    if stim_type not in stim_levels:
        raise ValueError(f"Invalid stimulus type. Must be one of: {list(stim_levels.keys())}")
    
    # Get all levels for the specified stimulus type
    levels = stim_levels[stim_type]
    
    # Create a figure with subplots for each channel
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 5*n_channels))
    if n_channels == 1:
        axes = [axes]  # Make axes iterable for single channel case
    
    # Process each channel
    for ch_idx, channel in enumerate(channels):
        ax = axes[ch_idx]
        
        # Process each level of the stimulus
        for level in levels:
            # Get data for this level and channel
            _, _, data_stim, _, ch_names, _, num_time_points = ttest(level, level, data_dir)
            channel_idx = ch_names.index(channel)
            
            # Create time vector
            times = np.linspace(-0.5, 1, num_time_points)
            
            # Calculate average amplitudes
            avg_stim = np.mean(data_stim, axis=0)
            
            # Find indices for both windows
            start_idx_early = np.where(times >= 0)[0][0]
            end_idx_early = np.where(times <= 0.1)[0][-1]
            start_idx_late = np.where(times >= 0.3)[0][0]
            end_idx_late = np.where(times <= 0.5)[0][-1]
            
            # Get data for both windows
            window_times_early = times[start_idx_early:end_idx_early+1]
            window_stim_early = avg_stim[channel_idx, start_idx_early:end_idx_early+1]
            window_times_late = times[start_idx_late:end_idx_late+1]
            window_stim_late = avg_stim[channel_idx, start_idx_late:end_idx_late+1]
            
            # Find peaks in both windows
            pos_peak_idx_early = np.argmax(window_stim_early)
            neg_peak_idx_early = np.argmin(window_stim_early)
            pos_peak_idx_late = np.argmax(window_stim_late)
            neg_peak_idx_late = np.argmin(window_stim_late)
            
            # Get peak values
            pos_time_early = window_times_early[pos_peak_idx_early]
            neg_time_early = window_times_early[neg_peak_idx_early]
            pos_amp_early = window_stim_early[pos_peak_idx_early]
            neg_amp_early = window_stim_early[neg_peak_idx_early]
            
            pos_time_late = window_times_late[pos_peak_idx_late]
            neg_time_late = window_times_late[neg_peak_idx_late]
            pos_amp_late = window_stim_late[pos_peak_idx_late]
            neg_amp_late = window_stim_late[neg_peak_idx_late]
            
            # Print peak information
            print(f"\nPeak points for {channel} - {level}:")
            print(f"0-100ms window:")
            print(f"  Positive peak: Time = {pos_time_early:.3f}s, Amplitude = {pos_amp_early}µV")
            print(f"  Negative peak: Time = {neg_time_early:.3f}s, Amplitude = {neg_amp_early}µV")
            print(f"300-500ms window:")
            print(f"  Positive peak: Time = {pos_time_late:.3f}s, Amplitude = {pos_amp_late}µV")
            print(f"  Negative peak: Time = {neg_time_late:.3f}s, Amplitude = {neg_amp_late}µV")
            
            # Plot the data
            ax.plot(times, avg_stim[channel_idx], label=level, linewidth=2)
            
            # Mark the windows
            ax.axvspan(0, 0.1, color='gray', alpha=0.1)
            ax.axvspan(0.3, 0.5, color='lightgray', alpha=0.1)
            
            # Add vertical and horizontal lines
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            # Set labels and title
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Average EEG Amplitude (µV)')
            ax.set_title(f'Channel {channel}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set x-axis ticks
            ax.set_xticks(np.arange(-0.5, 1.1, 0.1))
    
    plt.tight_layout()
    plt.show()

# def plot_erp_topographies(evoked, times, title=None, figsize=(15, 5)):
    """
    Plot ERP component topographies at specific time points.
    
    Parameters
    ----------
    evoked : mne.Evoked
        The evoked EEG data object
    times : list of float
        List of time points (in seconds) to plot topographies
    title : str, optional
        Main title for the figure
    figsize : tuple, optional
        Figure size in inches (width, height)
    """
    
    # Create figure with subplots
    # We need n_times + 1 axes (one for each time point plus one for the colorbar)
    n_times = len(times)
    fig, axes = plt.subplots(1, n_times + 1, figsize=figsize)
    
    # Plot topomap for each time point
    for idx, time in enumerate(times):
        # Convert time to milliseconds for the title
        time_ms = int(time * 1000)
        
        # Plot topomap with colorbar
        evoked.plot_topomap(times=time, axes=axes[idx], show=False, 
                          colorbar=True)
        axes[idx].set_title(f'{time_ms} ms')
    
    # Remove the last axis (colorbar axis) as it's not needed
    fig.delaxes(axes[-1])
    
    # Add main title if provided
    if title:
        fig.suptitle(title, y=1.05)
    
    plt.tight_layout()
    plt.show()

def plot_erp_topography(stim_type, times,  data_dir, figsize=(15, 5),):
    """
    Create a topography plot for a given evoked object at specific time points.
    ERP: Event-Related Potential: Looking at voltage changes over time
   
    Parameters
    ----------
    stim_type : type of stimulus ('Car', 'Air', or 'Vib')
    times : list of float
        List of time points (in seconds) to plot topographies
    data_dir : str
        Directory containing the data
    figsize : tuple, optional
        Figure size in inches (width, height)
    """
    # Get data for condition
    _, _, data_stim1, _, ch_names, _, _ = ttest(stim_type, stim_type, data_dir)
    
    # Calculate average across subjects
    avg_stim = np.mean(data_stim1, axis=0)  # shape: (n_channels, n_time)
    
    # Create MNE Evoked object
    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg') #contains all metadata

    #An evoked object contains 1.) averaged EEG data, 2.) metadata about the data (channel names, sampling rate, etc.) 3.) methods for visualizing the data
    #MNE now: 1.) knows which channel is which, 2.) knows the time scale, 3.) knows how to display topographies, 4.) knows how to analyze the data
    evoked = mne.EvokedArray(avg_stim, info)
    
    # Set up the standard 10-20 montage
    #Montage object: a map of where each electrode is located on the head
    #standard_1020: a standard montage that is used in EEG research
    montage = mne.channels.make_standard_montage('standard_1020') #tells MNE where each electrode is located on the head
    evoked.set_montage(montage)

    # Plot the topomap and get the figure
    fig = evoked.plot_topomap(times=times, show=False, ch_type='eeg')
    
    # Adjust the layout to make room for the title
    fig.subplots_adjust(top=0.85)
    
    # Add the title
    fig.suptitle("ERP Topographies for " + stim_type + " Condition", y=0.95)
    
    plt.show()

def plot_ersp_topography(stim_type, times, freqs, data_dir, figsize=(15, 10)):
    """
    Plot ERSP topographies for specific frequencies and time points.

    Parameters
    ----------
    stim_type : str
        Stimulus type ('Car', 'Air', 'Vib', etc.)
    times : list of float
        Time points in seconds (e.g., [0.1, 0.2])
    freqs : list of float
        Frequencies in Hz (e.g., [8, 13, 20, 30])
    data_dir : str
        Path to EEG data
    figsize : tuple
        Size of the figure
    """

    # Load data
    _, _, data_stim1, _, ch_names, _, _ = ttest(stim_type, stim_type, data_dir)
    avg_stim = np.mean(data_stim1, axis=0)  # shape: (n_channels, n_time)
    
    # Create EpochsArray with one fake trial
    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
    epochs = mne.EpochsArray(avg_stim[np.newaxis, :, :], info)

    # Define Morlet wavelet params
    all_freqs = np.arange(min(freqs), max(freqs) + 1, 1)
    n_cycles = all_freqs / 2.0
    power = tfr_morlet(epochs, freqs=all_freqs, n_cycles=n_cycles, return_itc=False)

    # No baseline (since your data starts at 0s)
    # power.apply_baseline(baseline=(0, 0.1), mode='logratio')  # Optional

    # Set montage
    montage = mne.channels.make_standard_montage('standard_1020')
    power.info.set_montage(montage)

    # Create topomap grid
    fig, axes = plt.subplots(len(freqs), len(times), figsize=figsize)
    fig.suptitle(f"ERSP Topographies for {stim_type}", fontsize=16)

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
    # Specify the directory containing all subject data
    data_dir = 'dataset/ValidationEvokedActivity'
    
    # Plot topographies at key ERP time points (in seconds)
    times = [0.1, 0.2, 0.3, 0.4]  # 100ms, 200ms, 300ms, 400ms
    freqs = [8, 13, 20, 30]
    # plot_erp_topography('Vib4', times, data_dir=data_dir) 
    plot_ersp_topography('Vib4', times, freqs, data_dir=data_dir)



main()