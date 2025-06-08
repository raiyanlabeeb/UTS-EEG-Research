"""
Basic template with name guard.
This module serves as a starting point for Python scripts using MNE.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import combine_evoked
from scipy.stats import ttest_rel


# Create a simple example
print(f"MNE version: {mne.__version__}")


# Replace with your file path
file_path = 'dataset/ValidationEvokedActivity/sub-01/pp_validation/sub-01_E_pp_validation.set'


# Load the raw EEG data
raw = mne.io.read_raw_eeglab(file_path, preload=True)

# Create event list from annotations
events, event_id = mne.events_from_annotations(raw)
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
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-1, tmax=3, baseline=(-1, 0), preload=True)    

categories = {
    'Air': ['Air1', 'Air2', 'Air3', 'Air4'],
    'Car': ['Car1', 'Car2', 'Car3', 'Car4'],
    'Vib': ['Vib1', 'Vib2', 'Vib3', 'Vib4']
}

evoked_dict = {}
for category, labels in categories.items():
    selected_epochs = epochs[labels]
    evoked = selected_epochs.average()
    evoked_dict[category] = evoked
    
# Performs a t-test comparing compare1 to compare2
def t_test(compare1: str, compare2: str) -> int:
    compare1_data = epochs[compare1].get_data()  # shape: (n_epochs, n_channels, n_times)
    compare2_data = epochs[compare2].get_data()

    # Pick a time window in seconds
    start, end = 0.1, 0.3  
    times = epochs.times
    time_mask = (times >= start) & (times <= end)

    # Average over the selected time window
    compare1_avg = compare1_data[:, :, time_mask].mean(axis=2)
    compare2_avg = compare2_data[:, :, time_mask].mean(axis=2)

    #first dimension: subject_id, 2nd: num trials, 3rd num channels, 4th times

    p_values = []
    for ch in range(compare1_avg.shape[1]):
        min_len = min(len(compare1_avg[:, ch]), len(compare2_avg[:, ch]))
        data1 = compare1_avg[:min_len, ch]
        data2 = compare2_avg[:min_len, ch]
        t_stat, p_val = ttest_rel(data1, data2)
        p_values.append(p_val)
    
    return p_values

def plot_compare_evokeds(compare1: str, compare2: str):
    compare1_evoked = epochs[compare1].average()
    compare2_evoked = epochs[compare2].average()

    mne.viz.plot_compare_evokeds({compare1: compare1_evoked, compare2: compare2_evoked},
                                picks='eeg',
                                combine='mean',
                                title=f'ERP Comparison: {compare1} vs {compare2}')

# Does a t_test on every pair of conditions, and returns the top 5 combinations that have the most differences
def return_most_different():
    res = []
    #gives every combo for Air and Car
    for i in range(1,5):
        for j in range(1,5):
            compare1 = 'Air' + str(i)
            compare2 = 'Car' + str(j)
            p_vals = t_test(compare1, compare2)
            ch_names = epochs.ch_names
            significant_channels = [ch_names[i] for i, p in enumerate(p_vals) if p < 0.05]
            res.append((significant_channels, compare1 + ' , ' + compare2))

    #gives every combo for Air and Vib
    for i in range(1,5):
        for j in range(1,5):
            compare1 = 'Air' + str(i)
            compare2 = 'Vib' + str(j)
            p_vals = t_test(compare1, compare2)
            ch_names = epochs.ch_names
            significant_channels = [ch_names[i] for i, p in enumerate(p_vals) if p < 0.05]
            res.append((significant_channels, compare1 + ' , ' + compare2))
    
    #gives every combo for Car and Vib
    for i in range(1, 5):
        for j in range(1,5):
            compare1 = 'Car' + str(i)
            compare2 = 'Vib' + str(j)
            p_vals = t_test(compare1, compare2)
            ch_names = epochs.ch_names
            significant_channels = [ch_names[i] for i, p in enumerate(p_vals) if p < 0.05]
            res.append((significant_channels, compare1 + ' , ' + compare2))
    
    res_sorted = sorted(res, key=lambda x: len(x[0]), reverse=True)
    return res_sorted[:5]

def paired_t_test(compare1: str, compare2: str) -> int:
    compare1_data = epochs[compare1].get_data()  # shape: (n_epochs, n_channels, n_times)
    compare2_data = epochs[compare2].get_data()

    # compare1_data.shape()
    # compare2_data.shape()
    print(compare1_data, compare2_data)

def main():
    """
    Main function that serves as the entry point of the script.
    """
    
    compare1 = 'Air4'
    compare2 = 'Car1'
    # plot_compare_evokeds(compare1, compare2)

    p_values = t_test(compare1, compare2)
    ch_names = epochs.ch_names
    significant_channels = [ch_names[i] for i, p in enumerate(p_values) if p < 0.05]
    print(f"Significant channels (p < 0.05) between {compare1} and {compare2}: ", significant_channels)
    print()
    print(p_values)

    res = return_most_different()
    compare1 = res[0][1].split(",")[0].strip()
    compare2 = res[0][1].split(",")[1].strip()
    plot_compare_evokeds(compare1, compare2)
        
    

if __name__ == "__main__":
    main()
