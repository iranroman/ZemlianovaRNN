import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import re

Ncycles = 50

def relu(x):
    return np.maximum(0, x)

def normalize(x):
    min_val = np.min(x, axis=0)
    max_val = np.max(x, axis=0)
    range_val = max_val - min_val
    return np.where(range_val > 0, (x - min_val) / range_val, 0)

def load_and_process_data(periods, directory='model_outputs', save_dir='fig1_plotting'):
    # Create directory to save outputs if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for period in periods:
        hidden_state_path = os.path.join(directory, f'hidden_states_{period}.npy')
        peaks_path = os.path.join(directory, f'peaks_{period}.npy')

        if os.path.exists(hidden_state_path) and os.path.exists(peaks_path):
            hidden_states = np.squeeze(np.load(hidden_state_path))
            peaks = np.load(peaks_path)
            
            firing_rates = relu(hidden_states)
            normalized_firing_rates = normalize(firing_rates)
            
            num_neurons = normalized_firing_rates.shape[1]
            excitatory_indices = range(int(num_neurons * 0.8))
            inhibitory_indices = range(int(num_neurons * 0.8), num_neurons)
           
            distances = {}
            for neuron_index in range(num_neurons):
                neuron_firing_rates = normalized_firing_rates[:, neuron_index]
                neuron_peaks = []
                if neuron_firing_rates.max():
                    for i in range(1, len(peaks) - 1):
                        window_start = peaks[i] - (peaks[i] - peaks[i - 1]) // 2
                        window_end = peaks[i] + (peaks[i + 1] - peaks[i]) // 2
                        max_index = np.argmax(neuron_firing_rates[window_start:window_end]) + window_start
                        neuron_peaks.append(max_index)
                else:
                    neuron_peaks = peaks[1:-1]-1

                # Calculate distances from defined peaks to actual neuron peaks
                distances[neuron_index] = np.median([p - peaks[i+1] for i, p in enumerate(neuron_peaks)])
            
            # Separate sorting for excitatory and inhibitory neurons
            def get_sorted_indices(indices):
                positive_sorted = []
                negative_sorted = []
                
                for idx in indices:
                    pos_distances = True if distances[idx] >= 0 else False
                    
                    if pos_distances:
                        positive_sorted.append((idx,distances[idx]))
                    else:
                        negative_sorted.append((idx,distances[idx]))
                
                # Sort both lists
                positive_sorted.sort(key=lambda x: x[1])  # Ascending order for positive
                negative_sorted.sort(key=lambda x: x[1])  # Descending order for negative
                
                # Combine and extract indices
                return [x[0] for x in positive_sorted + negative_sorted]
            
            excitatory_sorted = get_sorted_indices(excitatory_indices)
            inhibitory_sorted = get_sorted_indices(inhibitory_indices)
            
            # Combine sorted indices
            sorted_indices = excitatory_sorted + inhibitory_sorted
            
            # Visualization
            plt.figure(figsize=(20, 8))
            plt.imshow(normalized_firing_rates[10000:11000, sorted_indices].T, aspect='auto', cmap='viridis', interpolation='nearest')
            for peak in peaks:
                if 10000 < peak < 11000:  # Ensure peak is within the first 1000 samples
                    plt.axvline(x=peak-10000, color='k', linestyle='--')
            plt.colorbar(label='Normalized Firing Rate')
            plt.title(f'Firing Rates Sorted by Neuron Distance to Peak for Period {period}s')
            plt.xlabel('Time Steps')
            plt.ylabel('Neuron Index')
            plt.axhline(y=len(excitatory_sorted), color='r', linestyle='--')  # Divide line between Excitatory and Inhibitory
            plt.savefig(os.path.join(save_dir, f'firing_rate_plot_{period}.png'))  # Save plot

            print(f"Processed data for period {period}. Neurons sorted by distance to peaks.")
            np.save(os.path.join(save_dir, f'neuron_sorting_indices_for_{period}.npy'), sorted_indices)

def calculate_cycle_lengths(pca_results, peaks):
    # Calculate distances between consecutive peaks in PCA space
    distances = []
    for i in range(1, len(peaks)):
        if peaks[i] < pca_results.shape[0]:
            dist = pca_results[peaks[i-1]:peaks[i]]
            dist = np.diff(dist,axis=0)
            dist = np.sqrt(np.sum(dist**2, axis=1))
        distances.append(np.sum(dist))
    return distances 


def load_data_compute_pca_and_plot(periods, directory='model_outputs', save_dir='fig1_plotting'):
    all_rates = []  # Collect all rates across periods for PCA
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')  # 3D plot for PCA
    means = []
    std_devs = []

    # Load data, apply ReLU, normalize, and fit PCA
    for period in periods:
        hidden_state_path = os.path.join(directory, f'hidden_states_{period}.npy')
        if os.path.exists(hidden_state_path):
            hidden_states = np.squeeze(np.load(hidden_state_path))
            firing_rates = relu(hidden_states)
            firing_rates = normalize(firing_rates)
            all_rates.append(firing_rates)

    combined_rates = np.vstack(all_rates)  # Combine rates for PCA
    pca = PCA(n_components=3)
    pca.fit(combined_rates)  # Fit PCA on combined data

    # Transform and plot PCA results for each period
    colors = plt.cm.viridis(np.linspace(0, 1, len(periods)))
    for i, period in enumerate(periods):
        firing_rates = all_rates[i]
        pca_results = pca.transform(firing_rates)

        peaks_path = os.path.join(directory, f'peaks_{period}.npy')
        peaks = np.load(peaks_path) if os.path.exists(peaks_path) else []

        ax1.plot(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2], label=f'{int(np.round(1/period))} Hz', color=colors[i])
        for peak in peaks:
            ax1.scatter(pca_results[peak, 0], pca_results[peak, 1], pca_results[peak, 2], color='black', s=10)

        # Calculate and collect cycle lengths
        cycle_lengths = calculate_cycle_lengths(firing_rates, peaks)
        if cycle_lengths:
            means.append(np.mean(cycle_lengths))
            std_devs.append(np.std(cycle_lengths))

    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.set_zlabel('PCA Component 3')
    ax1.legend()

    # Plot mean vs. standard deviation of trajectory lengths
    ax2.scatter(means, std_devs, color='blue')
    for i, txt in enumerate(periods):
        ax2.annotate(f'{int(np.round(1/txt))} Hz', (means[i], std_devs[i]))
    ax2.set_xlabel('Mean Trajectory Length')
    ax2.set_ylabel('Standard Deviation of Trajectory Length')
    ax2.set_title('Mean vs. Standard Deviation of Trajectory Lengths')

    # Additional plotting for context cues vs. average inter-peak interval
    context_cues = []
    average_intervals = []
    colors = []  # Colors for the scatter plot points
    filename_pattern = re.compile(r'peaks_([0-9]+\.[0-9]+)\.npy')

    # List files in the directory
    for filename in os.listdir(directory):
        match = filename_pattern.match(filename)
        if match:
            period = float(match.group(1))
            file_path = os.path.join(directory, filename)
            peaks = np.load(file_path)

            # Calculate inter-peak intervals
            intervals = np.diff(peaks)
            if intervals.size > 0:
                average_interval = np.mean(intervals)
                context_cue = 0.1 / period

                # Store the data
                context_cues.append(context_cue)
                average_intervals.append(average_interval)

                # Check if the period is a trained period
                if period in periods:
                    colors.append('orange')  # Trained point
                else:
                    colors.append('gray')  # Untrained point

    ax3.scatter(context_cues, average_intervals, color=colors)
    for i, txt in enumerate(periods):
        if (0.1 / txt) in context_cues:
            ax3.scatter([0.1 / txt], [average_intervals[context_cues.index(0.1 / txt)]], color='orange', label='Trained Points' if i == 0 else "")
    ax3.set_xlabel('Context Cue Size (0.1 / Period)')
    ax3.set_ylabel('Average Inter-Peak Interval')
    ax3.set_title('Context Cue Size vs. Average Inter-Peak Interval')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'PCA_and_model_activity_stats.png'))  # Save plot

periods = [0.500, 0.333, 0.250, 0.200, 0.166, 0.143, 0.125]
data = load_and_process_data(periods)
pca_model = load_data_compute_pca_and_plot(periods, directory='model_outputs')
