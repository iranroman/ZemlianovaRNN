import os
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from zrnn.models import ZemlianovaRNN

N_RANDOM_PERIODS = 100

def generate_stimuli(period, T_onset, duration=52.0, dt=0.001, one_dur=0.01):
    t = np.arange(0, duration, dt)  # Time array from 0 to 52 seconds
    T_tones = np.arange(T_onset, T_onset + 1, period)  # Tone times within the first second after onset

    I_stim = np.zeros_like(t) # here the pulse input is zeros

    I_cc = np.zeros_like(t)
    I_cc[t >= T_onset] = 0.1 / period  # Continuous cue starting at T_onset

    z_t = np.zeros_like(t)
    z_t[t >= T_onset] = (np.cos(2 * np.pi * (t[t >= T_onset] - T_onset) / period) + 1) / 2  # Modulated target variable

    return t, I_stim, I_cc, z_t

   
def drive_model_and_save_outputs(model, periods, device, time_steps=52000, discard_steps=2000, save_dir='model_outputs'):
    # Create directory to save outputs if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate and process inputs
    for period in periods:
        t, I_stim, I_cc, z_t = generate_stimuli(period, 0.05)
        input_signal = np.stack([I_stim, I_cc], axis=1)

        # Convert to tensor and send to device
        input_tensor = torch.tensor(input_signal, dtype=torch.float32).unsqueeze(0).to(device)
        hidden = model.initHidden(1).to(device)

        # Process the input through the model with no_grad for evaluation
        outputs = []
        hidden_states = []
        with torch.no_grad():
            for t in range(time_steps):
                output, hidden = model(input_tensor[:, t, :], hidden)
                outputs.append(output.detach().cpu().numpy())
                if t >= discard_steps:
                    hidden_states.append(hidden.cpu().numpy())

        outputs = np.array(outputs).squeeze()
        valid_outputs = outputs[discard_steps:]  # Discard the first 2000 steps

        # Apply bandpass filter
        fs = 1 / 0.001  # Sampling frequency (1000 Hz, since dt = 0.001 s)
        low = 1 / (period + 0.1)  # Low frequency of the bandpass filter
        high = 1 / (period - 0.1)  # High frequency of the bandpass filter
        b, a = butter(N=2, Wn=[low, high], btype='band', fs=fs)
        filtered_outputs = filtfilt(b, a, valid_outputs)

        # Find peaks in the filtered outputs
        peaks, _ = find_peaks(filtered_outputs, height=0)

        # Determine window size for searching the highest peak in the original signal
        window_size = int((period / 2) / 0.001)  # Half period in terms of samples

        # Find the highest peak in the original signal near each detected peak in the filtered output
        true_peaks = []
        for peak in peaks:
            start = max(0, peak - window_size)
            end = min(len(valid_outputs), peak + window_size)
            true_peak = np.argmax(valid_outputs[start:end]) + start
            true_peaks.append(true_peak)

        # Plotting
        plt.figure(figsize=(25, 5))
        plt.plot(valid_outputs, label='Raw Output for Period: ' + str(period) + 's')
        plt.plot(true_peaks, valid_outputs[true_peaks], "x", label='Highest Peaks in Raw Output')
        plt.xlabel('Time Steps (after discard)')
        plt.ylabel('Model Activity')
        plt.title('Model Output for Period ' + str(period) + 's')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'plot_period_{period}.png'))  # Save plot
        plt.close()

        # Save data to files
        np.save(os.path.join(save_dir, f'valid_output_{period}.npy'), valid_outputs)
        np.save(os.path.join(save_dir, f'peaks_{period}.npy'), true_peaks)
        np.save(os.path.join(save_dir, f'hidden_states_{period}.npy'), np.array(hidden_states))
        print(f'finished driving model and saving activity for period {period}')

def sample_exponential_skew(num_samples, lam=5):
    u = np.random.rand(num_samples)
    skewed_samples = np.exp(-lam * (1 - u))
    return skewed_samples

def main(config_path='config.yaml', model_type=None):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    model = ZemlianovaRNN(config['model']['input_dim'], config['model']['hidden_dim'], config['model']['output_dim'], config['model']['dt'], config['model']['tau'], config['model']['excit_percent'], sigma_rec=config['model']['sigma_rec']).to(device)
    model.load_state_dict(torch.load(config['training']['save_path'], map_location=device))
    model.eval()
    
    # ADD a few extra randomly-generated periods
    drive_model_and_save_outputs(model, config['training']['periods']+[round(v+0.1,3) for v in list(sample_exponential_skew(N_RANDOM_PERIODS))], device)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
