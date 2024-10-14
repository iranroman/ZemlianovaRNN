import numpy as np
import torch
from torch.utils.data import Dataset

def generate_stimuli(period, T_onset, duration=2.0, dt=0.001, one_dur=0.01):
    t = np.arange(0, duration, dt)
    T_tones = np.arange(0, 1, period)  # Tone times from 0 to 1 second in steps of 'period'
    
    I_stim = np.zeros_like(t)
    for tone in T_tones:
        start_time = T_onset + tone
        end_time = start_time + one_dur
        if end_time > T_onset + 1:
            end_time = T_onset + 1  # Ensure it does not exceed T_onset + 1 second
        I_stim[(t >= start_time) & (t < end_time)] = 1  # Activate stimulus during tone periods

    I_cc = np.zeros_like(t)
    I_cc[t >= T_onset] = 0.1 / period
    
    z_t = np.zeros_like(t) 
    z_t[t >= T_onset] = (np.cos(2 * np.pi * (t[t >= T_onset] - T_onset) / period) + 1) / 2
    return t, I_stim, I_cc, z_t

class PulseStimuliDataset(Dataset):
    def __init__(self, periods, min_onset=0, max_onset=0.1, duration=2.0, dt=0.001, one_dur=0.01, size=1000):
        self.periods = periods
        self.min_onset = min_onset
        self.max_onset = max_onset
        self.duration = duration
        self.dt = dt
        self.one_dur = one_dur
        self.size = size  # Specify the size of the dataset
    
    def __len__(self):
        return self.size  # Return the total number of samples
    
    def __getitem__(self, idx):
        # Randomly choose a period and T_onset for each item
        period = np.random.choice(self.periods)
        T_onset = np.random.uniform(self.min_onset, self.max_onset)
        
        # Generate stimuli
        _, I_stim, I_cc, z_t = generate_stimuli(period, T_onset, self.duration, self.dt, self.one_dur)
        
        # Convert numpy arrays to tensors
        I_stim = torch.tensor(I_stim, dtype=torch.float32)
        I_cc = torch.tensor(I_cc, dtype=torch.float32)
        z_t = torch.tensor(z_t, dtype=torch.float32)
        
        # Stack I_stim and I_cc to form the input tensor
        inputs = torch.stack([I_stim, I_cc], dim=1)
        
        return inputs, z_t
