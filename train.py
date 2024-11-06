import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml
import os
from zrnn.models import ZemlianovaRNN, RNN
from zrnn.datasets import PulseStimuliDataset

def train(model, dataloader, optimizer, criterion, config, device):
    model.train()  # Set the model to training mode
    min_loss = float('inf')  # Initialize the minimum loss to a large value

    for epoch in range(config['training']['epochs']):
        total_loss = 0
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
            batch_size = inputs.shape[0]
            hidden = model.initHidden(batch_size).to(device)
            outputs = []

            for t in range(inputs.size(1)):  # process each time step
                output, hidden = model(inputs[:, t], hidden)
                outputs.append(output)

            outputs = torch.stack(outputs, dim=1)
            loss = criterion(outputs[..., 0], targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {avg_loss}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), config['training']['save_path'])
            print(f"Model saved with improvement at epoch {epoch+1} with loss {min_loss}")

        if avg_loss <= config['training']['early_stopping_loss']:
            print("Early stopping threshold reached.")
            break

    return model


def plot_results(model, dataloader, config, device):
    if not config['plotting']['enable']:
        return

    print('plotting some examples ...')

    # Ensure directory for plots exists
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for inputs, targets, periods in dataloader:
            # Ensure we're handling the device placement
            inputs, targets = inputs.to(device), targets.to(device)

            # Initialize hidden states for the batch
            hidden = model.initHidden(inputs.size(0)).to(device)
            outputs = []

            # Process each timestep in the sequence
            for t in range(inputs.size(1)):
                output, hidden = model(inputs[:, t, :], hidden)
                outputs.append(output)

            outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, features]
            outputs = outputs.squeeze(-1).cpu().numpy()  # Simplify if output features == 1
            targets = targets.squeeze(-1).cpu().numpy()
            inputs = inputs.squeeze(-1).cpu().numpy()

            plotted_periods = set()
            # Iterate over each sequence in the batch
            for i, period in enumerate(periods.cpu().numpy()):
                # Plot each period only once if they are the same in a batch
                if period in plotted_periods:
                    continue

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(outputs[i], label='Predictions')
                ax.plot(targets[i], label='Targets')
                ax.plot(inputs[i], label='Inputs')
                ax.legend()
                ax.set_title(f"Responses for Period {period:.3f} Seconds")
                plt.savefig(os.path.join(plot_dir, f"Period_{period:.3f}_seconds.png"))
                plt.close(fig)

                plotted_periods.add(period)

    print('done plotting')


def main(config_path='config.yaml', model_type=None):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if model_type:
        # Overriding model type from the command line
        config['model']['type'] = model_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Training with device:', device)

    dataset = PulseStimuliDataset(config['training']['periods'], size=config['training']['batch_size'], dt=config['model']['dt'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)

    if config['model']['type'] == "RNN":
        model = RNN(config['model']['input_dim'], config['model']['hidden_dim'], config['model']['output_dim'], config['model']['dt'], config['model']['tau']).to(device)
    else:
        model = ZemlianovaRNN(config['model']['input_dim'], config['model']['hidden_dim'], config['model']['output_dim'], config['model']['dt'], config['model']['tau'], config['model']['excit_percent'], sigma_rec=config['model']['sigma_rec']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()

    model = train(model, dataloader, optimizer, criterion, config, device)
    plot_results(model, dataloader, config, device)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
