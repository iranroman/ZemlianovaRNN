import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from zrnn.models import ZemlianovaRNN
from zrnn.datasets import PulseStimuliDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('training with device', device)

def train(model, dataloader, optimizer, criterion, epochs, save_path):

    model.train()  # Set the model to training mode
    min_loss = float('inf')  # Initialize the minimum loss to a large value

    for epoch in range(epochs):

        total_loss = 0

        for inputs, targets in dataloader:

            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
  
            # Initialize hidden state            
            batch_size = inputs.shape[0]

            hidden = model.init_hidden(batch_size).to(device)
            model.neuron_mask = model.neuron_mask.to(device)

            # Forward pass
            outputs, hidden = model(inputs, hidden)

            # Compute loss (comparing all outputs to targets across all time steps)
            loss = torch.sqrt(criterion(outputs[...,0], targets))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}')

        # Check if the current average loss is the lowest and save the model if so
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with improvement at epoch {epoch+1} with loss {min_loss}")

def main(PERIODS, BATCH_SIZE, LR, EPOCHS, SAVE_PATH):
    
    # Initialize the model
    model = ZemlianovaRNN(input_dim=2, hidden_dim=500, output_dim=1, tau=10, sigma_rec=0.01).to(device)

    # Set up the DataLoader
    dataset = PulseStimuliDataset(PERIODS, size=BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Assuming the DataLoader and dataset are defined and loaded as previously shown
    train(model, dataloader, optimizer, criterion, EPOCHS, SAVE_PATH)


if __name__ == "__main__":
    PERIODS = [0.500, 0.333, 0.250, 0.200, 0.166, 0.143, 0.125]
    BATCH_SIZE = 32
    LR = 0.001
    EPOCHS = 10000
    SAVE_PATH = 'best_model.pth'
    main(PERIODS, BATCH_SIZE, LR, EPOCHS, SAVE_PATH)
