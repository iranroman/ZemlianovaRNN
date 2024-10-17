import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ZemlianovaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=500, output_dim=1, tau=0.01, sigma_rec=0.01, dt=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.sigma_rec = sigma_rec
        self.input_dim = input_dim
        self.dt = dt  # Time step size for integration

        # Define weights and biases
        W_rec_initial = torch.rand(hidden_dim, hidden_dim) * 0.001
        self.W_rec = nn.Parameter(W_rec_initial)
        with torch.no_grad():
            self.W_rec.fill_diagonal_(0)  # Ensuring no self-loops in recurrent connections

        self.W_in = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.001)
        self.b_rec = nn.Parameter(torch.zeros(hidden_dim))
        self.W_out = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.001)
        self.b_out = nn.Parameter(torch.zeros(output_dim))

        # Mask for excitatory and inhibitory neurons
        self.neuron_mask = torch.ones(hidden_dim).to(device)
        self.neuron_mask[int(0.8 * hidden_dim):] = -1

    def forward(self, input, hidden):
        outputs = []
        for t in range(input.size(1)):  # process each time step
            noise = torch.sqrt(torch.tensor(2 * self.tau * (self.sigma_rec ** 2))) * torch.randn_like(hidden)

            # Update hidden state with dt scaling in the dynamics
            hidden_update = (-hidden + torch.mm(input[:, t], self.W_in) + torch.mm(F.relu(hidden), torch.sqrt(self.W_rec ** 2) * self.neuron_mask[:, None]) + self.b_rec + noise)
            hidden = hidden + self.dt / self.tau * hidden_update

            # Compute output
            output = torch.mm(F.relu(hidden), self.W_out) + self.b_out
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim).to(device)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt, tau):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau = tau

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        hidden = F.relu((self.i2h(inputs) + self.h2h(hidden)) * (self.dt / self.tau)) # the model does not learn if dt!=0.01. WHY????
        output = self.h2o(hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)
