import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ZemlianovaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt, tau, excit_percent, sigma_rec=0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau = tau
        self.excit_percent = excit_percent

        # Initialize weights and biases for input to hidden layer
        self.w_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b_ih = nn.Parameter(torch.Tensor(hidden_size))

        # Initialize weights and biases for hidden to hidden layer
        self.w_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))

        # Initialize weights and biases for hidden to output layer
        self.w_ho = nn.Parameter(torch.Tensor(output_size, hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(output_size))

        # Initialize all weights and biases
        self.init_weights()

        # Create masks for zeroing diagonal of w_hh and contain EI ratio
        self.zero_diag_mask = torch.ones(hidden_size, hidden_size) - torch.eye(hidden_size)
        self.EI_mask = torch.ones(hidden_size).to(device)
        self.EI_mask[int(self.excit_percent * hidden_size):] = -1


    def init_weights(self):
        # Initialize weights using xavier uniform and biases to zero
        nn.init.xavier_normal_(self.w_ih)
        nn.init.constant_(self.b_ih, 0)
        nn.init.xavier_normal_(self.w_hh)
        nn.init.constant_(self.b_hh, 0)
        nn.init.xavier_normal_(self.w_ho)
        nn.init.constant_(self.b_ho, 0)

    def forward(self, inputs, hidden):

        # Zero out diagonal of w_hh each forward pass
        w_hh_no_diag = self.w_hh * self.zero_diag_mask.to(self.w_hh.device)
        w_hh_no_diag_p = torch.abs(w_hh_no_diag)
        w_hh_EI = w_hh_no_diag_p * self.EI_mask

        # Compute the hidden state
        hidden_input = torch.matmul(inputs, self.w_ih.t()) + self.b_ih # removing this bias makes learning hard/impossible 
        hidden_hidden = torch.matmul(F.relu(hidden), w_hh_EI.t()) + self.b_hh
        hidden = (-hidden + hidden_input + hidden_hidden)  * (self.dt / self.tau)

        # Compute the output
        output = torch.matmul(F.relu(hidden), self.w_ho.t()) + self.b_ho
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)



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
