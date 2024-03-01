import torch
import torch.nn as nn
import numpy as np

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

class RND(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.target_net = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim, output_dim)),
            nn.ReLU()
        )

        self.predictor_net = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim, output_dim)),
            nn.ReLU()
        )

        self.predictor_optimizer = torch.optim.Adam(self.predictor_net.parameters(), lr=0.0001, weight_decay=1e-6)

    def forward(self, state):
        target = self.target_net(state)
        prediction = self.predictor_net(state)

        return target, prediction
    
    def compulate_intrinsic_reward(self, state):
        target, prediction = self.forward(state)
        prediction_error = torch.pow(target - prediction, 2).sum(dim=0)

        self.predictor_optimizer.zero_grad()
        prediction_loss = nn.MSELoss(target, prediction)
        prediction_loss.backward()
        self.predictor_optimizer.step()

        return prediction_error
    
    def step(self):
        pass        
        