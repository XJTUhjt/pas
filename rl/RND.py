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
    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
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

        # 注意：目标网络的参数不需要梯度
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.predictor_optimizer = torch.optim.Adam(self.predictor_net.parameters(), lr=0.0001, weight_decay=1e-6)

    def forward(self, state):
        target = self.target_net(state)
        prediction = self.predictor_net(state)
        return target, prediction
    
    def compute_intrinsic_reward(self, state):
        """
        计算内在奖励
        Args:
            state (torch.Tensor): 当前状态
        Returns:
            torch.Tensor: 预测误差作为内在奖励
        """
        with torch.no_grad():
            target = self.target_net(state)
        prediction = self.predictor_net(state)
        prediction_error = (target - prediction).pow(2).mean(1)  # 计算每个样本的均方误差

        # 计算损失并更新预测网络
        self.predictor_optimizer.zero_grad()
        prediction_loss = prediction_error.mean()  # 对所有样本的误差求平均，作为损失
        prediction_loss.backward()
        self.predictor_optimizer.step()

        # 返回预测误差作为内在奖励
        return prediction_error.detach()