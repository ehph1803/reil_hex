import torch
import torch.nn as nn
from joblib import load

class Actor(nn.Module):
    def __init__(self, n_actions, device, in_channels=1, kernel_size=3):
        super().__init__()

        self.device = device

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten()
        )

        self.lin = nn.Sequential(
            nn.Linear(64, n_actions),
            nn.Softmax()
        )

    def forward(self, X):
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        # print(len(tensor))
        tensor.unsqueeze_(-1)
        tensor = tensor.expand(len(tensor), len(tensor), 1)
        tensor = tensor.permute(2, 0, 1)
        # print(tensor)

        x = self.conv(tensor)
        x = torch.transpose(x, 0, 1)
        x = self.lin(x)

        return x.flatten()


actor_file = 'v14_hex_actor.a2c'

actor = load(actor_file)
torch.save(actor.state_dict(), 'actor_final_state_dict.torch')
