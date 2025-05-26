import torch
import torch.nn as nn
import numpy as np
from ..utils import *

# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, num_output):
        super().__init__()
        
        # CNN for the first input channel
        self.first_layer_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # CNN for the remaining input channels
        self.remaining_layers_cnn = nn.Sequential(
            nn.Conv2d(input_shape[0] - 1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Fully connected layers
        combined_dim = (16 * input_shape[1] * input_shape[2]) + (32 * input_shape[1] * input_shape[2])
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, num_output)
        )
        
        # Apply orthogonal initialization
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Linear) and module.out_features == self.fc[-1].out_features:
                nn.init.orthogonal_(module.weight, gain=0.01)
            else:
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Separate the first channel from the remaining channels
        first_channel = x[:, :1, :, :]  # First channel
        remaining_channels = x[:, 1:, :, :]  # Remaining channels

        # Pass through respective CNNs
        first_output = self.first_layer_cnn(first_channel)
        first_output = first_output.view(first_output.size(0), -1)
        
        remaining_output = self.remaining_layers_cnn(remaining_channels)
        remaining_output = remaining_output.view(remaining_output.size(0), -1)

        # Concatenate outputs and pass through fully connected layers
        combined_output = torch.cat((first_output, remaining_output), dim=1)
        x = self.fc(combined_output)

        return torch.softmax(x, dim=-1)
    
class CriticNetwork(nn.Module):
    def __init__(self, input_shape, num_output):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * input_shape[1] * input_shape[2], 128),
            nn.Tanh(),
            nn.Linear(128, num_output)
        )

        # Apply orthogonal initialization
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # Check if it's the output layer (last Linear layer in self.fc)
            if isinstance(module, nn.Linear) and module.out_features == self.fc[-1].out_features:
                # Apply orthogonal initialization with gain 1 for the output layer
                nn.init.orthogonal_(module.weight, gain=1.0)
            else:
                # Apply orthogonal initialization with gain sqrt(2) for other layers
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            
            # Set biases to zero
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim_actor, state_dim_critic, action_dim, policy_type="lcp_policy"):
        super().__init__()
        
        self.actor = ActorNetwork(state_dim_actor, action_dim).to(device)
        self.critic = CriticNetwork(state_dim_critic, 1).to(device)
        
        model_dir = get_latest_model_dir(policy_type)
        self.load_state_dict(torch.load(model_dir / 'model.pth', map_location=lambda storage, loc: storage))
    
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
        action_probs = self.actor(state)
        return action_probs.detach()
