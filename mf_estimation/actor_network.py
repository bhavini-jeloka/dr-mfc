import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from .utils import *

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
    def __init__(self, grid, state_dim_actor, state_dim_critic, action_dim, policy_type="lcp_policy_9x9"):
        super().__init__()
        
        self.grid = grid
        self.actor = ActorNetwork(state_dim_actor, action_dim).to(device)
        self.critic = CriticNetwork(state_dim_critic, 1).to(device)
        #TODO: make this more general
        model_dir = get_latest_model_dir(policy_type)
        self.load_state_dict(torch.load(model_dir / 'model.pth', map_location=lambda storage, loc: storage))
    
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
        action_probs = self.actor(state)
        return action_probs.detach()
    
    def get_actions(self, combined_state, team_name, num_agent_list, estimated_mf=None):
        # Step 1: Collect local obs and compute indices
        local_obs_list = []
        mf_obs_list = []
        agent_indices = []

        for j in range(num_agent_list):
            key = f"agent_{j}-local-obs"
            local_obs = combined_state[team_name][key]
            if estimated_mf is not None:
                coord = tuple(np.argwhere(local_obs[..., 0])[0])
                idx = np.ravel_multi_index(coord, self.grid)
                mean_field_obs = estimated_mf[idx].reshape(1, *self.grid)
            else:
                mean_field_obs = combined_state["global-obs"].transpose(2, 0, 1)
            
            local_obs_list.append(local_obs.transpose(2, 0, 1))
            mf_obs_list.append(mean_field_obs)
        
        # Step 2: Convert to tensors
        local_obs_tensor = torch.tensor(local_obs_list, dtype=torch.float32)       # (N, H, W)
        mf_obs_tensor = torch.tensor(mf_obs_list, dtype=torch.float32)             # (N, H, W)

        # Step 3: Stack along channel dim
        state_tensor = torch.cat([local_obs_tensor, mf_obs_tensor], dim=1)       # (N, 2, H, W)

        # Step 4: Pass through policy and sample actions
        action_dists = Categorical(self.act(state_tensor))                         # (N, num_actions)
        actions = action_dists.sample()                                            # (N,)

        return actions.cpu().numpy()
    
    def get_est_based_actions(self, combined_state, team_name, num_agent_list, opp_mf_estimate=None):
        # Step 1: Collect local obs and compute indices
        local_obs_list = []
        mf_obs_list = []

        for j in range(num_agent_list):
            key = f"agent_{j}-local-obs"
            local_obs = combined_state[team_name][key]
            if opp_mf_estimate is not None: 
                coord = tuple(np.argwhere(local_obs != 0)[0]) 
                idx = np.ravel_multi_index(coord[:2], self.grid) 
                idx += coord[2] * self.grid[0] * self.grid[1] 
                mean_field_opp_est = opp_mf_estimate[idx].reshape(2, *self.grid)
                if team_name=="blue":
                    mean_field_self_est = combined_state["global-obs"].transpose(2, 0, 1)[:2]
                    mean_field_obs = np.concatenate([mean_field_self_est, mean_field_opp_est], axis=0)
                else:
                    mean_field_self_est = combined_state["global-obs"].transpose(2, 0, 1)[2:]
                    mean_field_obs = np.concatenate([mean_field_opp_est, mean_field_self_est], axis=0)
            else:
                mean_field_obs = combined_state["global-obs"].transpose(2, 0, 1)
            
            local_obs_list.append(local_obs.transpose(2, 0, 1))
            mf_obs_list.append(mean_field_obs)
        
        # Step 2: Convert to tensors
        local_obs_tensor = torch.tensor(local_obs_list, dtype=torch.float32)       # (N, H, W)
        mf_obs_tensor = torch.tensor(mf_obs_list, dtype=torch.float32)             # (N, H, W)

        # Step 3: Stack along channel dim
        state_tensor = torch.cat([local_obs_tensor, mf_obs_tensor], dim=1)       # (N, 2, H, W)
        # Step 4: Pass through policy and sample actions
        action_dists = Categorical(self.act(state_tensor))                         # (N, num_actions)
        actions = action_dists.sample()                                            # (N,)

        return actions.cpu().numpy()
