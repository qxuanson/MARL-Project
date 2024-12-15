import torch
import torch.nn as nn

class EnhancedQNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        # Fix the input dimensionality issue
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], 13, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(13, 13, kernel_size=3),
            nn.ReLU()
        )
        
        dummy_input = torch.randn(1, observation_shape[-1], observation_shape[0], observation_shape[1])  # Add batch dimension
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(1, -1).shape[1]  # Get the flattened dimension
        
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_shape)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if not present
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten while preserving batch dimension
        return self.fc(x)