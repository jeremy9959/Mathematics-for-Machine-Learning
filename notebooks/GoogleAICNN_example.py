import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (grayscale), 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 input channels, 16 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layers:
        # The input features size for the first fully connected layer depends on the input image size
        # and pooling operations. For a 32x32 input image, it is 16*5*5 = 400 features.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 output classes (for digits 0-9)

    def forward(self, x):
        # Convolutional -> ReLU -> MaxPool
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # Convolutional -> ReLU -> MaxPool
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        # Fully connected -> ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Final output layer
        x = self.fc3(x)
        return x

# Example of how to use the model:
net = LeNet()
print(net)
