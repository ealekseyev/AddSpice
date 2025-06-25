import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoRatingModel(nn.Module):
    def __init__(self):
        super(VideoRatingModel, self).__init__()
        # Input: (batch, 1, 16, 192, 108)
        # But your input has (batch, frames, channels, width, height)
        # So we need to permute to (batch, channels, frames, width, height)

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))  # only spatial downsample

        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))  # global average pool

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, frames, channels, width, height)
        # Permute to (batch, channels, frames, width, height)
        x = x.permute(0, 2, 1, 3, 4)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # (batch, 64, 1, 1, 1)

        x = x.view(x.size(0), -1)  # flatten (batch, 64)
        x = torch.sigmoid(self.fc(x))  # output between 0 and 1
        return x * 10  # scale output to [0, 10]


# Example usage:
if __name__ == "__main__":
    model = VideoRatingModel()
    dummy_input = torch.randn(1, 16, 1, 192, 108)  # batch size 1
    output = model(dummy_input)
    print(output)  # tensor with shape (1, 1), values between 0 and 10
