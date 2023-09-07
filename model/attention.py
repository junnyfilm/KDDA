import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels=32, reduction_ratio=16, version='cur'):
        super(SpatialAttention, self).__init__()

        self.version = version  # Add a version parameter
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # You can add conditionals here based on self.version to modify behavior if needed
        b, c, t = x.size()
        y = self.avg_pool(x).view(b, c, 1)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y

class CrossAttention(nn.Module):
    def __init__(self, input_size=4096, version='default'):
        super(CrossAttention, self).__init__()
        self.version = version  # Add a version parameter
        self.input_size = input_size  
        # same initialization as before
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def forward(self, x, context):
        # You can add conditionals here based on self.version to modify behavior if needed

        # same forward function as before
        Q = self.query(x)
        K = self.key(context)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.input_size ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        V = self.value(context)
        attn_output = torch.bmm(attn_weights, V)
        return attn_output