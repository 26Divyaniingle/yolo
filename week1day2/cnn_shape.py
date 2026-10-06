import torch
import torch.nn as nn

x = torch.randn(1, 3, 640, 640)

print("Input shape:", x.shape)

conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=5,
    stride=1,
    padding=2
)

x = conv1(x)

print("After Conv1:", x.shape)

conv2 = nn.Conv2d(
    in_channels=32,
    out_channels=64,
    kernel_size=3,
    stride=2,
    padding=1
)

x = conv2(x)


print("After Conv2:", x.shape)

conv3 = nn.Conv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=2,
    padding=1
)

x = conv3(x)

print("After Conv3:", x.shape)