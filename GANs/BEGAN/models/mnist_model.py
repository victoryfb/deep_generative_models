import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(64, 128 * 8 * 8)
        self.main = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, inputs):
        inputs = self.l1(inputs)
        inputs = inputs.view(inputs.shape[0], 128, 8, 8)
        return self.main(inputs)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Down sampling
        self.down = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.ReLU()
        )
        # Fully-connected layers
        self.down_size = 32 // 2
        down_dim = 64 * (32 // 2) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Up sampling
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, 3, 1, 1)
        )

    def forward(self, inputs):
        inputs = self.down(inputs)
        inputs = self.fc(inputs.view(inputs.size(0), -1))
        return self.up(
            inputs.view(inputs.size(0), 64, self.down_size, self.down_size))
