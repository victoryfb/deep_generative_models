import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is a batch_size x 74 x 1 x 1 matrix
            nn.ConvTranspose2d(in_channels=74, out_channels=1024,
                               kernel_size=1, stride=1,
                               bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            # input is a batch_size x 1024 x 1 x 1 matrix
            nn.ConvTranspose2d(in_channels=1024, out_channels=128,
                               kernel_size=7, stride=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            # input is a batch_size x 128 x 7 x 7 matrix
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            # input is a batch_size x 64 x 14 x 14 matrix
            nn.ConvTranspose2d(in_channels=64, out_channels=1,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Sigmoid()
            # output is a batch_size x 1 x 28 x 28 matrix
        )

    def forward(self, x):
        return self.main(x)


# The common part of discriminator and Q.
class FrontEnd(nn.Module):
    def __init__(self):
        super(FrontEnd, self).__init__()
        self.main = nn.Sequential(
            # input is a batch_size x 1 x 14 x 14 matrix
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2,
                      padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # input is a batch_size x 64 x 14 x 14 matrix
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.1, inplace=True),
            # input is a batch_size x 128 x 7 x 7 matrix
            nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=7,
                      bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.1, inplace=True),
            # output is a batch_size x 1024 x 1 x 1 matrix
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is a batch_size x 1024 x 1 x 1 matrix
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1),
            nn.Sigmoid()
            # output is a batch_size x 1 x 1 x 1 matrix
        )

    def forward(self, x):
        return self.main(x)


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.main = nn.Sequential(
            # input is a batch_size x 1024 x 1 x 1 matrix
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.1, inplace=True),
            # output is a batch_size x 128 x 1 x 1 matrix
        )
        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        x = self.main(x)
        # discrete latent code that represents the digit
        disc_logits = self.conv_disc(x).squeeze()
        # continuous latent code that is represented by two independent
        # Gaussian distribution
        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
