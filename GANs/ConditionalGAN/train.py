import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from models.fashion_mnist_model import Generator, Discriminator
from dataloader import get_data
from plot import save_loss_curve, save_real_fake

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataset = "FashionMNIST"
# Number of workers for dataloader
workers = 0
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 28
# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Number of training epochs
num_epochs = 100
save_epoch = 10
# Learning rate for optimizers
lr = 1e-4
# Hyper-parameter for Adam optimizers
beta1 = 0.5
beta2 = 0.999
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device, " will be used.\n")

# Create the dataloader
dataloader = get_data(dataset, batch_size, image_size, workers)
# Create the generator
netG = Generator().to(device)
# Create the Discriminator
netD = Discriminator().to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))
# Print the model
print(netG)
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        optimizerD.zero_grad()
        # Format batch
        real_images = data[0].to(device)
        real_labels = data[1].to(device)
        # train with real images
        real_validity = netD(real_images, real_labels)
        real_loss = criterion(real_validity,
                              torch.ones(batch_size, device=device))
        # train with fake images
        z = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_labels = torch.tensor(np.random.randint(0, 10, batch_size),
                                   dtype=torch.long, device=device)
        fake_images = netG(z, fake_labels)
        fake_validity = netD(fake_images, fake_labels)
        fake_loss = criterion(fake_validity,
                              torch.zeros(batch_size, device=device))
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()
        z = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_labels = torch.tensor(np.random.randint(0, 10, batch_size),
                                   dtype=torch.long, device=device)
        fake_images = netG(z, fake_labels)
        validity = netD(fake_images, fake_labels)
        g_loss = criterion(validity, torch.ones(batch_size, device=device))
        g_loss.backward()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}]\tLoss_D: {d_loss.item():.4f}\t \
            Loss_G: {g_loss.item():.4f}')

        # Save Losses for plotting later
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        # Save network weights.
        if (epoch + 1) % save_epoch == 0:
            torch.save({
                'netG': netG.state_dict(),
                'netD': netD.state_dict(),
            }, f'checkpoint/model_epoch_{dataset}_{epoch + 1}')

# Plot loss curve
save_loss_curve(f"result/loss_curve_{dataset}", G_losses, D_losses)

# Plot the real images and fake images
real_batch = next(iter(dataloader))
real_images = vutils.make_grid(
    real_batch[0].to(device)[:64],
    padding=5,
    normalize=True
)
real_images = np.transpose(real_images.cpu(), (1, 2, 0))
fake_images = np.transpose(img_list[-1], (1, 2, 0))
save_real_fake(f"result/fake_vs_real_{dataset}", real_images, fake_images)
