import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from GANs.SSGAN.models.mnist_model import Generator, Discriminator
from initialization import weights_init
from dataloader import get_data
from plot import save_loss_curve, save_real_fake

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataset = "MNIST"
# Number of workers for dataloader
workers = 0
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 32
# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 32
# Size of feature maps in discriminator
ndf = 32
# Number of training epochs
num_epochs = 100
save_epoch = 10
# Learning rate for optimizers
lr = 0.0002
# Hyper-parameters for Adam optimizers
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
netG = Generator(nc, nz, ngf).to(device)
# Create the Discriminator
netD = Discriminator(nc, ndf).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)
netD.apply(weights_init)
# Print the model
print(netG)
print(netD)

criterionD = nn.CrossEntropyLoss()  # binary cross-entropy
criterionG = nn.MSELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_data = data[0].to(device)
        real_label = data[1].to(device)
        label = torch.full((batch_size,), real_label, dtype=torch.float,
                           device=device)
        # Forward pass real batch through D
        output = netD(real_data).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterionD(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake
        # batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print(
                '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader),
                   errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or (
                (epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

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
