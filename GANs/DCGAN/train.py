import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

from DCGAN.models.mnist_model import Generator, Discriminator
from DCGAN.models.DCGAN import DCGAN
from initialization import weights_init
from dataloader import get_data

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataset = "MNIST"
# Batch size during training
batch_size = 128
# Spatial size of training images.
image_size = 32
# Number of channels in the training images.
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Number of training epochs
epochs = 100
save_epoch = 10
# Learning rate for optimizers
lr = 0.0002
# Hyper-parameters for Adam optimizers
beta1 = 0.5
beta2 = 0.999

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, " will be used.\n")

# Create the dataloader
dataloader = get_data(dataset, batch_size, image_size)

# Plot some training images
real_batch = next(iter(dataloader))
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(
    vutils.make_grid(real_batch[0].to(device)[:64], padding=2,
                     normalize=True).cpu(), (1, 2, 0)))
plt.show()
plt.close(fig)

model = DCGAN(Generator, Discriminator, weights_init, epochs, batch_size, nc,
              image_size, save_epoch, nz, lr, beta1, beta2, 1, 0, device)

print("Starting Training Loop...")
model.train(dataloader, 'result/mnist')
