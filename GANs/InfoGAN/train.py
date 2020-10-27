import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from models.mnist_model import Generator, FrontEnd, Discriminator, Q, \
    weights_init
from dataloader import get_data


class NegativeLogLikelihoodNormalDist:
    """
    Calculate the negative log likelihood of normal distribution.
    """
    def __call__(self, x, mu, var):
        log_likelihood = - 0.5 * (var.mul(2 * np.pi) + 1e-6).log()\
                         - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        return -(log_likelihood.sum(1).mean())


def noise_sample(n_disc_c, dim_disc_c, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.
    --------
    n_disc_c : Number of discrete latent code is used.
    dim_disc_c : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Number incompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """
    z = torch.randn(batch_size, n_z, 1, 1, device=device)
    idx = np.zeros((n_disc_c, batch_size))
    disc_c = None
    if n_disc_c != 0:
        disc_c = torch.zeros(batch_size, n_disc_c, dim_disc_c, device=device)
        for i in range(n_disc_c):
            idx[i] = np.random.randint(dim_disc_c, size=batch_size)
            disc_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        disc_c = disc_c.view(batch_size, -1, 1, 1)

    if n_con_c != 0:
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if n_disc_c != 0:
        noise = torch.cat((z, disc_c), dim=1)
    if n_con_c != 0:
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx


# Set random seed for reproducibility.
seed = 999
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Configuration
dataset = 'MNIST'
batch_size = 128
num_epochs = 100
learning_rate = 2e-4
beta1 = 0.5
beta2 = 0.999
save_epoch = 25
num_z = 62  # dimension of incompressible noise
num_disc_c = 1  # number of discrete latent code used
dim_disc_c = 10  # dimension of discrete latent code
num_cont_c = 2  # number of continuous latent code used

# Use GPU if available.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

dataloader = get_data(dataset, batch_size)

# Plot the training images.
# sample_batch = next(iter(dataloader))
# plt.figure(figsize=(10, 10))
# plt.axis("off")
# plt.imshow(np.transpose(vutils.make_grid(
#     sample_batch[0].to(device)[: 100], nrow=10, padding=2,
#     normalize=True).cpu(), (1, 2, 0)))
# plt.savefig('result/Training Images {}'.format(dataset))
# plt.close('all')

# Initialise the network.
netG = Generator().to(device)
netG.apply(weights_init)
print(netG)

fe = FrontEnd().to(device)
fe.apply(weights_init)
print(fe)

netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)

netQ = Q().to(device)
netQ.apply(weights_init)
print(netQ)

# Loss for discrimination between real and fake images.
criterionD = nn.BCELoss()
# Loss for discrete latent code.
criterionQ_dis = nn.CrossEntropyLoss()
# Loss for continuous latent code.
criterionQ_con = NegativeLogLikelihoodNormalDist()

# Adam optimiser is used.
optimD = optim.Adam(
    [{'params': fe.parameters()}, {'params': netD.parameters()}],
    lr=2e-4, betas=(beta1, beta2)
)
optimG = optim.Adam(
    [{'params': netG.parameters()}, {'params': netQ.parameters()}],
    lr=1e-3, betas=(beta1, beta2)
)

# Fixed Noise
z = torch.randn(100, num_z, 1, 1, device=device)
fixed_noise = z
if num_disc_c != 0:
    idx = np.arange(dim_disc_c).repeat(10)
    dis_c = torch.zeros(100, num_disc_c, dim_disc_c, device=device)
    for i in range(num_disc_c):
        dis_c[torch.arange(0, 100), i, idx] = 1.0

    dis_c = dis_c.view(100, -1, 1, 1)
    fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

if num_cont_c != 0:
    con_c = torch.rand(100, num_cont_c, 1, 1, device=device) * 2 - 1
    fixed_noise = torch.cat((fixed_noise, con_c), dim=1)

real_label = 1
fake_label = 0

# List variables to store results pf training.
img_list = []
G_losses = []
D_losses = []

print("-" * 25)
print("Starting Training Loop...\n")
print(
    'Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(
        dataset) % (num_epochs, batch_size, len(dataloader)))
print("-" * 25)

start_time = time.time()
iters = 0

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    for i, (data, _) in enumerate(dataloader, 0):
        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)

        # Updating FrontEnd and Discriminator
        optimD.zero_grad()
        # Real data
        label = torch.full((b_size,), real_label, dtype=torch.float,
                           device=device)
        output1 = fe(real_data)
        probs_real = netD(output1).view(-1)
        loss_real = criterionD(probs_real, label)
        # Calculate gradients.
        loss_real.backward()

        # Fake data
        label.fill_(fake_label)
        noise, idx = noise_sample(num_disc_c, dim_disc_c, num_cont_c, num_z,
                                  b_size, device)
        fake_data = netG(noise)
        output2 = fe(fake_data.detach())
        probs_fake = netD(output2).view(-1)
        loss_fake = criterionD(probs_fake, label)
        # Calculate gradients.
        loss_fake.backward()
        # Net Loss for the discriminator
        D_loss = loss_real + loss_fake
        # Update parameters
        optimD.step()

        # Updating Generator and Q
        optimG.zero_grad()
        # Fake data treated as real.
        output = fe(fake_data)
        label.fill_(real_label)
        probs_fake = netD(output).view(-1)
        gen_loss = criterionD(probs_fake, label)

        q_logits, q_mu, q_var = netQ(output)
        target = torch.LongTensor(idx).to(device)
        # Calculating loss for discrete latent code.
        dis_loss = 0
        for j in range(num_disc_c):
            dis_loss += criterionQ_dis(q_logits[:, j * 10: j * 10 + 10],
                                       target[j])

        # Calculating loss for continuous latent code.
        con_loss = 0
        if num_cont_c != 0:
            con_loss = criterionQ_con(noise[:,
                                      num_z + num_disc_c *
                                      dim_disc_c:].view(-1, num_cont_c), q_mu,
                                      q_var) * 0.1

        # Net loss for generator.
        G_loss = gen_loss + dis_loss + con_loss
        # Calculate gradients.
        G_loss.backward()
        # Update parameters.
        optimG.step()

        # Check progress of training.
        if i != 0 and i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch + 1, num_epochs, i, len(dataloader),
                     D_loss.item(), G_loss.item()))

        # Save the losses for plotting.
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

        iters += 1

    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" % (epoch + 1, epoch_time))
    # Generate image after each epoch to check performance of the generator.
    # Used for creating animated gif later.
    with torch.no_grad():
        gen_data = netG(fixed_noise).detach().cpu()
    img_list.append(
        vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))

    # Generate image to check performance of generator.
    if (epoch + 1) == 1 or (epoch + 1) == num_epochs / 2:
        with torch.no_grad():
            gen_data = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(
            vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True),
            (1, 2, 0)))
        plt.savefig("result/Epoch_%d {}".format(dataset) % (epoch + 1))
        plt.close('all')

    # Save network weights.
    if (epoch + 1) % save_epoch == 0:
        torch.save({
            'netG': netG.state_dict(),
            'fe': fe.state_dict(),
            'netD': netD.state_dict(),
            'netQ': netQ.state_dict(),
            'optimD': optimD.state_dict(),
            'optimG': optimG.state_dict(),
        }, 'checkpoint/model_epoch_%d_{}'.format(dataset) % (
                epoch + 1))

training_time = time.time() - start_time
print("-" * 50)
print('Training finished!\nTotal Time for Training: %.2fm' % (
        training_time / 60))
print("-" * 50)

# Generate image to check performance of trained generator.
with torch.no_grad():
    gen_data = netG(fixed_noise).detach().cpu()
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(
    vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1, 2, 0)))
plt.savefig("Epoch_%d_{}".format(dataset) % num_epochs)

# Save network weights.
torch.save({
    'netG': netG.state_dict(),
    'fe': fe.state_dict(),
    'netD': netD.state_dict(),
    'netQ': netQ.state_dict(),
    'optimD': optimD.state_dict(),
    'optimG': optimG.state_dict(),
}, 'checkpoint/model_final_{}'.format(dataset))

# Plot the training losses.
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("result/Loss Curve {}".format(dataset))

# Animation showing the improvements of the generator.
fig = plt.figure(figsize=(10, 10))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in
       img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000,
                                 blit=True)
anim.save('result/infoGAN_{}.gif'.format(dataset), dpi=80,
          writer='imagemagick')
plt.show()
