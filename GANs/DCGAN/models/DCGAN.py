import os
import pickle

import torch
import torch.nn as nn
from torchvision import utils
import numpy as np
import matplotlib.pyplot as plt


class DCGAN:
    def __init__(self, Generator, Discriminator, weights_init, epochs,
                 batch_size, num_channels, img_size, save_epoch, dim_z,
                 lr, beta1, beta2=0.999, real_label=1, fake_label=0,
                 device='cpu'):
        self.device = device

        self.G = Generator().to(device)
        self.D = Discriminator().to(device)
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.criterion = nn.BCELoss()

        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=lr,
                                            betas=(beta1, beta2))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr,
                                            betas=(beta1, beta2))

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.img_size = img_size
        self.save_epoch = save_epoch
        self.dim_z = dim_z
        self.real_label = real_label
        self.fake_label = fake_label

    def train(self, train_loader, result_root='result/mnist'):
        if not os.path.isdir(result_root):
            os.mkdir(result_root)

        data_size = len(train_loader)
        iters = 0
        g_losses = []
        d_losses = []
        fixed_z = torch.randn(64, self.dim_z, 1, 1, device=self.device)

        for epoch in range(self.epochs):
            for i, (images, _) in enumerate(train_loader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with all-real batch
                # Format batch
                real_images = images.to(self.device)
                real_labels = torch.full((self.batch_size,), self.real_label,
                                         dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                outputs = self.D(real_images).view(-1)
                # Calculate loss on all-real batch
                d_loss_real = self.criterion(outputs, real_labels)
                real_score = outputs.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                z = torch.randn(self.batch_size, self.dim_z, 1, 1,
                                device=self.device)
                # Generate fake image batch with G
                fake_images = self.G(z)
                fake_labels = torch.full((self.batch_size,), self.fake_label,
                                         dtype=torch.float, device=self.device)
                # Classify all fake batch with D
                outputs = self.D(fake_images.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                d_loss_fake = self.criterion(outputs, fake_labels)
                fake_score_1 = outputs.mean().item()
                # Calculate the gradients for this batch
                self.D.zero_grad()
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                # Update D
                self.d_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                # Since we just updated D, perform another forward pass of
                # all-fake batch through D.
                outputs = self.D(fake_images).view(-1)
                # Calculate G's loss based on this output
                g_loss = self.criterion(outputs, real_labels)
                fake_score_2 = outputs.mean().item()
                # Calculate gradients for G
                self.G.zero_grad()
                g_loss.backward()
                # Update G
                self.g_optimizer.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

                # Output training states
                if i % 50 == 0:
                    self._show_training_state(epoch, i, data_size,
                                              d_loss, g_loss, real_score,
                                              fake_score_1, fake_score_2)

                # Save the trained parameters
                if (epoch + 1) % self.save_epoch == 0:
                    self.save_model(epoch+1, result_root)
                # Check the model by saving G's output on fixed_noise
                if (iters % 500 == 0) or (
                        (epoch == 100 - 1) and (i == len(train_loader) - 1)):
                    with torch.no_grad():
                        fake = self.G(fixed_z).detach().cpu()
                        fake_images = utils.make_grid(fake, padding=2,
                                                      normalize=True)
                        self.save_fake_images(fake_images, iters, result_root)

                iters += 1

        # Save the losses
        self.save_loss(d_losses, g_losses, result_root)

    def _show_training_state(self, epoch, iteration, data_size, d_loss,
                             g_loss, real_score, fake_score_1, fake_score_2):
        print(f'[{epoch}/{self.epochs}][{iteration}/{data_size}]\t\
              Loss_D: {d_loss:.4f}\tLoss_G: {g_loss:.4f}\t\
              D(x): {real_score:.4f}\t\
              D(G(z)): {fake_score_1:.4f} /{fake_score_2:.4f}')

    @staticmethod
    def save_loss(d_losses, g_losses, path):
        with open(f'{path}/losses.pkl', 'wb') as f:
            pickle.dump({'d_losses': d_losses, 'g_losses': g_losses}, f)
        print(f'Losses save to {path}/losses.pkl')
        fig = plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{path}/loss_curve.png", bbox_inches='tight')
        plt.close(fig)

    def save_model(self, epoch, path):
        torch.save(self.G.state_dict(), f'{path}/generator_{epoch}.pkl')
        torch.save(self.D.state_dict(), f'{path}/discriminator_{epoch}.pkl')
        print(f'Models save to {path}/')

    def load_model(self, discriminator_path, generator_path):
        self.D.load_state_dict(torch.load(discriminator_path))
        self.G.load_state_dict(torch.load(generator_path))
        print('Discriminator model loaded from {}-'.format(discriminator_path))
        print('Generator model loaded from {}.'.format(generator_path))

    @staticmethod
    def save_fake_images(images, iteration, path):
        fig = plt.figure()
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(images, (1, 2, 0)))
        plt.savefig(f'{path}/fake_images_{iteration}', bbox_inches='tight')
        plt.close(fig)
