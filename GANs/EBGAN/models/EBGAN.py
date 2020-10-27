import torch
import torch.nn as nn
from torchvision import utils

from utils.save import save_loss, save_fake_images


class EBGAN:
    def __init__(self, Generator, Discriminator, weights_init, epochs,
                 batch_size, img_size, save_epoch, dim_z, lr, beta1,
                 beta2=0.999, lambda_pt=0.1, margin=1, device='cpu'):
        self.device = device

        self.G = Generator().to(device)
        self.D = Discriminator().to(device)
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.criterion = nn.MSELoss()

        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=lr,
                                            betas=(beta1, beta2))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr,
                                            betas=(beta1, beta2))

        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.save_epoch = save_epoch
        self.dim_z = dim_z
        self.lambda_pt = lambda_pt
        self.margin = margin

    def train(self, train_loader, result_root='result/mnist'):
        data_size = len(train_loader)
        iters = 0
        g_losses = []
        d_losses = []
        fixed_z = torch.randn(64, self.dim_z, device=self.device)

        for epoch in range(self.epochs):
            for i, (images, _) in enumerate(train_loader):
                b_size = images.shape[0]
                ############################
                # (1) Update D network
                ###########################
                # Train with all-real batch
                # Format batch
                real_images = images.to(self.device)
                # Forward pass real batch through D
                d_real, _ = self.D(real_images)
                # Calculate loss on all-real batch
                d_loss_real = self.criterion(d_real, real_images)

                # Train with all-fake batch
                # Generate batch of latent vectors
                z = torch.randn(b_size, self.dim_z, device=self.device)
                # Generate fake image batch with G
                fake_images = self.G(z)
                # Classify all fake batch with D
                d_fake, embeddings = self.D(fake_images.detach())
                # Calculate D's loss on the all-fake batch
                d_loss_fake = self.criterion(d_fake, fake_images.detach())
                # Calculate the gradients for this batch
                self.D.zero_grad()
                d_loss = d_loss_real
                if self.margin - d_loss_fake.item() > 0:
                    d_loss += self.margin - d_loss_fake
                d_loss.backward()
                # Update D
                self.d_optimizer.step()

                ############################
                # (2) Update G network
                ###########################
                # Since we just updated D, perform another forward pass of
                # all-fake batch through D.
                d_fake, embeddings = self.D(fake_images)
                # Calculate G's loss based on this output
                g_loss = self.criterion(d_fake, fake_images) \
                         + self.lambda_pt * self.pullaway_loss(embeddings)
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
                                              d_loss, g_loss)

                # Save the trained parameters
                if (epoch + 1) % self.save_epoch == 0:
                    self.save_model(epoch + 1, result_root)
                # Check the model by saving G's output on fixed_noise
                if (iters % 500 == 0) or (
                        (epoch == 100 - 1) and (i == len(train_loader) - 1)):
                    with torch.no_grad():
                        fake = self.G(fixed_z).detach().cpu()
                        fake_images = utils.make_grid(fake, padding=2,
                                                      normalize=True)
                        save_fake_images(fake_images, iters, result_root)

                iters += 1
        # Save the losses
        save_loss(d_losses, g_losses, result_root)

    @staticmethod
    def pullaway_loss(embeddings):
        norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
        normalized_emb = embeddings / norm
        similarity = torch.matmul(normalized_emb,
                                  normalized_emb.transpose(1, 0))
        batch_size = embeddings.size(0)
        loss_pt = (torch.sum(similarity) - batch_size) / (
                batch_size * (batch_size - 1))
        return loss_pt

    def _show_training_state(self, epoch, iteration, data_size, d_loss,
                             g_loss):
        print(f'[{epoch}/{self.epochs}][{iteration}/{data_size}]\t\
              Loss_D: {d_loss:.4f}\tLoss_G: {g_loss:.4f}')

    def save_model(self, epoch, path):
        torch.save(self.G.state_dict(),
                   f'{path}/checkpoint/generator_{epoch}.pkl')
        torch.save(self.D.state_dict(),
                   f'{path}/checkpoint/discriminator_{epoch}.pkl')
        print(f'Models save to {path}/')

    def load_model(self, discriminator_path, generator_path):
        self.D.load_state_dict(torch.load(discriminator_path))
        self.G.load_state_dict(torch.load(generator_path))
        print('Discriminator model loaded from {}-'.format(discriminator_path))
        print('Generator model loaded from {}.'.format(generator_path))
