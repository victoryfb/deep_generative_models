import matplotlib.pyplot as plt
import numpy as np


# Plot loss curve
def save_loss_curve(filepath, g_loss, d_loss):
    fig = plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss, label="G")
    plt.plot(d_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)


# Compare the real images and generated images
def save_real_fake(filepath, real_images, fake_images):
    fig = plt.figure(figsize=(15, 15))
    # Plot the real images
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(real_images)

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(fake_images)

    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
