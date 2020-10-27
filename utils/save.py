import pickle

import matplotlib.pyplot as plt
import numpy as np


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


def save_fake_images(images, iteration, path):
    fig = plt.figure()
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(images, (1, 2, 0)))
    plt.savefig(f'{path}/fake_images_{iteration}', bbox_inches='tight')
    plt.close(fig)
