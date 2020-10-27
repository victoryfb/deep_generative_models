from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# Directory containing the data.
root = '../datasets/'


def get_data(dataset, batch_size, image_size=28, workers=0, train=True):
    # Get MNIST dataset.
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        dataset = dsets.MNIST(root + 'mnist/', train=train,
                              download=True, transform=transform)

    # Get SVHN dataset.
    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        split_type = 'train' if train else 'test'
        dataset = dsets.SVHN(root + 'svhn/', split=split_type,
                             download=True, transform=transform)

    # Get FashionMNIST dataset.
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        dataset = dsets.FashionMNIST(root + 'fashionmnist/', train=train,
                                     download=True, transform=transform)

    # Get CelebA dataset.
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        dataset = dsets.ImageFolder(root=root + 'celeba/', transform=transform)

    # Create dataloader.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=workers, pin_memory=True)
    return dataloader
