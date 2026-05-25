from torchvision import transforms


def get_augmix_transform():
    """
    Simplified AugMix-style transform for CIFAR-10.

    This is not the full AugMix algorithm with Jensen-Shannon loss.
    It uses torchvision RandAugment as a practical robust augmentation baseline.
    """

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),

        transforms.RandAugment(
            num_ops=3,
            magnitude=9
        ),

        transforms.ToTensor(),
    ])

    return transform