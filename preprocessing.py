from torchvision import transforms


def get_transforms(img_size):
    """Get training and testing transforms with data augmentation"""

    # Enhanced data augmentation for better generalization
    try:
        from torchvision.transforms import RandAugment
        randaug = [RandAugment(num_ops=2, magnitude=9)]  # Increased magnitude
    except Exception:
        randaug = []

    train_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),  # Slightly larger resize
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # Reduced probability
        transforms.RandomRotation(15),  # Added rotation
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),  # Enhanced jitter
        *randaug,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')  # Adjusted
    ])

    test_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_tfms, test_tfms
