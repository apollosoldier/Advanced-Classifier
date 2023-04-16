import random
from torchvision import transforms

class DataAugmentation:
    def __init__(self):
        # Define data augmentation strategies here
        self.transformations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getattr__(self, name):
        # Get the attribute from the transformations if it exists
        if hasattr(self.transformations, name):
            return getattr(self.transformations, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __call__(self, data):
        # Call the apply method when the instance is called as a function
        return self.apply(data)


    def apply(self, data):
        # Apply data augmentation to the input data
        return self.transformations(data)
