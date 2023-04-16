import random
from torchvision import transforms

class DataAugmentation:
    def __init__(self):
        """
        Initialize the DataAugmentation object by defining the list of data augmentation techniques to be used.

        Input:
        @param - None

        Output:
        @return - None
        """
        # Define data augmentation strategies here
        self.transformations = transforms.Compose([
            transforms.RandomHorizontalFlip(),    # Flip the input image horizontally with a probability of 0.5
            transforms.RandomVerticalFlip(),      # Flip the input image vertically with a probability of 0.5
            transforms.RandomRotation(degrees=15),# Rotate the input image randomly by an angle between -15 to 15 degrees
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Adjust the brightness, contrast, saturation, and hue of the input image randomly
            transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)), # Crop the input image to a random size and aspect ratio, and then resize it to 32x32
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5), # Apply a Gaussian blur to the input image with a probability of 0.5
            transforms.RandomGrayscale(p=0.1),   # Convert the input image to grayscale with a probability of 0.1
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Apply a random perspective transformation to the input image with a probability of 0.5
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10), # Apply a random affine transformation to the input image with a given degree of freedom
            transforms.ToTensor(),               # Convert the input image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the input image with mean and standard deviation
        ])

    def __getattr__(self, name):
        """
        Get the attribute from the transformations if it exists.

        Input:
        @param - name: the name of the attribute to be retrieved

        Output:
        @return - the attribute if it exists, otherwise raise an AttributeError
        """
        if hasattr(self.transformations, name):
            return getattr(self.transformations, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __call__(self, data):
        """
        Apply the data augmentation techniques to the input data.

        Input:
        @param - data: the input data to be augmented

        Output:
        @return - the augmented data
        """
        # Call the apply method when the instance is called as a function
        return self.apply(data)

    def apply(self, data):
        """
        Apply the data augmentation techniques to the input data.

        Input:
        @param - data: the input data to be augmented

        Output:
        @return - the augmented data
        """
        # Apply data augmentation to the input data
        return self.transformations(data)
