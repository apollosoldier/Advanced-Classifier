import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


from .src.advanced_classification_model import AdvancedClassificationModel
from .src.data_augmentation import DataAugmentation

def main():

    num_classifiers = 5
    num_classes = 10
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    learning_rate = 0.001
    weight_decay = 1e-5


    data_augmentation = DataAugmentation()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize the advanced classification model
    model = AdvancedClassificationModel(num_classifiers=num_classifiers, num_classes=num_classes)

    # Train the model
    model.train(train_loader, device, epochs, learning_rate, weight_decay)

    # Test the model
    model.base_classifier.to(device)
    model.base_classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.base_classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')

if __name__ == '__main__':
    main()
