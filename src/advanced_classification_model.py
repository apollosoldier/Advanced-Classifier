from src.data_augmentation import DataAugmentation
from src.adaptive_training_strategy import AdaptiveTrainingStrategy
from src.base_classifier import BaseClassifier
from src.self_supervised_learning import SelfSupervisedLearning, SimCLR, ProjectionHead
from src.ensemble_classifier import EnsembleClassifier
import torch.optim as optim
from torchvision.models import resnet18

class AdvancedClassificationModel:
    def __init__(self, num_classifiers=5,num_classes = 10):
        """
        Initialize the AdvancedClassificationModel object by defining the data augmentation, adaptive training strategy, base classifier, self-supervised learning, and ensemble classifier.

        Input:
        @param - num_classifiers: the number of classifiers to be used in the ensemble learning

        Output:
        @return - None
        """
        self.num_classes = num_classes
        self.data_augmentation = DataAugmentation()
        self.adaptive_training_strategy = AdaptiveTrainingStrategy()
        self.base_classifier = BaseClassifier(self.num_classes)
        self.simclr_framework = SimCLR(temperature=0.5)
        self.projection_head = ProjectionHead(512, 128)

        self.self_supervised_learning = SelfSupervisedLearning(self.simclr_framework, self.data_augmentation, self.projection_head)
        self.ensemble_classifier = EnsembleClassifier(num_classifiers, self.base_classifier, num_classes=self.num_classes)

    def train(self, data_loader, device, epochs, learning_rate, weight_decay):
        """
        Train the advanced classification model using the ensemble of classifiers.

        Input:
        @param - data_loader: the input data loader for training the model
        @param - device: the device to be used for training
        @param - epochs: the number of epochs for training
        @param - learning_rate: the learning rate for training
        @param - weight_decay: the weight decay for training

        Output:
        @return - None
        
        """
        # Pretrain using self-supervised learning
        model = resnet18(pretrained=False, num_classes=10)
        optimizer = optim.Adam(list(model.parameters()) + list(self.projection_head.parameters()), lr=3e-4)

        print("Starting self-supervised pretraining...")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.self_supervised_learning.pretrain(self.base_classifier, data_loader, device, 1, optimizer)
        print("Self-supervised pretraining completed.")

        # Train the ensemble of classifiers
        print("Starting ensemble classifier training...")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.ensemble_classifier.train(data_loader, device, 1, learning_rate, weight_decay)
        print("Ensemble classifier training completed.")


    def predict(self, data):
        """
        Make predictions using the ensemble of classifiers.

        Input:
        @param - data: the input data to be classified

        Output:
        @return - the predicted class labels
        """
        # Make predictions using the ensemble of classifiers
        return self.ensemble_classifier.predict(data)

    def augment_data(self, data):
        """
        Apply data augmentation to the input data.

        Input:
        @param - data: the input data to be augmented

        Output:
        @return - the augmented data
        """
        # Apply data augmentation to the input data
        return self.data_augmentation.apply(data)

    def select_training_strategy(self, model, data_loader, device):
        """
        Select the best strategy for training the model given the data and device.

        Input:
        @param - model: the input model to be trained
        @param - data_loader: the input data loader for training the model
        @param - device: the device to be used for training

        Output:
        @return - the selected training strategy
        """
        # Select the best strategy for training the model given the data and device
        return self.adaptive_training_strategy.select_strategy(model, data_loader, device)
