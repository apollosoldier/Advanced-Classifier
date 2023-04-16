import torch
import torch.optim as optim
import copy

class EnsembleClassifier:
    def __init__(self, num_classifiers, base_classifier, num_classes):
        """
        Initialize the EnsembleClassifier object by defining the number of classifiers, the base classifier, and the number of classes.

        Input:
        @param - num_classifiers: the number of classifiers to be used in the ensemble
        @param - base_classifier: the base classifier to be used in the ensemble
        @param - num_classes: the number of classes to be classified

        Output:
        @return - None
        """
        self.num_classifiers = num_classifiers
        self.classifiers = [copy.deepcopy(base_classifier) for _ in range(num_classifiers)]
        self.num_classes = num_classes

    def train(self, data_loader, device, epochs, learning_rate, weight_decay):
        """
        Train the ensemble of classifiers.

        Input:
        @param - data_loader: the input data loader for training the ensemble
        @param - device: the device to be used for training
        @param - epochs: the number of epochs for training
        @param - learning_rate: the learning rate for training
        @param - weight_decay: the weight decay for training

        Output:
        @return - None
        """
        for i, classifier in enumerate(self.classifiers):
            classifier.to(device)
            optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(epochs):
                classifier.train()
                for inputs, labels in data_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    outputs = classifier(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

            self.classifiers[i] = classifier.cpu()

    def predict(self, data):
        """
        Make predictions using the ensemble of classifiers.

        Input:
        @param - data: the input data to be classified

        Output:
        @return - the predicted class labels
        """
        predictions = torch.zeros((len(data), self.num_classes))
        for classifier in self.classifiers:
            classifier.eval()
            outputs = classifier(data)
            predictions += outputs

        _, predicted_labels = torch.max(predictions, dim=1)
        return predicted_labels
