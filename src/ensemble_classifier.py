import torch
import torch.optim as optim
import copy

class EnsembleClassifier:
    def __init__(self, num_classifiers, base_classifier, num_classes):
        self.num_classifiers = num_classifiers
        self.classifiers = [copy.deepcopy(base_classifier) for _ in range(num_classifiers)]
        self.num_classes = num_classes

    def train(self, data_loader, device, epochs, learning_rate, weight_decay):
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

        predictions = torch.zeros((len(data), self.num_classes))
        for classifier in self.classifiers:
            classifier.eval()
            outputs = classifier(data)
            predictions += outputs

        _, predicted_labels = torch.max(predictions, dim=1)
        return predicted_labels
