import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TrainingStrategy1:
    """
        Apply the first training strategy to the input model given the data and device.

        Input:
        - model: the input model to be trained
        - data_loader: the input data loader for training the model
        - device: the device to be used for training
        - optimizer: the optimizer to be used for training
        - criterion: the loss function to be used for training

        Output:
        - the average loss per batch during training
    """
    def apply(self, model, data_loader, device, optimizer, criterion):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / (i + 1)

class TrainingStrategy2:

        """
            Apply the second training strategy to the input model given the data and device.

            Input:
            - model: the input model to be trained
            - data_loader: the input data loader for training the model
            - device: the device to be used for training
            - optimizer: the optimizer to be used for training
            - criterion: the loss function to be used for training

            Output:
            - the average loss per batch during training
        """
    def apply(self, model, data_loader, device, optimizer, criterion):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            inputs = torch.nn.functional.dropout(inputs, p=0.2, training=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / (i + 1)


class AdaptiveTrainingStrategy:
       """
            Initialize the AdaptiveTrainingStrategy object by defining the list of training strategies to be used, their metric history, and the last selected strategy.

            Input:
            - None

            Output:
            - None
        """
    def __init__(self):
        self.strategies = {
            'strategy1': TrainingStrategy1(),
            'strategy2': TrainingStrategy2(),
        }
        self.metric_history = {
            'strategy1': [],
            'strategy2': [],
        }
        self.last_selected_strategy = None

    def select_strategy(self, model, data_loader, device):
        """
            Select the best strategy for training the model given the data and device.

            Input:
            - model: the input model to be trained
            - data_loader: the input data loader for training the model
            - device: the device to be used for training

            Output:
            - the selected training strategy
        """
        if self.last_selected_strategy is None:
            # If no strategy has been selected before, start with 'strategy1'
            return 'strategy1'
        else:
            # Compare the performance of both strategies based on their metric history
            avg_performance1 = sum(self.metric_history['strategy1']) / len(self.metric_history['strategy1'])
            avg_performance2 = sum(self.metric_history['strategy2']) / len(self.metric_history['strategy2'])

            # Select the strategy with the lowest average loss
            if avg_performance1 < avg_performance2:
                return 'strategy1'
            else:
                return 'strategy2'

    def apply_selected_strategy(self, selected_strategy, model, data_loader, device, optimizer, criterion):
        """
            Apply the selected training strategy to the model.

            Input:
            - selected_strategy: the selected training strategy
            - model: the input model to be trained
            - data_loader: the input data loader for training the model
            - device: the device to be used for training
            - optimizer: the optimizer to be used for training
            - criterion: the loss function to be used for training

            Output:
            - None
        """
        strategy = self.strategies[selected_strategy]
        strategy.apply(model, data_loader, device, optimizer, criterion)
        self.last_selected_strategy = selected_strategy

    def update_metric_history(self, selected_strategy, metric_value):
        """
            Update the metric history of the selected training strategy.

            Input:
            - selected_strategy: the selected training strategy
            - metric_value: the value of the metric to be updated

            Output:
            - None
        """
        self.metric_history[selected_strategy].append(metric_value)

    def adjust_learning_rate(self, optimizer, validation_loss):
        """
            Adjust the learning rate of the optimizer based on the validation loss.

            Input:
            - optimizer: the optimizer to be used for training
            - validation_loss: the validation loss of the current epoch

            Output:
            - None
        """
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        scheduler.step(validation_loss)

    def __call__(self, model, data_loader, device, optimizer, criterion, validation_loss):
        """
            Apply the adaptive training strategy to the model.

            Input:
            - model: the input model to be trained
            - data_loader: the input data loader for training the model
            - device: the device to be used for training
            - optimizer: the optimizer to be used for training
            - criterion: the loss function to be used for training
            - validation_loss: the validation loss of the current epoch

            Output:
            - None
        """
        selected_strategy = self.select_strategy(model, data_loader, device)
        self.apply_selected_strategy(selected_strategy, model, data_loader, device, optimizer, criterion)
        self.update_metric_history(selected_strategy, validation_loss)
        self.adjust_learning_rate(optimizer, validation_loss)
