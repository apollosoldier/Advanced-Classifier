import torch
import torch.nn.functional as F
import torch.optim as optim

class SimCLR:
    def __init__(self, temperature=0.5):
        """
        Initialize the SimCLR object by defining the temperature hyperparameter.

        Input:
        @param - temperature: the temperature hyperparameter

        Output:
        @return - None
        """
        self.temperature = temperature

    def loss(self, z_i, z_j):
        """
        Calculate the contrastive loss between two sets of embeddings.

        Input:
        @param - z_i: the embeddings of the first set
        @param - z_j: the embeddings of the second set

        Output:
        @return - the contrastive loss
        """
        batch_size = z_i.size(0)
        z = torch.cat((z_i, z_j), dim=0)

        similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        similarity_matrix = similarity_matrix / self.temperature

        labels = torch.arange(batch_size).to(z.device)
        labels = torch.cat((labels, labels), dim=0)

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def __call__(self, z_i, z_j):
        """
        Calculate the contrastive loss between two sets of embeddings.

        Input:
        @param - z_i: the embeddings of the first set
        @param - z_j: the embeddings of the second set

        Output:
        @return - the contrastive loss
        """
        return self.loss(z_i, z_j)

    def __getattr__(self, name):
        """
        Get the attribute from the class if it exists.

        Input:
        @param - name: the name of the attribute

        Output:
        @return - the attribute
        """
        return getattr(self.simclr, name)


class SelfSupervisedLearning:
    def __init__(self, ssl_framework, data_augmentation, projection_head):
        """
        Initialize the SelfSupervisedLearning object by defining the SSL framework, data augmentation, and projection head.

        Input:
        @param - ssl_framework: the SSL framework to be used
        @param - data_augmentation: the data augmentation to be used
        @param - projection_head: the projection head to be used

        Output:
        @return - None
        """
        self.ssl_framework = ssl_framework
        self.data_augmentation = data_augmentation
        self.projection_head = projection_head

    def pretrain(self, model, data_loader, device, epochs, optimizer):
        """
        Pretrain the model using self-supervised learning.

        Input:
        @param - model: the model to be pretrained
        @param - data_loader: the input data loader for pretraining
        @param - device: the device to be used for pretraining
        @param - epochs: the number of epochs for pretraining
        @param - optimizer: the optimizer for pretraining

        Output:
        @return - metrics: a dictionary containing loss, accuracy, and MSE for the current epoch
        """
        model = model.to(device)
        self.projection_head = self.projection_head.to(device)

        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        mse = 0

        for epoch in range(epochs):
            model.train()
            self.projection_head.train()

            for data, target in data_loader:
                optimizer.zero_grad()

                # Apply data augmentation to the same image twice
                x_i = self.data_augmentation.apply(data).to(device)
                x_j = self.data_augmentation.apply(data).to(device)

                # Extract features using the model
                h_i = model(x_i)
                h_j = model(x_j)

                # Obtain the projections using the projection head
                z_i = self.projection_head(h_i)
                z_j = self.projection_head(h_j)

                # Calculate the self-supervised learning loss
                loss = self.ssl_framework.loss(z_i, z_j)

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

                # Update metrics
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                correct_predictions += (z_i.argmax(1) == z_j.argmax(1)).sum().item()
                mse += torch.mean((z_i - z_j) ** 2).item()

        metrics = {
            'loss': total_loss / total_samples,
            'accuracy': correct_predictions / total_samples,
            'mse': mse / len(data_loader)
        }

        return metrics
                
class ProjectionHead(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ProjectionHead, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)


    def forward(self, x):
        return self.linear(x)
