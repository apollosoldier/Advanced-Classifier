import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


class CustomBertClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes, hidden_size=768):
        super(CustomBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True
        )
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(encoding['token_type_ids'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train(model, train_dataloader, optimizer, device):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, val_dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask, token_type_ids)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Load and preprocess your dataset
def load_data(data_path):
    # Load your dataset in the required format:
    # [{'text': 'example text', 'label': 0}, {'text': 'another example', 'label': 1}, ...]
    # Replace this with the actual data loading process.
    return []


if __name__ == '__main__':
    data = load_data("path_to_your_data")

    train_data, val_data = train_test_split(data, test_size=0.2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = CustomBertClassifier(pretrained_model='bert-base-uncased', num_classes=2)
    model.to(device)

    max_len = 128
    batch_size = 16
    num_epochs = 3
    learning_rate = 2e-5

    train_dataset = CustomDataset(train_data, tokenizer, max_len)
    val_dataset = CustomDataset(val_data, tokenizer, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_dataloader, optimizer, device)
        acc = evaluate(model, val_dataloader, device)
        print(f"Validation accuracy: {acc:.4f}")

    torch.save(model.state_dict(), './custom_bert_classifier.pth')
