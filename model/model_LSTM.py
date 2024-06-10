import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import wandb

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

run_name = 'LSTM_lr_0.001_epochs_150'

wandb.init(
    project = "Reddit_Sentiment_Analysis",
    entity = "kiri487",
    name = run_name,
    config = {
        "learning_rate": 0.001,
        "epochs": 150,
        "batch_size": 256,
    }
)

data = np.load('./dataset/data.npz')
train_data = data['train_data']
train_tags = data['train_tags']
test_data = data['test_data']
test_tags = data['test_tags']

train_tags = np.where(train_tags == -1, 2, train_tags)
test_tags = np.where(test_tags == -1, 2, test_tags)

train_dataset = TensorDataset(torch.LongTensor(train_data), torch.LongTensor(train_tags))
test_dataset = TensorDataset(torch.LongTensor(test_data), torch.LongTensor(test_tags))

train_loader = DataLoader(train_dataset, batch_size = wandb.config.batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = wandb.config.batch_size, shuffle = False)

class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size):
        super(SentimentAnalysisModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.lstm = nn.LSTM(16, 32, bidirectional = True, batch_first = True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(p = 0.2)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim = 1)
        x = torch.relu(self.fc1(h_n))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.softmax(self.fc3(x), dim = 1)
        return x

max_vocab_index = np.max(train_data) + 1
print(f"Detected maximum vocabulary index: {max_vocab_index}")

model = SentimentAnalysisModel(max_vocab_index).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = wandb.config.learning_rate)

num_epochs = wandb.config.epochs

def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average = 'weighted')
    recall = recall_score(all_labels, all_preds, average = 'weighted')
    f1 = f1_score(all_labels, all_preds, average = 'weighted')

    return accuracy, precision, recall, f1

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc = f"Epoch {epoch+1}/{num_epochs}", leave = False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        progress_bar.set_postfix(loss = loss.item())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    train_acc, train_precision, train_recall, train_f1 = evaluate(model, train_loader)
    test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader)

    wandb.log({
        "Epoch": epoch + 1,
        "Loss": epoch_loss,
        "Train Accuracy": train_acc,
        "Train Precision": train_precision,
        "Train Recall": train_recall,
        "Train F1": train_f1,
        "Test Accuracy": test_acc,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1": test_f1
    })

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, Test Accuracy: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

torch.save(model, run_name + '.pkl')
print("Model saved!")
wandb.finish()