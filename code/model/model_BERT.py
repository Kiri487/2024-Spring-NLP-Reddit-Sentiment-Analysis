import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

run_name = 'BERT_lr_2e-5_epochs_5'

wandb.init(
    project = "Reddit_Sentiment_Analysis",
    entity = "kiri487",
    name = run_name,
    config = {
        "learning_rate": 2e-5,
        "epochs": 5,
        "batch_size": 16,
    }
)

class RedditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype = torch.long)
        }

df = pd.read_csv('./dataset/Reddit_Data.csv')

df.dropna(inplace = True)

df['category'] = df['category'].apply(lambda x: 2 if x == -1 else x)

comments = df['clean_comment'].tolist()
categories = df['category'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(comments, categories, test_size = 0.2, random_state = 42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_data = RedditDataset(train_texts, train_labels, tokenizer)
val_data = RedditDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_data, batch_size = wandb.config.batch_size, shuffle = True)
val_loader = DataLoader(val_data, batch_size = wandb.config.batch_size, shuffle = False)

class SentimentAnalysisModel(nn.Module):
    def __init__(self):
        super(SentimentAnalysisModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask = attention_mask)
        pooled_output = output[1]
        x = self.fc1(pooled_output)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SentimentAnalysisModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = wandb.config.learning_rate)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs, batch['attention_mask'].to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average = 'weighted')
    recall = recall_score(all_labels, all_preds, average = 'weighted')
    f1 = f1_score(all_labels, all_preds, average = 'weighted')
    return accuracy, precision, recall, f1

for epoch in range(wandb.config.epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc = f"Epoch {epoch+1}/{wandb.config.epochs}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    train_acc, train_precision, train_recall, train_f1 = evaluate(model, train_loader)
    test_acc, test_precision, test_recall, test_f1 = evaluate(model, val_loader)

    wandb.log({
        "Epoch": epoch + 1,
        "Loss": total_loss / len(train_loader),
        "Train Accuracy": train_acc,
        "Train Precision": train_precision,
        "Train Recall": train_recall,
        "Train F1": train_f1,
        "Test Accuracy": test_acc,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1": test_f1
    })

    print(f"Epoch {epoch+1}/{wandb.config.epochs}, Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, Test Accuracy: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

torch.save(model, run_name + '.pkl')
print("Model saved!")
wandb.finish()