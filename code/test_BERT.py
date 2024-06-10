import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RedditDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text': text
        }

class SentimentAnalysisModel(nn.Module):
    def __init__(self):
        super(SentimentAnalysisModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = output[1]
        x = self.fc1(pooled_output)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the model
model_path = 'BERT_lr_2e-5_epochs_5.pkl'
model = torch.load(model_path)
model.to(device)
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Directory containing JSON files
input_dir = './JSON'
output_dir = './Prediction'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each JSON file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('.json', '.csv'))
        
        # Load JSON file
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        sentences = [item['sentence'] for item in data]

        # Create DataLoader
        dataset = RedditDataset(sentences, tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        # Make predictions
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.tolist())

        # Save predictions
        output_data = []
        for i, sentence in enumerate(sentences):
            output_data.append({'sentence': sentence, 'prediction': predictions[i]})

        df = pd.DataFrame(output_data)
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
