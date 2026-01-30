"""
Complete training script - No imports needed!
Just save this file and run: python train_complete.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# STEP 1: GENERATE TRAINING DATA


print("\n" + "="*60)
print("STEP 1: GENERATING TRAINING DATA")
print("="*60)

essential_transactions = [
    "WOOLWORTHS SUPERMARKET", "COLES SUPERMARKET", "ALDI STORES",
    "ELECTRICITY BILL PAYMENT", "WATER CORPORATION", "GAS BILL PAYMENT",
    "INTERNET SERVICE PROVIDER", "PHONE BILL PAYMENT",
    "PHARMACY PRESCRIPTION", "MEDICAL CENTRE", "DENTAL CLINIC",
    "FUEL STATION", "PETROL STATION", "CAR INSURANCE",
    "PUBLIC TRANSPORT", "TRAIN TICKET", "BUS FARE",
    "RENT PAYMENT", "MORTGAGE PAYMENT", "HOME INSURANCE",
    "SCHOOL FEES", "UNIVERSITY TUITION", "CHILDCARE CENTRE",
    "GYM MEMBERSHIP", "FITNESS CLASS",
]

non_essential_transactions = [
    "NETFLIX SUBSCRIPTION", "SPOTIFY PREMIUM", "DISNEY PLUS",
    "UBER EATS DELIVERY", "MENULOG ORDER", "DELIVEROO",
    "RESTAURANT DINING", "CAFE PURCHASE", "STARBUCKS",
    "AMAZON PURCHASE", "EBAY PURCHASE", "FASHION RETAILER",
    "STEAM GAMES PURCHASE", "PLAYSTATION STORE", "XBOX STORE",
    "BEAUTY SALON", "HAIR SALON", "SPA TREATMENT", "NAIL SALON",
    "CINEMA TICKETS", "LIQUOR STORE", "BAR TAB", "NIGHTCLUB",
]

data = []
n_samples = 2000

# Generate essential (label=0)
for _ in range(n_samples // 2):
    transaction = random.choice(essential_transactions)
    if random.random() < 0.3:
        transaction = transaction + " " + str(random.randint(1000, 9999))
    data.append({'transaction_description': transaction, 'label': 0})

# Generate non-essential (label=1)
for _ in range(n_samples // 2):
    transaction = random.choice(non_essential_transactions)
    if random.random() < 0.3:
        transaction = transaction + " " + str(random.randint(1000, 9999))
    data.append({'transaction_description': transaction, 'label': 1})

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('transactions_training.csv', index=False)

print(f"Generated {n_samples} training samples")
print(f"Essential: {n_samples//2}, Non-Essential: {n_samples//2}")
print("Saved to: transactions_training.csv\n")


# STEP 2: PREPARE DATASET


print("="*60)
print("STEP 2: PREPARING DATASET")
print("="*60)

class TransactionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load and split data
df['transaction_description'] = df['transaction_description'].str.lower().str.strip()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['transaction_description'].values,
    df['label'].values,
    test_size=0.2,
    random_state=42,
    stratify=df['label'].values
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}\n")


# STEP 3: LOAD MODEL


print("="*60)
print("STEP 3: LOADING DISTILBERT MODEL")
print("="*60)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)
model.to(device)

print("Model loaded successfully\n")

# Create datasets
train_dataset = TransactionDataset(train_texts, train_labels, tokenizer)
val_dataset = TransactionDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 4
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# STEP 4: TRAINING


print("="*60)
print("STEP 4: TRAINING MODEL")
print("="*60)

best_val_loss = float('inf')

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')
    print('-' * 60)
    
    # Training
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_train_loss = total_loss / len(train_loader)
    print(f'Training Loss: {avg_train_loss:.4f}')
    
    # Validation
    model.eval()
    predictions = []
    actual_labels = []
    total_val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    
    print('\nClassification Report:')
    print(classification_report(actual_labels, predictions, 
                               target_names=['Essential', 'Non-Essential']))
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f'Saving best model...')
        model.save_pretrained('./transaction_classifier')
        tokenizer.save_pretrained('./transaction_classifier')


# STEP 5: TEST PREDICTIONS


print("\n" + "="*60)
print("STEP 5: TESTING MODEL")
print("="*60 + "\n")

model.eval()

test_transactions = [
    "WOOLWORTHS SUPERMARKET",
    "NETFLIX SUBSCRIPTION",
    "ELECTRICITY BILL PAYMENT",
    "UBER EATS DELIVERY",
    "PHARMACY PRESCRIPTION",
    "STEAM GAMES PURCHASE"
]

for transaction in test_transactions:
    encoding = tokenizer.encode_plus(
        transaction.lower(),
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    
    label = 'Non-Essential' if prediction == 1 else 'Essential'
    confidence = probabilities[prediction] * 100
    
    print(f"Transaction: {transaction}")
    print(f"  → {label} (confidence: {confidence:.1f}%)\n")

print("="*60)
print("✨ TRAINING COMPLETE!")
print("="*60)
print("\nModel saved to: ./transaction_classifier")
print("\nNext step: Run your Streamlit app")
print("streamlit run streamlit_app.py")