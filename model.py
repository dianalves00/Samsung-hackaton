# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:08:27 2025

@author: user
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

df = pd.read_csv("C:\\Users\\user\\OneDrive\\Documentos\\tesla_news_prices_NEW.csv").drop('Unnamed: 0', axis=1)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Set the model to evaluation mode (not training)
model.eval()

def preprocess_text(text, max_length=512):
    # Tokenize the text and handle padding and truncation
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    return inputs

def get_embeddings(text, max_length=512):
    # Preprocess the text (tokenize, pad, and truncate)
    inputs = preprocess_text(text, max_length)

    # Perform a forward pass through the model with output_hidden_states=True
    with torch.no_grad():  # Disable gradient calculation to save memory
        outputs = model(**inputs, output_hidden_states=True)

    embeddings = outputs.hidden_states[-1]  # Get the last layer embeddings

    # Get the embedding for the [CLS] token (first token in the sequence)
    cls_embedding = embeddings[:, 0, :]  # CLS token is the first token (index 0)

    return cls_embedding

def split_text_into_chunks(text, max_length=512):
    # Tokenize the text and split into chunks of max_length tokens
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return chunks

def get_embeddings_for_large_text(text, max_length=512):
    # Split large text into chunks
    chunks = split_text_into_chunks(text, max_length)

    embeddings_list = []
    for chunk in chunks:
        # Convert tokens back to text for each chunk (to be compatible with model)
        chunk_text = tokenizer.convert_tokens_to_string(chunk)

        # Get embeddings for each chunk
        chunk_embedding = get_embeddings(chunk_text, max_length)
        embeddings_list.append(chunk_embedding)

    # Average the embeddings from each chunk (this represents the entire document)
    final_embedding = torch.mean(torch.stack(embeddings_list), dim=0)

    return final_embedding

rows = []
for i in df.news:
    rows.append(get_embeddings_for_large_text(i))
    
new_rows = []
for row in rows:
    new_rows.append(row[0].numpy())
    
new_rows = pd.DataFrame(new_rows)

merged = pd.concat([df, new_rows], axis=1).drop(['title', 'description', 'news_type', 'news'], axis=1)
merged = merged.groupby('Date').mean().sort_index().reset_index(drop=True)

from sklearn.preprocessing import StandardScaler

window_size = 7  # Look at the last 7 days of data to predict the next day

X = []
y = []

for i in range(window_size, len(merged) - 1):
    X.append(np.concatenate([
        merged.iloc[i - window_size:i, 10:],  # Embeddings for the last 7 days
        merged.iloc[i - window_size:i, :9],  # Prices for the last 7 days
    ], axis=1))  # Axis 1 is concatenating the feature dimension

    y.append(df.target[i])  # Predicting price at day t+1

X = np.array(X)  # Shape: (num_samples, window_size, input_dim)
y = np.array(y)  # Shape: (num_samples, 1)

# Normalize/Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

train_size = int(0.7 * len(X))  # 70% for training
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

import torch
import torch.nn as nn
import torch.optim as optim

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # Output layer with number of classes

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM output
        output = self.fc(lstm_out[:, -1, :])  # Use the output of the last time step
        return output

# Initialize the model
input_size = 777 # The input size is the number of features (e.g., embeddings, price, other features)
hidden_size = 64  # Number of hidden units in the LSTM layer
num_classes = 3  # Number of output classes (UP, DOWN, NEUTRAL)

model = LSTMClassifier(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Target labels should be LongTensor
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i in range(len(X_train_tensor)):
        inputs = X_train_tensor[i].unsqueeze(0)  # Add batch dimension
        targets = y_train_tensor[i].unsqueeze(0)  # Add batch dimension

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backpropagation
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")

from sklearn.metrics import classification_report

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    _, predicted = torch.max(val_outputs, 1)  # Get the class with the highest probability
    val_accuracy = (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
    print(classification_report(predicted, y_val_tensor))
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
import joblib
import torch

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Save the trained LSTM model
torch.save(model.state_dict(), "lstm_model.pth")

