import streamlit as st
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os

# Dummy model (Replace with your actual model)
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.fc = nn.Linear(128, 1)  # Replace this with actual Transformer layers

    def forward(self, x):
        return self.fc(x)  # Replace with forward pass through your Transformer

# Custom Dataset to load data from the directory
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.txt')]
        self.data = self.load_data()

    def load_data(self):
        all_data = []
        for file in self.file_list:
            with open(file, 'r', encoding='utf-8') as f:
                all_data.extend(f.readlines())  # Read your data (tokenized already)
        return all_data  # Replace with actual tokenized data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Simulating tokenized data and target (replace with actual tokenized data)
        input_data = torch.tensor([ord(ch) for ch in self.data[idx][:128]], dtype=torch.float32)  # Simulated input
        target_data = torch.tensor([ord(ch) for ch in self.data[idx][128:]], dtype=torch.float32)  # Simulated target
        return input_data, target_data

# Function to train and return metrics
def train_epoch(model, data_loader, optimizer, criterion, epoch, stats):
    model.train()
    total_loss = 0
    for batch_idx, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Update stats
        stats["loss"].append(total_loss / (batch_idx + 1))
        
        # Real-time plot update
        plot_metrics(stats)

def plot_metrics(stats):
    plt.clf()
    plt.plot(stats['loss'], label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss in Real-time')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

# Main UI code
def main():
    st.title("Real-Time Training Dashboard")

    # Load actual data from the Data directory
    data_dir = st.text_input("Enter the directory path to your data", "Data")
    
    # Load dataset and create a DataLoader
    dataset = CustomDataset(data_dir)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize model, optimizer, and loss function
    model = TransformerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Adjust if necessary
    
    # Stats dictionary to store training stats
    stats = {
        "loss": []
    }
    
    # Training Loop
    epochs = st.slider("Select number of epochs", 1, 10, 3)
    
    for epoch in range(epochs):
        st.write(f"Epoch {epoch+1}/{epochs}")
        train_epoch(model, data_loader, optimizer, criterion, epoch, stats)
        st.write(f"Epoch {epoch+1} completed")

if __name__ == '__main__':
    main()
