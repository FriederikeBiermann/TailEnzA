import torch.nn as nn
import os
import torch
import random
import sys
import json
import linecache
import argparse
import pandas as pd
from datetime import datetime
import numpy as np
import torch.optim as optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import  DataLoader, IterableDataset
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.nn.init as init
import random
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from plotting_results import *
import torch.autograd.profiler as profiler

import torch
import torch.nn as nn
import torch.nn.init as init

class LSTMClassifier(nn.Module):
    def __init__(self, seq_input_dim, normal_input_dim, fragment_lengths, num_classes, hidden_dim=64, num_layers=2, dropout_prob=0.1):
        super(LSTMClassifier, self).__init__()

        self.fragment_lengths = fragment_lengths

        # For each sequence fragment
        self.lstms = nn.ModuleList([nn.LSTM(seq_input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0) for _ in fragment_lengths])

        # Intermediate fully connected layer
        # The input will be the concatenated outputs of all LSTMs and normal features
        self.fc1 = nn.Linear(hidden_dim*2*len(fragment_lengths) + normal_input_dim, hidden_dim)

        # Dropout layer
        self.dropout_fc1 = nn.Dropout(p=dropout_prob)

        # Final fully connected layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Apply the initialization
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

    def forward(self, seq_data, normal_data):
        lstm_outs = []

        for i, lstm in enumerate(self.lstms):
            # Assuming seq_data contains all fragments in order
            fragment = seq_data[..., sum(self.fragment_lengths[:i]):sum(self.fragment_lengths[:i+1])]
            lstm_out, _ = lstm(fragment)
            # Taking the final hidden state from the BiLSTM
            lstm_outs.append(lstm_out[:, -1, :])

        combined_lstm_out = torch.cat(lstm_outs, dim=-1)
        combined_data = torch.cat([combined_lstm_out, normal_data], dim=-1)

        x = F.relu(self.fc1(combined_data))
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        
        return x

def train_bilstm_classifier(train_dataloader, fragment_lengths, num_classes, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # normal_input_dim is the length of charge features
    normal_input_dim = len(fragment_lengths)*3+3
    model = LSTMClassifier(7, normal_input_dim, fragment_lengths, num_classes, hidden_dim=64, num_layers=2, dropout_prob=0.1).to(device)
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    batch_losses = []
    epoch_losses = []
    writer = SummaryWriter()
    clip_value = 1.0
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (seq_data, normal_data, tags, _, _) in enumerate(train_dataloader):
            seq_data, normal_data, tags = seq_data.to(device), normal_data.to(device), tags.to(device)

            optimizer.zero_grad()
            outputs = model(seq_data, normal_data)
            
            # Using Cross Entropy Loss
            loss = criterion(outputs, tags)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            total_loss += loss.item()
            batch_losses.append(loss.item())
            
        avg_loss = total_loss / (i + 1)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss}")
        epoch_losses.append(avg_loss)
        scheduler.step()

    writer.close()
    return model, epoch_losses, batch_losses



def get_model_predictions_and_labels(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true = []
    y_pred = []
    rows = []
    
    torch.cuda.empty_cache()

    with torch.no_grad():
        for record_id, encoded_fragment, charge_tensors, bgc_type in dataloader:
            encoded_fragment, charge_tensors, bgc_type = encoded_fragment.to(device), charge_tensors.to(device), bgc_type.to(device)

            with profiler.profile(use_cuda=True) as prof:
                with profiler.record_function("model_inference"):
                    outputs = model(encoded_fragment, charge_tensors)

            labels = bgc_type.argmax(dim=1)
            predicted_labels = outputs.argmax(dim=1)
            
            y_true.extend(labels.tolist())
            y_pred.extend(predicted_labels.tolist())
            
            valid_positions = charge_tensors.tolist()
            valid_predicted_scores = outputs.tolist()
            
            for pos, score in zip(valid_positions, valid_predicted_scores):
                rows.append((record_id, pos, score))
                
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows, columns=['filename', 'position', 'predicted_score'])
    print(type(y_pred), type(y_true))
    
    return y_true, y_pred, outputs, df

