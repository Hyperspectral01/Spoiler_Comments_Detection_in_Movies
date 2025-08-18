###############################  NOTES ###################################
"""
The input dir should be having all the subtitle files with the same name as the comments file, but only with a different extension,
 this time the extension should be with .csv instead of the .srt file extension
"""

#####################################   INPUTS    ################################
train_dir_path="./train"
valid_dir_path="./valid"

#####################################   DEPENDENCIES    ################################

import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
from transformers import pipeline, AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

###############################  FUNCTIONS ###################################

def extract_dialogues_from_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Remove timestamps and serial numbers
    dialogues = re.sub(r'\d+\n|\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)

    # Remove empty lines and join dialogues into a single string
    dialogue_string = ' '.join(dialogues.split('\n')).strip()

    return dialogue_string

def my_collate_fn(batch):
    to_return_comments=[]
    to_return_labels=[]
    to_return_srt=""

    for i,(srt,comment,label) in enumerate(batch):
        if (i==0):
            to_return_srt=srt
        
        to_return_comments.append(comment)
        to_return_labels.append(label)

    return to_return_srt,to_return_comments,to_return_labels

def train_one_epoch(models, dataloaders, optimizers, loss_function):
    """
    Train models for one epoch with proper logging and regularization
    """
    # Set models to training mode
    for model in models:
        model.train()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    batch_count = 0
    
    for dataloader_idx, dataloader in enumerate(dataloaders):
        print(f"Training on dataloader {dataloader_idx + 1}/{len(dataloaders)}")
        
        for batch_idx, (srt_string, comments, labels) in enumerate(dataloader):
            try:
                # Clear gradients
                for optim in optimizers:
                    optim.zero_grad()
                
                # Forward pass
                srt_tensors = models[0](srt_string)  # 20 x 1024
                comments_tensor = models[1](comments)  # B x 768
                context_tensors = models[2](srt_tensors, comments_tensor)  # B x 2 x 1024
                pred = models[3](comments_tensor, context_tensors)  # B x 2
                
                # Convert labels to tensor and move to device
                labels_tensor = torch.tensor(labels, dtype=torch.long)
                pred = pred
                
                # Calculate loss
                loss = loss_function(pred, labels_tensor)
                
                # Add L2 regularization
                l2_reg = torch.tensor(0.)
                for model in models:
                    for param in model.parameters():
                        l2_reg += torch.norm(param, 2)
                loss += 0.001 * l2_reg  # L2 regularization with weight 0.001
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                for model in models:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                for optim in optimizers:
                    optim.step()
                
                # Collect statistics
                total_loss += loss.item()
                total_samples += len(labels)
                batch_count += 1
                
                # Get predictions
                predicted_classes = torch.argmax(pred, dim=1)
                all_predictions.extend(predicted_classes.detach().numpy())
                all_labels.extend(labels)
                
                # Print batch progress
                if batch_idx % 5 == 0:
                    current_accuracy = accuracy_score(all_labels, all_predictions) * 100 if all_labels else 0
                    print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}, Accuracy = {current_accuracy:.2f}%")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx} of dataloader {dataloader_idx}: {str(e)}")
                continue
    
    # Calculate final metrics
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    accuracy = accuracy_score(all_labels, all_predictions) * 100 if all_labels else 0
    
    print(f"\n=== TRAINING EPOCH SUMMARY ===")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total Samples: {total_samples}")
    print("=" * 30)
    
    return models, optimizers, avg_loss, accuracy

def eval_one_epoch(models, dataloaders, loss_function):
    """
    Evaluate models for one epoch with proper logging
    """
    # Set models to evaluation mode - THIS IS CRUCIAL
    for model in models:
        model.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    batch_count = 0
    
    # No gradient computation during evaluation
    with torch.no_grad():
        for dataloader_idx, dataloader in enumerate(dataloaders):
            print(f"Evaluating on dataloader {dataloader_idx + 1}/{len(dataloaders)}")
            
            for batch_idx, (srt_string, comments, labels) in enumerate(dataloader):
                try:
                    # Forward pass
                    srt_tensors = models[0](srt_string)  # 20 x 1024
                    comments_tensor = models[1](comments)  # B x 768
                    context_tensors = models[2](srt_tensors, comments_tensor)  # B x 2 x 1024
                    pred = models[3](comments_tensor, context_tensors)  # B x 2
                    
                    # Convert labels to tensor and move to device
                    labels_tensor = torch.tensor(labels, dtype=torch.long)
                    pred = pred
                    
                    # Calculate loss (without regularization for validation)
                    loss = loss_function(pred, labels_tensor)
                    
                    # Collect statistics
                    total_loss += loss.item()
                    total_samples += len(labels)
                    batch_count += 1
                    
                    # Get predictions
                    predicted_classes = torch.argmax(pred, dim=1)
                    all_predictions.extend(predicted_classes.numpy())
                    all_labels.extend(labels)
                    
                    # Print batch progress
                    if batch_idx % 5 == 0:
                        current_accuracy = accuracy_score(all_labels, all_predictions) * 100 if all_labels else 0
                        print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}, Accuracy = {current_accuracy:.2f}%")
                        
                except Exception as e:
                    print(f"Error in batch {batch_idx} of dataloader {dataloader_idx}: {str(e)}")
                    continue
    
    # Calculate final metrics
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    accuracy = accuracy_score(all_labels, all_predictions) * 100 if all_labels else 0
    
    print(f"\n=== VALIDATION EPOCH SUMMARY ===")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total Samples: {total_samples}")
    print("=" * 35)
    
    return avg_loss, accuracy

###############################  CLASSES  #########################################
class Model1(nn.Module):
    def __init__(self, input_size_to_GRU=768, size_of_GRU=1024, number_of_hidden_states=20):
        super().__init__()
        self.number_of_chunks = number_of_hidden_states
        self.gru = nn.GRU(input_size_to_GRU, size_of_GRU, batch_first=True, dropout=0.3)
        model_path = "./my_model_from_huggingface"

        # Load the tokenizer and model from the local dataset folder
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        
        # Create a pipeline for feature extraction
        pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
        self.pipe = pipe

    def forward(self, srt_string):
        words = srt_string.split()
        chunk_size = len(words) // self.number_of_chunks + (len(words) % self.number_of_chunks > 0)
        list_of_chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        hidden_states = []

        for chunk in list_of_chunks:
            list_of_tensors = []
            for i in range(0, len(chunk.split(" ")), 100):
                list_of_100_words = []
                list_of_100_words.extend(chunk.split(" ")[i:i+100])
                str_for_embedding = ""
                for word in list_of_100_words:
                    str_for_embedding += word + " "

                out = self.pipe(str_for_embedding)
                list_of_tensors.append(torch.mean(torch.tensor(out), dim=1).unsqueeze(1))

            for i in range(1, len(list_of_tensors)):
                list_of_tensors[0] = torch.cat((list_of_tensors[0], list_of_tensors[i]), dim=1)

            outputs, hidden = self.gru(list_of_tensors[0])
            hidden_states.append(hidden.squeeze(1))

        return torch.cat(hidden_states, dim=0)

class Model2(nn.Module):
    def __init__(self, input_size_to_GRU=768, size_of_GRU=768, max_seq_len=256):  # Reduced max_seq_len
        super().__init__()
        self.gru = nn.GRU(input_size_to_GRU, size_of_GRU, batch_first=True, dropout=0.3)
        self.max_seq_len = max_seq_len
        
        model_path = "./my_model_from_huggingface"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        
        pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
        self.pipe = pipe

    def forward(self, comment_strings):
        batched_comments_list = []
        max_length = 0
        
        # First pass: get embeddings and find max length
        for comment_string in comment_strings:
            # Truncate very long comments
            if len(comment_string.split()) > 100:
                comment_string = ' '.join(comment_string.split()[:100])
                
            out = self.pipe(comment_string)
            embedding = torch.tensor(out).squeeze(0)
            
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            
            batched_comments_list.append(embedding)
            max_length = max(max_length, embedding.shape[0])
        
        # Apply max sequence length limit
        max_length = min(max_length, self.max_seq_len)
        
        # Second pass: pad all embeddings to max_length
        padded_embeddings = []
        for embedding in batched_comments_list:
            seq_len = embedding.shape[0]
            
            if seq_len > max_length:
                embedding = embedding[:max_length]
            elif seq_len < max_length:
                padding_size = max_length - seq_len
                padding = torch.zeros(padding_size, embedding.shape[1])
                embedding = torch.cat([embedding, padding], dim=0)
            
            padded_embeddings.append(embedding)
        
        batched_comments_tensor = torch.stack(padded_embeddings, dim=0)
        outputs, hidden = self.gru(batched_comments_tensor)
        
        return hidden.squeeze(0)

class Model3(nn.Module):
    def __init__(self, srt_hidden_size=1024, comment_hidden_size=768, k_hidden_states=2, dropout=0.3):
        super().__init__()
        self.k = k_hidden_states
        self.srt_hidden_size = srt_hidden_size
        self.comment_hidden_size = comment_hidden_size
        
        # Simpler transformation layers
        self.comment_transform = nn.Sequential(
            nn.Linear(comment_hidden_size, srt_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.srt_transform = nn.Sequential(
            nn.Linear(srt_hidden_size, srt_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, hidden_states_srt, hidden_states_comments):
        batch_size = hidden_states_comments.size(0)
        
        transformed_comments = self.comment_transform(hidden_states_comments)
        transformed_srt = self.srt_transform(hidden_states_srt)
        
        # Compute attention scores using dot product
        attention_scores = torch.matmul(transformed_comments, transformed_srt.T)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Select top-k SRT states
        top_vals, top_indices = torch.topk(attention_weights, k=self.k, dim=-1)
        
        selected_states = []
        for i in range(batch_size):
            selected = transformed_srt[top_indices[i]]
            selected_states.append(selected)
        
        selected_states = torch.stack(selected_states, dim=0)
        
        return selected_states

class Model4(nn.Module):
    def __init__(self, k_hidden_states=2, srt_dim=1024, comment_dim=768, output_neurons=2, dropout=0.4):
        super().__init__()
        
        srt_flattened_dim = srt_dim * k_hidden_states
        input_dim = srt_flattened_dim + comment_dim
        
        # Simplified network to prevent overfitting
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            
            nn.Linear(256, output_neurons)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, tensor_comments, tensor_srt):
        batch_size = tensor_comments.size(0)
        
        flattened_srt = tensor_srt.view(batch_size, -1)
        combined_input = torch.cat([tensor_comments, flattened_srt], dim=-1)
        output = self.network(combined_input)
        
        return output

class my_dataset(Dataset):
    def __init__(self, srt_file_path):
        self.comments = []
        self.labels = []
     
        self.srt_string = extract_dialogues_from_srt(srt_file_path)
        
        with open(srt_file_path[:-4]+".csv", 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) < 2:
                    continue
                comment_type = row[0].strip()
                comment_text = row[1].strip()
    
                if comment_type.lower() == "spoiler":
                    self.labels.append(1)
                elif comment_type.lower() == "non-spoiler":
                    self.labels.append(0)
                else:
                    continue
    
                self.comments.append(comment_text)

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        return self.srt_string, self.comments[idx], self.labels[idx]

###############################  MAIN ###################################



# Reduced hyperparameters to prevent overfitting
n_epochs = 20
batch_size = 8  # Reduced batch size
lr = 0.0005  # Reduced learning rate
loss_function = nn.CrossEntropyLoss()

# Initialize models and move to device
models = [Model1(768, 1024, 20), Model2(), Model3(), Model4()]

train_datasets = []
valid_datasets = []

# Load datasets
for file_name in os.listdir(train_dir_path):
    if file_name.endswith(".srt"):
        srt_file_path = os.path.join(train_dir_path, file_name)
        train_datasets.append(my_dataset(srt_file_path))

for file_name in os.listdir(valid_dir_path):
    if file_name.endswith(".srt"):
        srt_file_path = os.path.join(valid_dir_path, file_name)
        valid_datasets.append(my_dataset(srt_file_path))

# Create dataloaders
train_dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn) for dataset in train_datasets]
test_dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn) for dataset in valid_datasets]  # No shuffle for validation

# Initialize optimizers with weight decay for regularization
optimizers = [optim.Adam(model.parameters(), lr=lr, weight_decay=0.01) for model in models]

# Learning rate scheduler
schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True) for optimizer in optimizers]

# Create output directory
os.makedirs("./out", exist_ok=True)

# Training loop with early stopping
best_val_accuracy = 0
patience = 5
wait = 0
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(n_epochs):
    print(f"\n{'='*50}")
    print(f"EPOCH {epoch + 1}/{n_epochs}")
    print(f"{'='*50}")
    
    # Training
    models, optimizers, train_loss, train_acc = train_one_epoch(models, train_dataloaders, optimizers, loss_function)
    
    # Validation
    val_loss, val_acc = eval_one_epoch(models, test_dataloaders, loss_function)
    
    # Update learning rate schedulers
    for scheduler in schedulers:
        scheduler.step(val_loss)
    
    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    # Early stopping
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        wait = 0
        # Save best models
        for idx in range(4):
            torch.save(models[idx].state_dict(), f"./out/best_model{idx+1}_weights.pth")
            torch.save(models[idx], f"./out/best_model{idx+1}.pth")
        print(f"New best validation accuracy: {best_val_accuracy:.2f}% - Models saved!")
    else:
        wait += 1
        print(f"No improvement in validation accuracy. Patience: {wait}/{patience}")
        
        if wait >= patience:
            print("Early stopping triggered!")
            break
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        for idx in range(4):
            torch.save(models[idx], f"./out/checkpoint_model{idx+1}_epoch{epoch+1}.pth")
            torch.save(models[idx].state_dict(), f"./out/checkpoint_model{idx+1}_weights_epoch{epoch+1}.pth")

print(f"\nTraining completed!")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('./out/training_curves.png')
plt.show()