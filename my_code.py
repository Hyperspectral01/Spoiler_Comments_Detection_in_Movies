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


    # to_return_comments=np.array(to_return_comments)
    # to_return_labels=np.array(to_return_labels)


    return to_return_srt,to_return_comments,to_return_labels #string, (B,), (B,), these are lists





def train_one_epoch(models, dataloaders, optimizers, loss_function):
    """
    Train models for one epoch with proper logging
    """
    # Set models to training mode
    for model in models:
        model.train()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    batch_count = 0
    
    # Clear gradients before starting
    for optim in optimizers:
        optim.zero_grad()
    
    for dataloader_idx, dataloader in enumerate(dataloaders):
        print(f"Training on dataloader {dataloader_idx + 1}/{len(dataloaders)}")
        
        for batch_idx, (srt_string, comments, labels) in enumerate(dataloader):
            try:
                # Forward pass
                srt_tensors = models[0](srt_string)  # 20 x 1024
                #print("Model - 1 dims:",srt_tensors.shape)
                comments_tensor = models[1](comments)  # B x 768
                #print("Model - 2 dims:",comments_tensor.shape)
                context_tensors = models[2](srt_tensors, comments_tensor)  # B x 2 x 1024
                #print("Model-3 dims:",context_tensors.shape)
                pred = models[3](comments_tensor, context_tensors)  # B x 2
                #print("Model - 4 dims:",pred.shape)
                
                # Convert labels to tensor
                labels_tensor = torch.tensor(labels, dtype=torch.long)
                
                # Calculate loss
                loss = loss_function(pred, labels_tensor)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                for optim in optimizers:
                    optim.step()
                    optim.zero_grad()
                
                # Collect statistics
                total_loss += loss.item()
                total_samples += len(labels)
                batch_count += 1
                
                # Get predictions
                predicted_classes = torch.argmax(pred, dim=1)
                all_predictions.extend(predicted_classes.detach().cpu().numpy())
                all_labels.extend(labels)
                
                # Print batch progress
                if batch_idx % 10 == 0:
                    current_accuracy = accuracy_score(all_labels, all_predictions) * 100
                    print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}, Accuracy = {current_accuracy:.2f}%")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx} of dataloader {dataloader_idx}: {str(e)}")
                continue
    
    # Calculate final metrics
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
    precision = precision_score(all_labels, all_predictions, average='weighted') * 100
    recall = recall_score(all_labels, all_predictions, average='weighted') * 100
    
    print(f"\n=== TRAINING EPOCH SUMMARY ===")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"Total Samples: {total_samples}")
    print("=" * 30)
    
    return models, optimizers


def eval_one_epoch(models, dataloaders, optimizers, loss_function):
    """
    Evaluate models for one epoch with proper logging
    """
    # Set models to evaluation mode
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
                    
                    # Convert labels to tensor
                    labels_tensor = torch.tensor(labels, dtype=torch.long)
                    
                    # Calculate loss
                    loss = loss_function(pred, labels_tensor)
                    
                    # Collect statistics
                    total_loss += loss.item()
                    total_samples += len(labels)
                    batch_count += 1
                    
                    # Get predictions
                    predicted_classes = torch.argmax(pred, dim=1)
                    all_predictions.extend(predicted_classes.cpu().numpy())
                    all_labels.extend(labels)
                    
                    # Print batch progress
                    if batch_idx % 10 == 0:
                        current_accuracy = accuracy_score(all_labels, all_predictions) * 100
                        print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}, Accuracy = {current_accuracy:.2f}%")
                        
                except Exception as e:
                    print(f"Error in batch {batch_idx} of dataloader {dataloader_idx}: {str(e)}")
                    continue
    
    # Calculate final metrics
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
    precision = precision_score(all_labels, all_predictions, average='weighted') * 100
    recall = recall_score(all_labels, all_predictions, average='weighted') * 100
    
    print(f"\n=== VALIDATION EPOCH SUMMARY ===")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"Total Samples: {total_samples}")
    print("=" * 35)
    
    return models, optimizers
        

        
    


      



###############################  CLASSES  #########################################
class Model1(nn.Module):
  def __init__(self ,input_size_to_GRU=768,size_of_GRU=1024,number_of_hidden_states=20):
    super().__init__()
    self.number_of_chunks=number_of_hidden_states
    self.gru=nn.GRU(input_size_to_GRU,size_of_GRU,batch_first=True)
    model_path = "./my_model_from_huggingface"

    # Load the tokenizer and model from the local dataset folder
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    
    # Create a pipeline for feature extraction (or another task if desired)
    pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    self.pipe = pipe

  def forward(self,srt_string):
    words = srt_string.split()
    chunk_size = len(words) // self.number_of_chunks + (len(words) % self.number_of_chunks > 0)
    list_of_chunks=[' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # print(len(list_of_chunks))
    # print(list_of_chunks)

    hidden_states=[]

    for chunk in list_of_chunks:
      list_of_tensors=[]
      for i in range(0,len(chunk.split(" ")),100):
        list_of_100_words=[]
        list_of_100_words.extend(chunk.split(" ")[i:i+100])
        str_for_embedding=""
        for word in list_of_100_words:
          str_for_embedding+=word+" "

        out=self.pipe(str_for_embedding)  #out is a list

        list_of_tensors.append(torch.mean(torch.tensor(out),dim=1).unsqueeze(1))

      for i in range(1,len(list_of_tensors)):
        list_of_tensors[0]=torch.cat((list_of_tensors[0],list_of_tensors[i]),dim=1)

      # print(list_of_tensors[0].shape)
      # print(list_of_tensors[0])

      outputs, hidden=self.gru(list_of_tensors[0])
      hidden_states.append(hidden.squeeze(1))

    return torch.cat(hidden_states,dim=0)
  



class Model2(nn.Module):
    def __init__(self, input_size_to_GRU=768, size_of_GRU=768, max_seq_len=512):
        super().__init__()
        self.gru = nn.GRU(input_size_to_GRU, size_of_GRU, batch_first=True)
        self.max_seq_len = max_seq_len
        
        model_path = "./my_model_from_huggingface"
        # Load the tokenizer and model from the local dataset folder
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        
        # Create a pipeline for feature extraction
        pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
        self.pipe = pipe

    def forward(self, comment_strings):
        batched_comments_list = []
        max_length = 0
        
        # First pass: get embeddings and find max length
        for comment_string in comment_strings:
            out = self.pipe(comment_string)  # out is a list of embeddings
            # Convert to tensor and squeeze if needed
            embedding = torch.tensor(out).squeeze(0)  # Remove batch dim if present
            
            # Handle case where embedding might be 1D (single token)
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
                # Truncate if longer than max_length
                embedding = embedding[:max_length]
            elif seq_len < max_length:
                # Pad if shorter than max_length
                padding_size = max_length - seq_len
                padding = torch.zeros(padding_size, embedding.shape[1])
                embedding = torch.cat([embedding, padding], dim=0)
            
            padded_embeddings.append(embedding)
        
        # Stack all padded embeddings into a batch
        batched_comments_tensor = torch.stack(padded_embeddings, dim=0)  # (B, max_length, 768)
        
        # Pass through GRU
        outputs, hidden = self.gru(batched_comments_tensor)
        
        return hidden.squeeze(0)  # (B, 768)
  




# Updated Model3 - Batched Comments with Dot Product Attention
class Model3(nn.Module):
    def __init__(self, srt_hidden_size=1024, comment_hidden_size=768, k_hidden_states=2, dropout=0.1):
        super().__init__()
        self.k = k_hidden_states
        self.srt_hidden_size = srt_hidden_size
        self.comment_hidden_size = comment_hidden_size
        
        # Transform comment embeddings to match SRT dimension
        self.comment_transform = nn.Sequential(
            nn.Linear(comment_hidden_size, srt_hidden_size),  # B×768 -> B×1024
            nn.BatchNorm1d(srt_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Transform SRT embeddings 
        self.srt_transform = nn.Sequential(
            nn.Linear(srt_hidden_size, srt_hidden_size),  # 20×1024 -> 20×1024
            nn.BatchNorm1d(srt_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, hidden_states_srt, hidden_states_comments):
        # hidden_states_srt: (20, 1024)
        # hidden_states_comments: (B, 768)
        
        batch_size = hidden_states_comments.size(0)
        
        # Transform comments to match SRT dimension
        transformed_comments = self.comment_transform(hidden_states_comments)  # (B, 1024)
        
        # Transform SRT states
        transformed_srt = self.srt_transform(hidden_states_srt)  # (20, 1024)
        
        # Compute attention scores using dot product
        # transformed_comments: (B, 1024), transformed_srt: (20, 1024)
        attention_scores = torch.matmul(transformed_comments, transformed_srt.T)  # (B, 20)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, 20)
        
        # Apply attention to get weighted SRT representations for each comment
        # attention_weights: (B, 20), transformed_srt: (20, 1024)
        context_vectors = torch.matmul(attention_weights, transformed_srt)  # (B, 1024)
        
        # Select top-k SRT states for each comment based on attention
        top_vals, top_indices = torch.topk(attention_weights, k=self.k, dim=-1)  # (B, k)
        
        # Gather selected states for each batch item
        selected_states = []
        for i in range(batch_size):
            selected = transformed_srt[top_indices[i]]  # (k, 1024)
            selected_states.append(selected)
        
        selected_states = torch.stack(selected_states, dim=0)  # (B, k, 1024)
        
        return selected_states


# Updated Model4 - Simple Deep Feed Forward Network
class Model4(nn.Module):
    def __init__(self, k_hidden_states=2, srt_dim=1024, comment_dim=768, output_neurons=2, dropout=0.2):
        super().__init__()
        
        # Calculate input dimension: flattened SRT + comment embeddings
        srt_flattened_dim = srt_dim * k_hidden_states  # 2*1024 = 2048
        input_dim = srt_flattened_dim + comment_dim  # 2048 + 768 = 2816
        
        # Simple deep feed forward network
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048),  # B×2816 -> B×2048
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(2048, 1536),  # B×2048 -> B×1536
            nn.BatchNorm1d(1536),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(1536, 1024),  # B×1536 -> B×1024
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),  # B×1024 -> B×512
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(512, 256),  # B×512 -> B×256
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(256, 128),  # B×256 -> B×128
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),
            
            nn.Linear(128, output_neurons)  # B×128 -> B×2
        )
        
        # Initialize weights
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
        # tensor_comments: (B, 768)
        # tensor_srt: (B, k, 1024) where k=2
        
        batch_size = tensor_comments.size(0)
        
        # Flatten SRT states for each batch item
        flattened_srt = tensor_srt.view(batch_size, -1)  # (B, k*1024) = (B, 2048)
        
        # Concatenate comments and SRT directly
        combined_input = torch.cat([tensor_comments, flattened_srt], dim=-1)  # (B, 768+2048) = (B, 2816)
        
        # Pass through deep network
        output = self.network(combined_input)  # (B, 2)
        
        return output
    

  

class my_dataset(Dataset):
    def __init__(self,srt_file_path):
        self.comments=[]      #at the same index, so movie at index_1, will have [comment_1,comment_2...comment_n] at index 1
        self.labels=[]
     
        self.srt_string=extract_dialogues_from_srt(srt_file_path)
        
        with open(srt_file_path[:-4]+".csv", 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Expecting each row to have two columns: type and comment
                if len(row) < 2:
                    continue
                comment_type = row[0].strip()
                comment_text = row[1].strip()
    
                # Map the type to a label: Spoiler -> 1, Non-Spoiler -> 0
                if comment_type.lower() == "spoiler":
                    self.labels.append(1)
                elif comment_type.lower() == "non-spoiler":
                    self.labels.append(0)
                else:
                    continue  # Skip rows that do not match the expected types
    
                self.comments.append(comment_text)


    def __len__(self):
        return len(self.comments)


    def __getitem__(self,idx):
        return self.srt_string,self.comments[idx],self.labels[idx]

###############################  MAIN ###################################

n_epochs=10
batch_size=16
lr=0.001
loss_function=nn.CrossEntropyLoss()

models=[Model1(768,1024,20),Model2(),Model3(),Model4()]
train_datasets=[]
valid_datasets=[]

for file_name in os.listdir(train_dir_path):
    if (file_name.endswith(".srt")):
        srt_file_path=os.path.join(train_dir_path,file_name)
        train_datasets.append(my_dataset(srt_file_path))

for file_name in os.listdir(valid_dir_path):
    if (file_name.endswith(".srt")):
        srt_file_path=os.path.join(valid_dir_path,file_name)
        valid_datasets.append(my_dataset(srt_file_path))



train_dataloaders=[DataLoader(dataset,batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn) for dataset in train_datasets]
test_dataloaders=[DataLoader(dataset,batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn) for dataset in valid_datasets]
optimisers=[optim.Adam(model.parameters(), lr=lr) for model in models]


for i in range(n_epochs):
    models, optimisers=train_one_epoch(models,train_dataloaders,optimisers,loss_function)
    models, optimisers=eval_one_epoch(models,test_dataloaders,optimisers,loss_function)

    for idx in range(4):
        model_path = f"./out/mock_model{idx+1}_{i+1}_epoch.pth"
        weights_path = f"./out/mock_model{idx+1}_weights_{i+1}_epoch.pth"
    
        torch.save(models[idx], model_path)
        torch.save(models[idx].state_dict(), weights_path)
