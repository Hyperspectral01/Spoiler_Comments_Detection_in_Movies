# -*- coding: utf-8 -*-
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



############################# INPUT #####################
model1_weights_path="/home/bce22157/Shrey_Playground/Spoilers_in_movies/out/best_model1_weights.pth"
model2_weights_path="/home/bce22157/Shrey_Playground/Spoilers_in_movies/out/best_model2_weights.pth"
model3_weights_path="/home/bce22157/Shrey_Playground/Spoilers_in_movies/out/best_model3_weights.pth"
model4_weights_path="/home/bce22157/Shrey_Playground/Spoilers_in_movies/out/best_model4_weights.pth"


srt_file_path="/home/bce22157/Shrey_Playground/Spoilers_in_movies/subtitles_for_test/Avengers_.Endgame.2019.720p.BluRay.x264.[YTS.MX]-English-en.srt"

sample_comments=["Thanos dies","Thanos got all the infinity stones.","Damn, in the end, Tony sacrifices.","Salute to Tony Stark","Is that new hammer for Thor???","Is that new weapon for Thor???","Damn he got the big one.","Is that racoon or rocket??!!","So this was the one and only chance","Thanos knows about Doctor Strange."]


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

def extract_dialogues_from_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Remove timestamps and serial numbers
    dialogues = re.sub(r'\d+\n|\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)

    # Remove empty lines and join dialogues into a single string
    dialogue_string = ' '.join(dialogues.split('\n')).strip()

    return dialogue_string

srt_string=extract_dialogues_from_srt(srt_file_path)




models=[]
models=[Model1(768,1024,20),Model2(),Model3(),Model4()]

models[0].load_state_dict(torch.load(model1_weights_path,weights_only=True))

models[1].load_state_dict(torch.load(model2_weights_path,weights_only=True))

models[2].load_state_dict(torch.load(model3_weights_path,weights_only=True))

models[3].load_state_dict(torch.load(model4_weights_path,weights_only=True))


for model in models:
    model.eval()


srt_tensors = models[0](srt_string)  # 20 x 1024
#print("Model - 1 dims:",srt_tensors.shape)
comments_tensor = models[1](sample_comments)  # B x 768
#print("Model - 2 dims:",comments_tensor.shape)
context_tensors = models[2](srt_tensors, comments_tensor)  # B x 2 x 1024
#print("Model-3 dims:",context_tensors.shape)
pred = models[3](comments_tensor, context_tensors)  # B x 2

print(pred)

for i,comment in enumerate(sample_comments):
  print(comment," : ",pred[i])






