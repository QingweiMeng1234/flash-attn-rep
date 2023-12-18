import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from glob import glob
import os
from transformers import BertConfig, BertForMaskedLM, AdamW
import time
import numpy as np
from tqdm import tqdm
import psutil
import matplotlib.pyplot as plt

# Function to get current memory usage
torch.manual_seed(0)
def get_gpu_memory_usage():
    # Returns the current GPU memory usage in MB
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    cached = torch.cuda.memory_reserved() / (1024 * 1024)
    return allocated, cached

# Define the text files dataset
# Define the text files dataset
class TextFolderDataset(Dataset):
    def __init__(self, file_directory, file_pattern, tokenizer, max_length):
        self.filepaths = glob(os.path.join(file_directory, file_pattern))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        with open(self.filepaths[idx], 'r', encoding='utf-8') as file:
            text = file.read()
        encoding = self.tokenizer.encode(text)
        input_ids = encoding.ids[:self.max_length] + [0] * (self.max_length - len(encoding.ids[:self.max_length]))
        attention_mask = [1] * len(encoding.ids[:self.max_length]) + [0] * (self.max_length - len(encoding.ids[:self.max_length]))
        
        # Create labels (for masked language modeling, you might mask some tokens here)
        labels = input_ids[:]  # In practice, apply masking strategy here

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)  # Add labels
        }

# Function to train a WordPiece tokenizer
def train_tokenizer(file_directory, file_pattern):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=30522,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    files = glob(os.path.join(file_directory, file_pattern))
    tokenizer.train(files, trainer)
    return tokenizer

def flat_accuracy(preds, labels):
    """
    Function to calculate the accuracy of our predictions vs labels.
    It flattens both the predictions and labels arrays to compare them element-wise.
    """
    # Convert the highest logit to predicted label (argmax over the last dimension)
    pred_flat = np.argmax(preds, axis=2).flatten()
    
    # Flatten the true labels array
    labels_flat = labels.flatten()

    # Calculate the number of correct predictions
    correct_predictions = np.sum(pred_flat == labels_flat)

    # Calculate accuracy as the ratio of correct predictions to total predictions
    accuracy = correct_predictions / len(labels_flat)
    
    return accuracy

# Set file paths and directory
file_directory = "openwebtext/"
file_pattern = "urlsf_subset01-32*"
eval_file_pattern = 'urlsf_subset01-33*'
max_lengths = [128, 256, 512, 768, 1024,1200,1300,1400,1900,2300,2800]

# Arrays to store runtime and memory usage
training_times = []
memory_usages = []

for max_length in tqdm(max_lengths):
    print("Processing max_length: ", max_length)

    # Train the tokenizer
    tokenizer = train_tokenizer(file_directory, file_pattern)
    tokenizer.save(f"model/tokenizer_bert_standard_{max_length}.json")

    # Load the tokenizer
    tokenizer = Tokenizer.from_file(f"model/tokenizer_bert_standard_{max_length}.json")

    # Create the dataset
    dataset = TextFolderDataset(file_directory, file_pattern, tokenizer, max_length)
    eval_dataset = TextFolderDataset(file_directory, eval_file_pattern, tokenizer, max_length)

    # Create the DataLoader
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    eval_data_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    # Initialize the BERT config and model
    config = BertConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=max_length,
    )
    model = BertForMaskedLM(config).to(torch.device("cuda"))

    # Define loss and optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Start recording time and memory
    start_time = time.time()
    
    start_allocated, start_cached = get_gpu_memory_usage()
    print((start_allocated, start_cached))

    # Training loop
    model.train()
    for epoch in range(5):  # Use a smaller number of epochs for demonstration
        total_loss = 0
        for batch in data_loader:
            input_ids = batch['input_ids'].to(torch.device("cuda"))
            attention_mask = batch['attention_mask'].to(torch.device("cuda"))
            labels = input_ids.clone()
            print(f"Batch input_ids shape: {input_ids.shape}")
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            

    # End recording time and memory
    end_time = time.time()
    end_allocated, end_cached = get_gpu_memory_usage()

    # Store the runtime and memory usage
    delta_allocated = end_allocated - start_allocated
    delta_cached = end_cached - start_cached

    # Store the runtime and memory usage difference
    training_times.append(end_time - start_time)
    memory_usages.append(delta_cached)
    del model, optimizer, tokenizer, dataset,  eval_dataset, data_loader, eval_data_loader
    torch.cuda.empty_cache()  # Clear CUDA cache
    avg_loss = total_loss / input_ids.shape[0]
    print(avg_loss)

# Plotting the graphs
plt.figure(figsize=(12, 6))

# Plot for Training Time
plt.subplot(1, 2, 1)
plt.plot(max_lengths, training_times, marker='o')
plt.title('Training Time vs Max Length')
plt.xlabel('Max Length')
plt.ylabel('Training Time (seconds)')

# Plot for Memory Usage
plt.subplot(1, 2, 2)
plt.plot(max_lengths, memory_usages, marker='o', color='red')
plt.title('Memory Usage vs Max Length')
plt.xlabel('Max Length')
plt.ylabel('Memory Usage (MB)')

plt.tight_layout()
plt.show()
