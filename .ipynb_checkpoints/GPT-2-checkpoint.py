from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from glob import glob
import os
from tqdm import tqdm
import time

torch.manual_seed(0)
# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Retrieve the correct vocabulary size from the tokenizer
correct_vocab_size = tokenizer.vocab_size

# Update your GPT2 configuration with the correct vocabulary size
config = GPT2Config(
    vocab_size=correct_vocab_size,  # Updated size
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)

# Initialize the model with the updated configuration
model = GPT2LMHeadModel(config)

# Check for CUDA GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transfer the model to the GPU (if available)
model.to(device)

# Define your TextFolderDataset
# Define your TextFolderDataset
class TextFolderDataset(Dataset):
    def __init__(self, file_directory, file_pattern, max_length):
        self.filepaths = glob(os.path.join(file_directory, file_pattern))
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad_token
        self.max_length = max_length

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        with open(self.filepaths[idx], 'r', encoding='utf-8') as file:
            text = file.read()

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True, 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length', 
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Create your dataset and data loader
file_directory = "openwebtext/"
file_pattern = "urlsf_subset01-3[2,3]*"
max_length = 512
dataset = TextFolderDataset(file_directory, file_pattern, max_length)
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training Loop
from transformers import AdamW
from tqdm import tqdm

optimizer = AdamW(model.parameters(), lr=5e-5)
accumulation_steps = 2
num_epochs = 50

start_time = time.time()

for epoch in tqdm(range(num_epochs)):
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_dataloader):
        # Transfer the data to the GPU
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(inputs, labels=labels)
        loss = outputs.loss

        # Normalize loss to account for accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
    
    avg_loss = total_loss / (len(train_dataloader) / accumulation_steps)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    if avg_loss < 0.8:
        break

end_time = time.time()
print(f"The time needed for training is {end_time - start_time} seconds.")
