import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from flash_attn_interface import flash_attention
from torch.utils.data import DataLoader, Dataset
from glob import glob
from tqdm import tqdm
import time
from transformers import AdamW
import os
from torch.cuda.amp import autocast, GradScaler
torch.manual_seed(0)
class FlashAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.n_embd / config.num_attention_heads)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, hidden_size = hidden_states.size()

        # Apply linear transformations and reshape for multi-head attention
        q = self.query(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        k = self.key(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        v = self.value(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)

        # Transpose to get the shape (batch_size, num_heads, seq_length, head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Call flash_attention
        attention_output = flash_attention(q, k, v, dropout_prob=0.0, causal=False)
        # Reshape attention output back to (batch_size, seq_length, hidden_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        return (attention_output,)

class CustomGPT2Block(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        # Replace the standard attention with FlashAttentionLayer
        self.attn = FlashAttentionLayer(config)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        # Attention block
        attn_outputs = self.attn(hidden_states, attention_mask)
        a = attn_outputs[0]  # Output of the flash attention

        # Feed Forward block
        mlp_output = self.mlp(self.ln_2(a))
        hidden_states = mlp_output + a

        outputs = (hidden_states,) + attn_outputs[1:]  # Add attention outputs if they are present

        return outputs

class CustomGPT2Model(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # Replace standard GPT2 blocks with CustomGPT2Block
        self.h = nn.ModuleList([CustomGPT2Block(config) for _ in range(config.n_layer)])
        # Initialize the embedding layers
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        default_layer_norm_eps = 1e-5

        # Initialize the final layer normalization with the default epsilon
        self.ln_f = nn.LayerNorm(config.n_embd, eps=default_layer_norm_eps)

        # Initialize the language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Prepare inputs to the model
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        if position_ids is None:
            # Create default position_ids if None provided
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, input_shape[-1]).to(input_ids.device)

        # Get embeddings from GPT2Model
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        for block in self.h:
            outputs = block(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]

        # Final layer normalization and head
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Retrieve the correct vocabulary size from the tokenizer
correct_vocab_size = tokenizer.vocab_size

# Update your GPT2 configuration with the correct vocabulary size
config = GPT2Config(
    vocab_size=correct_vocab_size,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)
trad_model = GPT2LMHeadModel(config)
# Initialize the custom model with Flash Attention
custom_model = CustomGPT2Model(config)
traditional_params = set(trad_model.state_dict().keys())
custom_params = set(custom_model.state_dict().keys())

# Find common parameters
common_params = traditional_params & custom_params

# Copy only common parameters
common_state_dict = {name: trad_model.state_dict()[name] for name in common_params}
custom_model.load_state_dict(common_state_dict, strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transfer the custom_model to the GPU (if available)
custom_model.to(device)

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

from tqdm import tqdm

optimizer = AdamW(custom_model.parameters(), lr=5e-5)
accumulation_steps = 2
num_epochs = 50

start_time = time.time()
scaler = GradScaler()
# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in tqdm(range(num_epochs)):
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_dataloader):
        # Transfer the data to the GPU
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        with autocast():
            logits = custom_model(inputs)
            # Reshape logits and labels for CrossEntropyLoss
            # logits shape: [batch_size, seq_length, vocab_size]
            # labels shape: [batch_size, seq_length]
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Backward pass and optimization
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / (len(train_dataloader) / accumulation_steps)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    if avg_loss < 1.5:
        break


end_time = time.time()
print(f"The time needed for training is {end_time - start_time} seconds.")
