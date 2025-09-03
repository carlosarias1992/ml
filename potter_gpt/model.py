import torch.nn.functional as F
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader
import os
import time
from tqdm import tqdm

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism, a key component of the Transformer model.
    It allows the model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, dim_in, dim_out, context_length, dropout, num_heads):
        super().__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5  # For scaling dot product

        # Query, Key, Value projections
        self.W_q = nn.Linear(dim_in, dim_out)
        self.W_k = nn.Linear(dim_in, dim_out)
        self.W_v = nn.Linear(dim_in, dim_out)

        # Final output projection
        self.out_proj = nn.Linear(dim_out, dim_out)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Causal mask to prevent attending to future positions
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        # Register buffer makes sure the mask is not considered a model parameter
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch_size, embed_size, _ = x.size()  # Batch, sequence_length, embedding_dim

        # Linear projections
        Q = self.W_q(x)  # (batch_size, seq_len, dim_out)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, embed_size, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, embed_size, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, embed_size, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        attention_scores = (Q @ K.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)

        # Apply causal mask
        mask = self.mask[:embed_size, :embed_size] == 1
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        context = attention_weights @ V  # (B, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, embed_size, -1)  # (batch_size, seq_len, dim_out)

        return self.out_proj(context)


class LayerNorm(nn.Module):
    """
    Implements Layer Normalization. It normalizes the inputs across the features
    for each data point independently.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5 # Epsilon, a small number to prevent division by zero
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Implements the Gaussian Error Linear Unit (GELU) activation function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    """
    A simple feed-forward network with a GELU activation in between two linear layers.
    """
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerLayer(nn.Module):
    """
    A single Transformer block, which consists of multi-head attention and a feed-forward network.
    Uses residual connections and layer normalization.
    """
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            dim_in=config.embedding_dim,
            dim_out=config.embedding_dim,
            context_length=config.context_length,
            num_heads=config.attention_head,
            dropout=config.drop_rate
        )
        self.ff = FeedForward(config.embedding_dim)
        self.norm1 = LayerNorm(config.embedding_dim)
        self.norm2 = LayerNorm(config.embedding_dim)
        self.drop_shortcut = nn.Dropout(config.drop_rate)

    def forward(self, x):
        # First residual connection (around multi-head attention)
        residual1 = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + residual1

        # Second residual connection (around feed-forward network)
        residual2 = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + residual2
        return x


class GPTModel(nn.Module):
    """
    The main GPT-2 model architecture.
    """
    def __init__(self, config):
        super().__init__()
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.drop_rate)

        # A sequence of Transformer layers
        self.transformer_layers = nn.Sequential(
            *[TransformerLayer(config) for _ in range(config.transformer_layer)]
        )

        # Final normalization and output layer
        self.final_norm = LayerNorm(config.embedding_dim)
        self.final_output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # Get token embeddings
        token_embedding = self.token_embedding(in_idx)
        # Get positional embeddings
        # The device is inferred from the input index tensor 'in_idx'
        positional_embedding = self.positional_embedding(torch.arange(seq_len, device=in_idx.device))
        # Add them together
        x = token_embedding + positional_embedding
        x = self.dropout(x)

        x = self.transformer_layers(x)
        x = self.final_norm(x)
        logits = self.final_output(x)
        return logits


class Config:
    """Configuration class for the GPT model."""
    def __init__(self):
        self.vocab_size = 50257
        self.embedding_dim = 768
        self.drop_rate = 0.1
        self.transformer_layer = 12
        self.attention_head = 12
        self.context_length = 1024


class PotterDataset(Dataset):
    def __init__(self, data, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.data = tokenizer.encode(data)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train_model(model, dataloader, optimizer, device, epochs=10, should_train=True):
    """
    Trains the model with a progress bar and a flag to enable/disable training.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): The data loader for training data.
        optimizer (Optimizer): The optimizer to use.
        device (torch.device): The device to train on.
        epochs (int): Number of training epochs.
        should_train (bool): If False, skips the training loop.
    """
    if not should_train:
        print("Skipping training as per configuration.")
        return

    print("\nStarting training...")
    model.train() # Set model to training mode
    training_start_time = time.time()

    for epoch in range(epochs):
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for xb, yb in progress_bar:
            # Move data to the selected device
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # **NEW**: Update the progress bar with the current loss on each step
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Average Loss: {avg_loss:.4f} ---")

    elapsed_time = time.time() - training_start_time
    print(f"🎉 Training finished in {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
    
    # Save the trained model
    torch.save(model.state_dict(), "gpt_potter.pt")
    print("Model saved to gpt_potter.pt")


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Generates text token by token."""
    model.eval() # Ensure model is in evaluation mode for inference
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def run_inference(model, tokenizer, config, device):
    """
    Runs the text generation (inference) part of the script.
    
    Args:
        model (nn.Module): The GPT model.
        tokenizer: The tokenizer.
        config (Config): The model configuration object.
        device (torch.device): The device to run inference on.
    """
    print("\nRunning inference to generate sample text...")
    
    start_context = "Harry entered"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    
    # Generate text
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=1000,
        context_size=config.context_length
    )
    decoded_text = tokenizer.decode(out.squeeze(0).cpu().tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nDecoded Full Text:", decoded_text)


def main():
    """Main function to orchestrate the script execution."""
    
    # --- Configuration and Setup ---
    PERFORM_TRAINING = False 
    
    config = Config()
    model = GPTModel(config)
    tokenizer = tiktoken.get_encoding("gpt2")

    # --- Device Setup ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    model.to(device)

    # --- Load Pre-trained Model (if available) ---
    model_file = "gpt_potter.pt"
    if os.path.exists(model_file):
        print(f"\nLoading pre-trained model from '{model_file}'")
        model.load_state_dict(torch.load(model_file, map_location=device))
    else:
        print(f"\nWarning: Model file '{model_file}' not found.")

    # --- Training Phase ---
    dataset_file = "harrypotter_clean.txt"
    if os.path.exists(dataset_file):
        with open(dataset_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        dataset = PotterDataset(raw_text, tokenizer)
        dataloader = DataLoader(dataset, batch_size=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Call the training function
        train_model(model, dataloader, optimizer, device, epochs=3, should_train=PERFORM_TRAINING)
    else:
        print(f"\n'{dataset_file}' not found. Skipping training.")
        
    # --- Inference Phase ---
    run_inference(model, tokenizer, config, device)


# --- Main Execution Entry Point ---
if __name__ == '__main__':
    main()
