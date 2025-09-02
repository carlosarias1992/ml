import torch.nn.functional as F
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader
import os
import time

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


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generates text token by token.
    `idx` is a (B, T) tensor of indices in the current context.
    """
    for _ in range(max_new_tokens):
        # Crop context if it's longer than what the model supports
        idx_cond = idx[:, -context_size:]

        # Get the model's predictions (logits)
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step's logits
        logits = logits[:, -1, :]

        # Greedily sample the token with the highest probability
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append the new token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# --- Main Execution ---

# 1. Initialize Model and Configuration
config = Config()
model = GPTModel(config)

# --- Device Setup (CPU or Apple Silicon GPU) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

model.to(device) # Move model to the selected device
model.eval()  # Set model to evaluation mode

# 2. Setup Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# --- Example of asking a question (Shakespeare theme) ---
# By structuring the prompt this way, we guide the model to provide an answer
# in a style it would have learned from Shakespeare's works.
start_context = "To be"

# 3. Encode the input text and move to device
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)

print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
print("\nInput text:", start_context)
print("Encoded input text:", encoded)
print("encoded_tensor.shape:", encoded_tensor.shape)

# Dummy model weights file for demonstration purposes.
# In a real scenario, you would train your model on Shakespeare's works.
model_file = "gpt_shakespeare.pt"
if not os.path.exists(model_file):
    print(f"\nWarning: Model file '{model_file}' not found.")
    print("Generating random weights for demonstration.")
else:
    print(f"\nLoading pre-trained model from '{model_file}'")
    model.load_state_dict(torch.load(model_file, map_location=device))


# 4. Generate text based on the prompt
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=20,  # Generate up to 20 new tokens
    context_size=config.context_length
)
# Move output tensor to CPU for decoding if it's not already
decoded_text = tokenizer.decode(out.squeeze(0).cpu().tolist())

print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
print("\nGenerated Output Tensor:", out)
print("Output length:", len(out[0]))
print("\nDecoded Full Text:", decoded_text)


# --- (Optional) Training Code ---
# The following code is for training the model.
# You would need a 'shakespeare.txt' file or your own dataset to run this.

class ShakespeareDataset(Dataset):
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

# Helper function to format time
def format_time(seconds):
    """Converts seconds into HH:MM:SS format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def train(model, dataloader, optimizer, device, epochs=10):
    model.train() # Set model to training mode
    
    # 1. Get total batches for the entire training process
    num_batches = len(dataloader)
    total_batches = num_batches * epochs
    log_interval = 1 # Log at each step

    # 2. Record the start time for the entire training
    training_start_time = time.time()

    for epoch in range(epochs):
        total_loss = 0
        for i, (xb, yb) in enumerate(dataloader):
            # Move data to the selected device
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # --- CORRECT ETA CALCULATION ---
            # 3. Calculate how many batches have been completed so far
            batches_done = epoch * num_batches + (i + 1)
            
            # 4. Calculate elapsed time and average time per batch
            elapsed_time = time.time() - training_start_time
            avg_time_per_batch = elapsed_time / batches_done
            
            # 5. Calculate remaining batches and estimate remaining time
            batches_remaining = total_batches - batches_done
            eta_seconds = avg_time_per_batch * batches_remaining
            eta_formatted = format_time(eta_seconds)
            # --- END OF ETA CALCULATION ---

            # Print progress indicator
            if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}, ETA: {eta_formatted}")
            
        avg_loss = total_loss / num_batches
        print(f"--- Epoch {epoch+1} Average Loss: {avg_loss:.4f} ---")

    total_training_time = format_time(time.time() - training_start_time)
    print(f"🎉 Training finished in {total_training_time}")

# # Example training loop (commented out by default)
# if __name__ == '__main__':
#     # Check if a dataset file exists
#     if os.path.exists("shakespeare.txt"):
#         with open("shakespeare.txt", "r", encoding="utf-8") as f:
#             raw_text = f.read()

#         # Prepare dataset and dataloader
#         dataset = ShakespeareDataset(raw_text, tokenizer)
#         dataloader = DataLoader(dataset, batch_size=32) # Increased batch size for GPU

#         # Train the model
#         optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#         print("\nStarting training...")
#         train(model, dataloader, optimizer, device, epochs=3) # Pass device to train function

#         # Save the trained model
#         torch.save(model.state_dict(), "gpt_shakespeare.pt")
#         print("Model saved to gpt_shakespeare.pt")
#     else:
#         print("\n'shakespeare.txt' not found. Skipping training.")

