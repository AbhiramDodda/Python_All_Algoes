import numpy as np

# Helper function for scaling and attention mechanism
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute the scaled dot-product attention.
    Args:
    - q: Query matrix
    - k: Key matrix
    - v: Value matrix
    - mask: Optional mask to ignore certain positions
    Returns:
    - output: The result of applying attention weights to the value matrix
    - attention_weights: The attention weights
    """
    d_k = q.shape[-1]
    
    # Scaled dot-product attention formula
    scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Mask to prevent attending to certain positions
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    attention_weights = softmax(scores)
    output = np.matmul(attention_weights, v)
    return output, attention_weights

# Softmax function for normalizing attention scores
def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Multi-head attention using NumPy
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.
        Args:
        - d_model: Dimensionality of the model (total dimension for the attention heads)
        - num_heads: Number of attention heads
        """
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Random weights for linear projections
        self.q_weight = np.random.randn(d_model, d_model)
        self.k_weight = np.random.randn(d_model, d_model)
        self.v_weight = np.random.randn(d_model, d_model)
        self.out_weight = np.random.randn(d_model, d_model)

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        """
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_k)

    def forward(self, q, k, v, mask=None):
        """
        Perform multi-head attention.
        Args:
        - q: Query matrix
        - k: Key matrix
        - v: Value matrix
        - mask: Optional mask
        Returns:
        - output: The result of multi-head attention
        """
        # Apply linear projections
        q = np.matmul(q, self.q_weight)
        k = np.matmul(k, self.k_weight)
        v = np.matmul(v, self.v_weight)

        # Split into multiple heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Perform scaled dot-product attention
        attention_output, _ = scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads and apply final linear layer
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(q.shape[0], -1, self.num_heads * self.d_k)
        return np.matmul(attention_output, self.out_weight)

# Feed-forward network using two fully connected layers
class FeedForward:
    def __init__(self, d_model, d_ff):
        """
        Initialize the feed-forward network.
        Args:
        - d_model: Input/output dimensionality of the model
        - d_ff: Dimensionality of the hidden layer
        """
        self.w1 = np.random.randn(d_model, d_ff)
        self.w2 = np.random.randn(d_ff, d_model)
        self.b1 = np.random.randn(d_ff)
        self.b2 = np.random.randn(d_model)

    def forward(self, x):
        """
        Forward pass through the feed-forward network.
        Args:
        - x: Input data
        Returns:
        - output: The result of the feed-forward network
        """
        # First linear layer followed by ReLU activation
        x = np.maximum(0, np.matmul(x, self.w1) + self.b1)
        # Second linear layer
        return np.matmul(x, self.w2) + self.b2

# Positional encoding
class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        """
        Initialize the positional encoding.
        Args:
        - d_model: Dimensionality of the model
        - max_len: Maximum sequence length for encoding
        """
        self.encoding = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = np.sin(position * div_term)
        self.encoding[:, 1::2] = np.cos(position * div_term)

    def forward(self, x):
        """
        Apply positional encoding.
        Args:
        - x: Input sequence
        Returns:
        - output: Sequence with added positional encoding
        """
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len, :]

# Transformer encoder layer
class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        """
        Initialize the encoder layer.
        Args:
        - d_model: Dimensionality of the model
        - num_heads: Number of attention heads
        - d_ff: Dimensionality of the feed-forward layer
        """
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = lambda x: (x - x.mean()) / x.std()  # LayerNorm approximation
        self.norm2 = lambda x: (x - x.mean()) / x.std()  # LayerNorm approximation

    def forward(self, x, mask=None):
        """
        Forward pass through the encoder layer.
        Args:
        - x: Input sequence
        - mask: Optional mask
        Returns:
        - output: The result of the encoder layer
        """
        attn_output = self.self_attn.forward(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward.forward(x)
        return self.norm2(x + ff_output)

# Full Transformer model
class Transformer:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers):
        """
        Initialize the Transformer model.
        Args:
        - src_vocab_size: Size of the source vocabulary
        - tgt_vocab_size: Size of the target vocabulary
        - d_model: Dimensionality of the model
        - num_heads: Number of attention heads
        - d_ff: Dimensionality of the feed-forward network
        - num_layers: Number of encoder/decoder layers
        """
        self.encoder_embedding = np.random.randn(src_vocab_size, d_model)
        self.decoder_embedding = np.random.randn(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def encode(self, src, mask=None):
        """
        Encode the source sequence.
        Args:
        - src: Input source sequence
        - mask: Optional mask
        Returns:
        - output: Encoded sequence
        """
        # Embed source sequence and add positional encoding
        src = self.positional_encoding.forward(np.matmul(src, self.encoder_embedding))
        # Pass through the encoder layers
        for layer in self.encoder_layers:
            src = layer.forward(src, mask)
        return src

# Example usage
src_vocab_size = 100
tgt_vocab_size = 100
d_model = 64
num_heads = 8
d_ff = 256
num_layers = 2

# Initialize model
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers)

# Example source input
src = np.random.randint(0, src_vocab_size, (32, 10))  # Batch of 32, sequence length of 10
output = model.encode(src)

print(output.shape)  # Expected: (32, 10, d_model)