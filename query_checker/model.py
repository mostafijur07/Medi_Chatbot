import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # Create an embedding vector of size (vocab_size, d_model). if vocab_size = 3, then there are three word in vocabulary and it create a vector for each word represented with d_model=512 mumbers. 

    def forward(self, x): 
        return self.embedding(x) * math.sqrt(self.d_model) # Extract only those word's embediing which are present in the input sentance from the embedding of all word present in the vocabulary. 
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) #used to prevent overfitting by randomly setting some elements to zero during training.

        PE_tensor = torch.zeros(seq_len, d_model) #PE_tensor(Positional Encoding Tensor) initialize with zeros of size (seq_len, d_model).
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #position tensor(pos)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #calculating the div term 
        PE_tensor[:, 0::2] = torch.sin(position * div_term) #update values by applying the formula on even position of positional Encoding tensor(PE_tensor) 
        PE_tensor[:, 1::2] = torch.cos(position * div_term) #update values by applying the formula on odd position of positional Encoding tensor(PE_tensor) 
        PE_tensor = PE_tensor.unsqueeze(0) # Adds a batch dimension to the positional encoding tensor, resultant tensor (batch, seq_len, d_model)
        self.register_buffer('PE_tensor', PE_tensor) # Meaning it won't be considered a model parameter, but it will be part of the model's state (useful for saving and loading the model).

    def forward(self, x):
        x = x + (self.PE_tensor[:, :x.shape[1], :].requires_grad_(False)) # self.pe[:, :x.shape[1], :] selects the positional encodings up to the current sequence length (x.shape[1]), ensuring the positional encoding tensor matches the input sequence length.
        # The .requires_grad_(False) ensures that the positional encodings are not updated during backpropagation, as they are fixed.
        return self.dropout(x) # applies dropout to the result, randomly setting some elements to zero during training to prevent overfitting.
    
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6): #features is the dimension. if a token is represented by d_model = 512 numbers, than number of features is 512.
        super().__init__()
        self.eps = eps # eps is a small value added to the standard deviation to avoid division by zero during normalization.
        self.alpha = nn.Parameter(torch.ones(features)) # self.alpha is a learnable scale parameter initialized to ones with shape [features].
        self.bias = nn.Parameter(torch.zeros(features)) # self.bias is a learnable bias parameter initialized to zeros with shape [features].

    def forward(self, x): # x is the input tensor.
        mean = x.mean(dim=-1, keepdim=True) # calculates the mean of x along the last dimension.
        std = x.std(dim=-1, keepdim=True) # calculates the standard deviation of x along the last dimension.
        return self.alpha * (x - mean) / (std + self.eps) + self.bias  # Applying the LayerNormalization Formula.
    
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer): # x is the input tensor. sublayer is a callable representing a sublayer, such as a multihead attention block or a feedforward block.
        return x + self.dropout(sublayer(self.norm(x))) # The input tensor x is first normalized by self.norm(x), then passed through the sublayer function then Dropout is applied to the output and finaly The result is added to the original input tensor x (residual connection).
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float): # The dimension of the hidden layer inside the feedforward block (typically larger than d_model, e.g., 2048 if d_model is 512).
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # A fully connected layer that maps from d_model to d_ff.
        self.dropout  = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # A fully connected layer that maps from d_ff back to d_model.

    def forward(self, x):
        # Input Tensor (x) of size [batch_size, seq_len, d_model] -> Linear Transformation (x1 = x * W1 + b1) and the transform tensor of size [batch_size, seq_len, d_ff] -> ReLU Activation (x2 = ReLU(x1)) on output tensor of size [batch_size, seq_len, d_ff] -> Dropout (x3 = Dropout(x2)) on output tensor of size [batch_size, seq_len, d_ff] -> Linear Transformation (output = x3 * W2 + b2) so tensor size become [batch_size, seq_len, d_model] which is the final output tensor. 
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False) # w_q, w_k, w_v: Linear layers to project input embeddings to query, key, and value vectors, respectively.
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False) # w_o: Linear layer to project the concatenated output of all heads back to d_model dimensions.
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # Calculates attention scores by computing the dot product between query and key, scaled by the square root of d_k.
        if mask is not None: # Optionally applies a mask to the attention scores.
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # Converts attention scores to probabilities.
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores # Computes the weighted sum of the value vectors based on the attention scores and then return.

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # Projects the input embeddings to query, key, and value vectors.
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # Reshapes and transposes the projected vectors to prepare for multi-head attention.
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) # Computes the attention using the static method.
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # Concatenates the multi-head outputs.
        return self.w_o(x) # Projects the concatenated output back to d_model dimensions.

class EncoderBlock(nn.Module): # features: The dimension of the input and output embeddings (i.e., the model dimension d_model).
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block # IN this case the MultiHeadAttentionBlock.
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)]) # It creates two instances of the ResidualConnection class, storing them in a ModuleList. Each residual connection includes layer normalization and dropout.

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) #The first residual connection from the ModuleList wraps the self-attention block.
        x = self.residual_connections[1](x, self.feed_forward_block) # The second residual connection from the ModuleList wraps the feed-forward block.
        return x
    
class Encoder(nn.Module): # Transformer model that consists of multiple EncoderBlock layers(In the paper it is 6 of them), followed by a layer normalization.
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers # layers: A ModuleList of EncoderBlock instances that define the layers of the encoder.
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers: # The input tensor x is passed sequentially through each EncoderBlock in self.layers.
            x = layer(x, mask) # Each EncoderBlock processes the tensor using self-attention, feed-forward network, residual connections, and normalization.
        return self.norm(x) # After passing through all encoder layers, the final output is normalized using self.norm.
    
class ProjectionLayer(nn.Module): # Defines a simple linear transformation layer, which is typically used to project the output of the Transformer model to the desired number of output classes.
    def __init__(self, d_model, num_classes) -> None: # num_classes: The number of output classes(target variable).
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes) # The linear layer that performs this projection.

    def forward(self, x) -> None:
        return self.proj(x) # This output can then be used to compute class scores, which can be further processed (e.g., by applying a softmax function) to obtain class probabilities.


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, src_embed: InputEmbeddings, src_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:  # Removed decoder-related parameters
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask): # Input Embedding and Positional Embedding then Encoder block execute.
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def project(self, x): # Project the encoder output.
        return self.projection_layer(x[:, 0, :])  # Taking only the [CLS] token representation for classification
        # The code self.projection_layer(x[:, 0, :]) extracts only the representation(enbedding) of the first token(<cls>) from the encoder's output for classification purposes and then the extracted embedding is used to project or classify the class or label because:
        # During the encoding process, this [CLS] token's representation is influenced by all other tokens in the sequence due to the self-attention mechanism. Hence, it contains a summary of the entire sequence.

    def forward(self, src, src_mask):
        encoder_output = self.encode(src, src_mask)
        return self.project(encoder_output)
    
def build_transformer(src_vocab_size: int, num_classes: int, seq_len: int, d_model: int, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    src_pos = PositionalEncoding(d_model, seq_len, dropout)
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    projection_layer = ProjectionLayer(d_model, num_classes)  # Changed vocab_size to num_classes
    transformer = Transformer(encoder, src_embed, src_pos, projection_layer)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer