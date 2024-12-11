"""
Implementation of a wrapper class MoEUTTimeSeriesDecoder that adapts the MoEUT model for time series prediction and makes it compatible with the TimeSeriesDecoder class. This class allows you to use MoEUT as a drop-in replacement for TimeSeriesDecoder.

Key Steps:

Input Projection:
* Project the input time series data from d_input dimensions to d_model dimensions using a linear layer.


Adapt MoEUT Model:
* Remove the embedding layer from MoEUTLM since we are dealing with continuous inputs rather than token IDs.
* Ensure the MoEUT model can accept inputs of shape [B, T, d_model] and produce outputs accordingly.


Output Projection:
* After processing through the MoEUT layers, take the output corresponding to the last time step (or use a pooling mechanism if appropriate).
* Apply a layer normalization and a linear layer to project the output to the desired number of output features (n_outputs).


Positional Encoding:
* Use Rotary Positional Encoding to incorporate positional information, which is important in time series data.

Compatibility with TimeSeriesDecoder:
* Match the interface of the TimeSeriesDecoder class, including the initialization parameters and the forward method.
* Ensure that the forward method accepts inputs of shape [B, T, d_input] and returns outputs of shape [B, n_outputs].
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# Re-use the RotaryPositionalEmbedding and MoEUT implementation from the previous response

class MoEUTTimeSeriesDecoder(nn.Module):
    """
    MoEUT-based Time Series Decoder compatible with TimeSeriesDecoder.

    This class wraps the MoEUT model for time series prediction, making it compatible
    with the TimeSeriesDecoder interface.
    """

    def __init__(
        self,
        d_input: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_expert_size: int = 256,
        ff_n_experts: int = 4,
        ff_k: int = 2,
        dropout: float = 0.1,
        n_outputs: int = 2,
        entropy_reg: float = 0.01,
        use_rotary: bool = True,
        base: int = 10000,
        **kwargs
    ):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.input_projection = nn.Linear(d_input, d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize MoEUT without the embedding layer
        self.transformer = MoEUT(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            n_experts=ff_n_experts,
            expert_size=ff_expert_size,
            k=ff_k,
            dropout=dropout,
            entropy_reg=entropy_reg,
            base=base,
            use_rotary=use_rotary,
            **kwargs
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, n_outputs)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the MoEUT Time Series Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_input].

        Returns:
            torch.Tensor: Output tensor of shape [B, n_outputs].
        """
        B, T, C = x.shape  # batch size, sequence length, input dimensions
        x = self.input_projection(x)  # Project to model dimension
        x = self.dropout(x)

        # Pass through MoEUT transformer
        x, reg_loss = self.transformer(x)

        # Use the output at the last time step for prediction
        x = self.ln_f(x[:, -1, :])  # Shape: [B, d_model]
        x = self.output_projection(x)  # Shape: [B, n_outputs]
        return x

"""
Required Modifications to MoEUT:
We need to adjust the MoEUT class to ensure it works seamlessly with continuous inputs and is compatible with the new wrapper class. Here are the key modifications:

Remove Embedding Layer:
* Since we're dealing with continuous inputs, we no longer need the embedding layer present in MoEUTLM.
* The inputs are now projected directly using a linear layer (input_projection) in the wrapper class.

Ensure Input Shapes are Handled Correctly:
* The MoEUT model should accept inputs of shape [B, T, d_model] and return outputs of the same shape.

Adjust Positional Encoding:
* Use the RotaryPositionalEmbedding within the attention layers to handle positional information inherent in time series data.

Simplify Attention Mechanism if Needed:
* If specific attention mechanisms (e.g., multi-scale attention) are required, they can be incorporated or adapted as necessary.

Below is the adjusted MoEUT class:
"""


class MoEUT(nn.Module):
    """
    Mixture of Experts Universal Transformer (MoEUT) model for time series data.
    """

    def __init__(self, n_layers: int, d_model: int, n_heads: int, n_experts: int,
                 expert_size: int, k: int, dropout: float = 0.0,
                 expert_dropout: float = 0.0, base: int = 10000,
                 entropy_reg: float = 0.01, use_rotary: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            MoEUTLayer(d_model, n_heads, n_experts, expert_size, k=k,
                       dropout=dropout, expert_dropout=expert_dropout, base=base,
                       use_rotary=use_rotary)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.entropy_reg_weight = entropy_reg  # Regularization weight

    def forward(self, x, attention_mask=None):
        reg_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x = layer(x, attention_mask)
            # Accumulate regularization loss
            reg_loss = reg_loss + layer.moe.get_reg_loss() * self.entropy_reg_weight

        x = self.ln_f(x)
        return x, reg_loss


# Adjusted MoEUTLayer:
class MoEUTLayer(nn.Module):
    """
    Single layer of MoEUT, consisting of attention with optional RoPE and a SigmaMoE feedforward network.
    """

    def __init__(self, d_model: int, n_heads: int, n_experts: int, expert_size: int, k: int,
                 dropout: float = 0.0, expert_dropout: float = 0.0, base: int = 10000,
                 use_rotary: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = RotaryMultiheadAttention(d_model, n_heads, dropout=dropout, base=base,
                                                  use_rotary=use_rotary)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.moe = SigmaMoE(d_model, n_experts, expert_size, k=k, expert_dropout=expert_dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # Attention block
        x_norm = self.ln1(x)
        attn_output = self.attention(x_norm, attention_mask=attention_mask)
        x = x + self.dropout1(attn_output)

        # MoE Feedforward block
        x_norm = self.ln2(x)
        moe_output = self.moe(x_norm)
        x = x + self.dropout2(moe_output)

        return x


# Adjusted RotaryMultiheadAttention:
class RotaryMultiheadAttention(nn.Module):
    """
    Multi-head attention with optional Rotary Positional Embedding.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, base=10000, use_rotary=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_rotary = use_rotary

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        # Projection layers
        self.q_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.k_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.v_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.q_proj_bias = nn.Parameter(torch.zeros(embed_dim))
        self.k_proj_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_proj_bias = nn.Parameter(torch.zeros(embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Positional embedding
        if self.use_rotary:
            self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, base=base)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.q_proj_bias)
        nn.init.zeros_(self.k_proj_bias)
        nn.init.zeros_(self.v_proj_bias)

    def forward(self, x, attention_mask=None):
        B, S, D = x.size()

        # Query, Key, Value projections
        q = F.linear(x, self.q_proj_weight.t(), self.q_proj_bias)
        k = F.linear(x, self.k_proj_weight.t(), self.k_proj_bias)
        v = F.linear(x, self.v_proj_weight.t(), self.v_proj_bias)

        # Reshape for multi-head attention
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, S, head_dim]
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k if enabled
        if self.use_rotary:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # [B, num_heads, S, S]

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)  # [B, num_heads, S, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)  # [B, S, D]

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output


# Usage Example:
# # Example usage
# d_input = 3
# d_model = 64
# n_heads = 4
# n_layers = 6
# ff_n_experts = 4
# expert_size = 256
# ff_k = 2
# dropout = 0.1
# n_outputs = 2

# model = MoEUTTimeSeriesDecoder(
#     d_input=d_input,
#     d_model=d_model,
#     n_heads=n_heads,
#     n_layers=n_layers,
#     ff_expert_size=expert_size,
#     ff_n_experts=ff_n_experts,
#     ff_k=ff_k,
#     dropout=dropout,
#     n_outputs=n_outputs,
#     entropy_reg=0.01
# )

# # Input tensor of time series data
# batch_size = 32
# seq_len = 100
# input_data = torch.randn(batch_size, seq_len, d_input)

# # Forward pass
# output = model(input_data)

# print(output.shape)  # Should be [batch_size, n_outputs]

"""
Explanation:


Input and Output Dimensions:
* The input tensor x has the shape [B, T, d_input].
* After processing through the model, the output tensor has the shape [B, n_outputs].


Projection Layers:
* input_projection projects the input from d_input to d_model.
* output_projection projects the final output from d_model to n_outputs.


Transformer Layers:
* The MoEUT model consists of multiple layers of transformations, incorporating mixture-of-experts and attention mechanisms.


Positional Encoding:
* Rotary Positional Encoding (RotaryPositionalEmbedding) is used to inject positional information, which is important for time series data.



Regularization Loss:
* The regularization loss from the MoE layers is accumulated and can be used during training to encourage balanced expert utilization.



Compatibility:
* The MoEUTTimeSeriesDecoder class matches the interface of the TimeSeriesDecoder, allowing it to be used as a drop-in replacement.


Training:
* Remember to include the regularization loss (reg_loss) in your loss function during training to promote balanced expert usage.


Further Customization:
* If you need to incorporate multi-scale attention or other specific mechanisms, you can extend the attention layers or modify the MoEUTLayer accordingly.
"""
