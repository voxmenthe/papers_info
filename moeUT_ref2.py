import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def entropy_l(l: torch.Tensor) -> torch.Tensor:
    """
    Computes the entropy of a tensor's log-softmax values.

    Args:
    l (torch.Tensor): Input tensor representing log-softmax values.

    Returns:
    torch.Tensor: Entropy of the log-softmax values.
    """
    return - (l * l.exp()).sum(-1)


def entropy_reg(sel: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Calculates the entropy regularization term for MoE routing.

    Encourages uniform routing of experts.

    Args:
    sel (torch.Tensor): Selection scores for experts.
    dim (int): Dimension along which to compute softmax.

    Returns:
    torch.Tensor: Entropy regularization loss.
    """
    sel = F.log_softmax(sel, dim=dim)
    sel = sel.logsumexp(dim=dim) - math.log(sel.shape[dim])
    return - entropy_l(sel).mean()


class SigmaMoE(nn.Module):
    """
    SigmaMoE layer that efficiently dispatches inputs to expert networks.
    We need to efficiently route inputs to their selected experts and accumulate the outputs without using explicit loops over the batch and sequence dimensions.
    Explanation:
        Efficient Dispatching:
        Flattened the inputs and gating tensors for efficient processing.
        Used vectorized operations and indexing to dispatch inputs to the selected experts.
        Avoided explicit loops over batch and sequence dimensions.

        Accumulating Outputs:
        Used index_add_ to accumulate outputs back to their original positions.
    """

    def __init__(self, d_model: int, n_experts: int, expert_size: int, k: int,
                 activation=F.relu, expert_dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.k = k
        self.activation = activation
        self.expert_dropout = expert_dropout

        # Gating network parameters
        self.expert_sel = nn.Linear(d_model, n_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_size),
                nn.ReLU(),
                nn.Linear(expert_size, d_model)
            ) for _ in range(n_experts)
        ])

        # For regularization loss
        self.sel_hist = []

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.expert_sel.weight)
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def get_reg_loss(self) -> torch.Tensor:
        """
        Computes the regularization loss based on expert selection entropy.

        Returns:
        torch.Tensor: Regularization loss.
        """
        if not self.sel_hist:
            return torch.tensor(0.0, device=self.expert_sel.weight.device)

        # Stack selection histories and compute entropy
        sel_scores = torch.cat(self.sel_hist, dim=0)  # [num_calls, B, S, n_experts]
        reg_loss = entropy_reg(sel_scores, dim=-1)
        self.sel_hist = []
        return reg_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SigmaMoE layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, S, D]

        Returns:
            torch.Tensor: Output tensor of shape [B, S, D]
        """
        B, S, D = x.size()

        # Compute selection scores
        sel_scores = self.expert_sel(x)  # [B, S, n_experts]

        if self.training:
            self.sel_hist.append(sel_scores.detach())

        # Apply gating activation
        sel_probs = torch.sigmoid(sel_scores)  # [B, S, n_experts]

        # Apply expert dropout
        if self.training and self.expert_dropout > 0:
            dropout_mask = (torch.rand_like(sel_probs) < self.expert_dropout)
            sel_probs = sel_probs.masked_fill(dropout_mask, float('-inf'))

        # Select top-k experts
        sel_vals, sel_indices = torch.topk(sel_probs, self.k, dim=-1)  # Both are [B, S, k]

        # Prepare for dispatching
        B_flat = B * S * self.k
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.k, -1)  # [B, S, k, D]
        x_flat = x_expanded.reshape(B_flat, D)  # [B*S*k, D]
        sel_indices_flat = sel_indices.reshape(B_flat)  # [B*S*k]
        sel_vals_flat = sel_vals.reshape(B_flat)  # [B*S*k]

        # Create buffers to hold outputs
        output = torch.zeros(B, S, D, device=x.device)

        # Dispatch inputs to experts
        for expert_id in range(self.n_experts):
            # Find positions where the expert is selected
            mask = (sel_indices_flat == expert_id)
            if mask.sum() == 0:
                continue
            x_expert = x_flat[mask]  # Inputs for this expert
            weights = sel_vals_flat[mask].unsqueeze(1)  # Gating weights

            # Compute expert output
            y = self.experts[expert_id](x_expert)  # [N, D]
            y = y * weights  # Scale by gating weights

            # Accumulate outputs
            indices = mask.nonzero(as_tuple=False).squeeze()  # Indices in x_flat
            # Map indices back to (B, S)
            batch_indices = indices // (S * self.k)
            seq_indices = (indices % (S * self.k)) // self.k
            output.index_add_(0, batch_indices, torch.zeros_like(output[batch_indices]).index_add_(1, seq_indices, y))
            # Alternatively, we can accumulate directly using advanced indexing
            output[batch_indices, seq_indices] += y

        return output


class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE).
    """

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None, offset=0):
        """
        Applies rotary positional embedding to input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape [..., seq_len, dim]
            seq_len (int, optional): Sequence length. If None, inferred from x.
            offset (int): Position offset.

        Returns:
            torch.Tensor: Tensor with RoPE applied.
        """
        if seq_len is None:
            seq_len = x.size(-2)

        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype) + offset  # Shape [seq_len]
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)  # Shape [seq_len, dim/2]

        emb = torch.cat([freqs, freqs], dim=-1)  # Shape [seq_len, dim]
        sinusoid = torch.sin(emb)
        cosinusoid = torch.cos(emb)

        return (x * cosinusoid) + (rotate_half(x) * sinusoid)


def rotate_half(x):
    """Helper function to rotate the last dimension."""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class RotaryMultiheadAttention(nn.Module):
    """
    Multi-head attention with Rotary Positional Embedding.
    Custom Attention Mechanism with Rotary Positional Encoding
    Attention mechanism that includes rotary positional encoding, replicating the behavior of SwitchHeadRope.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, base=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        # Projection layers
        self.q_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.k_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.v_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        if bias:
            self.q_proj_bias = nn.Parameter(torch.empty(embed_dim))
            self.k_proj_bias = nn.Parameter(torch.empty(embed_dim))
            self.v_proj_bias = nn.Parameter(torch.empty(embed_dim))
        else:
            self.register_parameter('q_proj_bias', None)
            self.register_parameter('k_proj_bias', None)
            self.register_parameter('v_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Positional embedding
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, base=base)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        if self.q_proj_bias is not None:
            nn.init.zeros_(self.q_proj_bias)
            nn.init.zeros_(self.k_proj_bias)
            nn.init.zeros_(self.v_proj_bias)

    def forward(self, x, attention_mask=None):
        """
        Forward pass of the Rotary Multi-head Attention.

        Args:
            x (torch.Tensor): Input tensor of shape [B, S, D]
            attention_mask (torch.Tensor, optional): Attention mask of shape [B, S]

        Returns:
            torch.Tensor: Output tensor of shape [B, S, D]
        """
        B, S, D = x.size()

        # Query, Key, Value projections
        q = F.linear(x, self.q_proj_weight, self.q_proj_bias)
        k = F.linear(x, self.k_proj_weight, self.k_proj_bias)
        v = F.linear(x, self.v_proj_weight, self.v_proj_bias)

        # Reshape for multi-head attention
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, S, head_dim]
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k
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


class MoEUTLayer(nn.Module):
    """
    Single layer of MoEUT, consisting of attention with RoPE and a SigmaMoE feedforward network.
    Assemble the MoEUTLayer and the overall MoEUT model using the components above.
    """

    def __init__(self, d_model: int, n_heads: int, n_experts: int, expert_size: int, k: int,
                 dropout: float = 0.0, expert_dropout: float = 0.0, base: int = 10000):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = RotaryMultiheadAttention(d_model, n_heads, dropout=dropout, base=base)
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


class MoEUT(nn.Module):
    """
    Mixture of Experts Universal Transformer (MoEUT) model.
    """

    def __init__(self, n_layers: int, d_model: int, n_heads: int, n_experts: int,
                 expert_size: int, k: int, dropout: float = 0.0,
                 expert_dropout: float = 0.0, base: int = 10000):
        super().__init__()
        self.layers = nn.ModuleList([
            MoEUTLayer(d_model, n_heads, n_experts, expert_size, k=k,
                       dropout=dropout, expert_dropout=expert_dropout, base=base)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.entropy_reg_weight = 0.01  # Regularization weight (adjust as needed)

    def forward(self, x, attention_mask=None):
        reg_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x = layer(x, attention_mask)
            # Accumulate regularization loss
            reg_loss = reg_loss + layer.moe.get_reg_loss() * self.entropy_reg_weight

        x = self.ln_f(x)
        return x, reg_loss


class MoEUTLM(nn.Module):
    """
    MoEUT Language Model.
    Wrap the MoEUT model with an embedding layer and a language modeling head for language modeling tasks.
    """

    def __init__(self, vocab_size: int, n_layers: int, d_model: int, n_heads: int,
                 n_experts: int, expert_size: int, k: int, dropout: float = 0.0,
                 expert_dropout: float = 0.0, base: int = 10000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = MoEUT(n_layers, d_model, n_heads, n_experts, expert_size,
                                 k=k, dropout=dropout, expert_dropout=expert_dropout, base=base)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the language model.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [B, S]
            attention_mask (torch.Tensor, optional): Attention mask of shape [B, S]

        Returns:
            logits (torch.Tensor): Output logits of shape [B, S, vocab_size]
            reg_loss (torch.Tensor): Regularization loss from the MoE layers
        """
        x = self.embedding(input_ids)  # [B, S, D]
        x, reg_loss = self.transformer(x, attention_mask=attention_mask)
        logits = self.lm_head(x)
        return logits, reg_loss


# Notes and Considerations

# Expert Dispatching: The SigmaMoE layer efficiently dispatches inputs to selected experts using vectorized operations and index_add_ for accumulating outputs.
# Rotary Positional Encoding: The attention mechanism (RotaryMultiheadAttention) includes Rotary Positional Embedding, capturing the original model's behavior.
# Regularization Loss: Regularization loss is accumulated during the forward pass, and the entropy regularization encourages balanced expert utilization.



# Example Usage
# # Example usage
# vocab_size = 10000
# n_layers = 6
# d_model = 512
# n_heads = 8
# n_experts = 4
# expert_size = 2048
# k = 2
# dropout = 0.1
# expert_dropout = 0.1

# model = MoEUTLM(vocab_size, n_layers, d_model, n_heads, n_experts, expert_size, k, dropout, expert_dropout)

# # Input tensor of token IDs
# input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
# attention_mask = None  # Or provide an appropriate attention mask

# # Forward pass
# logits, reg_loss = model(input_ids, attention_mask)

# # Compute loss (e.g., CrossEntropyLoss) and add the regularization loss
# criterion = nn.CrossEntropyLoss()
# lm_loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
# total_loss = lm_loss + reg_loss
