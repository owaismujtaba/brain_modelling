import torch 
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual feedforward block: Linear -> ReLU -> BN -> Dropout -> Linear + skip"""
    def __init__(self, dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, D] or [B*T, D]
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x)
        return F.relu(x + residual)


class GRUDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._setup_conf_params()
        
        # Dropout
        self.input_dropout = nn.Dropout(self.input_dropout)

        # Day embeddings (additive)
        self.day_embed = nn.Embedding(self.n_days, self.neural_dim)

        # Input dimension adjustment for patching
        self.input_size = self.neural_dim * (self.patch_size if self.patch_size > 0 else 1)

        # GRU backbone
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout if self.n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        out_dim = self.n_units * (2 if self.gru.bidirectional else 1)

        # Multi-head self-attention on GRU outputs
        self.attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=self.attn_heads,
            dropout=self.attn_dropout,
            batch_first=True,
        )

        # Residual FC head (stacked)
        self.fc_blocks = nn.Sequential(*[
            ResidualBlock(out_dim, self.hidden_fc, dropout=0.3)
            for _ in range(self.n_resblocks)
        ])

        # Final classification
        self.out = nn.Linear(out_dim, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(torch.empty(self.n_layers * (2 if self.gru.bidirectional else 1), 1, self.n_units))
        nn.init.xavier_uniform_(self.h0)

        self.norm = nn.LayerNorm(self.input_size)

    def _setup_conf_params(self):
        model_conf = self.config['model']
        self.name = model_conf['name']
        self.neural_dim = model_conf['neural_dim']
        self.n_units = model_conf['n_units']
        self.n_layers = model_conf['n_layers']
        self.max_seq_elements = model_conf['max_seq_elements']
        self.input_dropout = model_conf['input_dropout']
        self.rnn_dropout = model_conf['rnn_dropout']
        self.patch_size = model_conf['patch_size']
        self.patch_stride = model_conf['patch_stride']
        self.n_days = model_conf['n_days']
        self.n_classes = model_conf['n_classes']
        
        self.hidden_fc= 256
        self.n_resblocks=2
        self.attn_heads=4
        self.attn_dropout=0.1
        self.sequence_output = True

    def forward(self, x, day_idx, states=None, return_state=False):
        """
        x: [B, T, D]
        day_idx: [B] (long tensor with day indices)
        """
        B, T, D = x.shape

        # Add day embeddings
        day_vecs = self.day_embed(day_idx)  # [B, D]
        x = x + day_vecs.unsqueeze(1)

        x = self.input_dropout(x)

        # Optional patching
        if self.patch_size > 0:
            x = x.permute(0, 2, 1)  # [B, D, T]
            patches = x.unfold(2, self.patch_size, self.patch_stride)  # [B, D, num_patches, patch_size]
            x = patches.permute(0, 2, 3, 1).reshape(B, -1, self.input_size)  # [B, num_patches, input_size]

        x = self.norm(x)

        # Init hidden state
        if states is None:
            states = self.h0.expand(-1, B, -1).contiguous()

        # GRU
        output, hidden_states = self.gru(x, states)  # [B, T, H]

        # Self-attention (contextual refinement)
        attn_out, _ = self.attn(output, output, output)  # [B, T, H]
        output = output + attn_out  # residual connection

        # Sequence or pooled output
        if self.sequence_output:
            out_fc = self.fc_blocks(output.reshape(-1, output.size(-1)))  # [B*T, H]
            logits = self.out(out_fc).view(B, -1, self.n_classes)  # [B, T, C]
        else:
            pooled = output.mean(dim=1)  # [B, H]
            out_fc = self.fc_blocks(pooled)
            logits = self.out(out_fc)  # [B, C]

        if return_state:
            return logits, hidden_states
        return logits