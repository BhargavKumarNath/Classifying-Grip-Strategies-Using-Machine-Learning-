import torch
import torch.nn as nn
import math
import config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class GripFormer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, num_classes, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.input_embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=config.MAX_SEQ_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.classifier_head = nn.Linear(d_model, num_classes)

    def forward(self, src, src_key_padding_mask):
        src = self.input_embedding(src) * math.sqrt(self.d_model)

        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)

        output = self.transformer_encoder(src, src_key_padding_mask=~src_key_padding_mask)

        output = output * src_key_padding_mask.unsqueeze(-1)
        pooled_output = output.sum(dim=1) / src_key_padding_mask.sum(dim=1).unsqueeze(-1)

        logits = self.classifier_head(pooled_output)
        return logits

