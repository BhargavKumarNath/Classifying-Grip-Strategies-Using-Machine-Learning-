import torch
import torch.nn as nn
import math

# (The LSTMClassifier and PositionalEncoding classes remain the same)
class LSTMClassifier(nn.Module):
    # ... (no changes here) ...
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden_state = h_n[-1]
        out = self.fc(last_hidden_state)
        return out

class PositionalEncoding(nn.Module):
    # ... (no changes here) ...
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GripTransformerClassifier(nn.Module):
    """
    A Transformer-based classifier.
    (Final version with a clean method for returning attention)
    """
    def __init__(self, input_features, num_classes, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, seq_length):
        super(GripTransformerClassifier, self).__init__()
        self.d_model = d_model
        
        self.input_embedding = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_length + 1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, return_attention=False):
        batch_size = x.shape[0]
        x_emb = self.input_embedding(x) # Embed features
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_with_cls = torch.cat((cls_tokens, x_emb), dim=1)
        
        # Permute for positional encoding, which expects (seq, batch, feat)
        x_with_cls = x_with_cls.permute(1, 0, 2)
        x_pos = self.pos_encoder(x_with_cls)
        x_pos = x_pos.permute(1, 0, 2) # Permute back to (batch, seq, feat)

        if not return_attention:
            # Normal forward pass during training/evaluation
            transformer_output = self.transformer_encoder(x_pos)
            cls_output = transformer_output[:, 0, :]
            logits = self.classifier(cls_output)
            return logits
        else:
            # Special forward pass to extract attention from the last layer
            attention_weights = None
            output = x_pos
            # Iterate through all but the last layer normally
            for i in range(self.transformer_encoder.num_layers - 1):
                output = self.transformer_encoder.layers[i](output)
            
            # For the last layer, we call it manually to get weights
            last_layer = self.transformer_encoder.layers[-1]
            
            # This is the key part: we manually call the self-attention block
            # of the final layer and set need_weights=True
            attn_output, attention_weights = last_layer.self_attn(
                last_layer.norm1(output), last_layer.norm1(output), last_layer.norm1(output),
                need_weights=True
            )
            output = output + last_layer.dropout1(attn_output)
            # Finish the rest of the last layer's computations
            output = output + last_layer._ff_block(last_layer.norm2(output))
            
            cls_output = output[:, 0, :]
            logits = self.classifier(cls_output)
            return logits, attention_weights

if __name__ == '__main__':
    BATCH_SIZE = 4
    SEQ_LENGTH = 512
    INPUT_FEATURES = 18
    NUM_CLASSES = 37
    
    # --- Test LSTM ---
    print("--- Testing LSTMClassifier ---")
    lstm_model = LSTMClassifier(input_size=INPUT_FEATURES, hidden_size=128, num_layers=2, num_classes=NUM_CLASSES)
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_FEATURES)
    output = lstm_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Expected: (BATCH_SIZE, NUM_CLASSES)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    print("LSTM test passed!")

    # --- Test Transformer ---
    print("\n--- Testing GripTransformerClassifier ---")
    transformer_model = GripTransformerClassifier(
        input_features=INPUT_FEATURES,
        num_classes=NUM_CLASSES,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        seq_length=SEQ_LENGTH
    )
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_FEATURES)
    output = transformer_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Expected: (BATCH_SIZE, NUM_CLASSES)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    print("Transformer test passed!")