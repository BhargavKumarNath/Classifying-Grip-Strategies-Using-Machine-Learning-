import torch
import torch.nn as nn
import math

class LSTMClassifier(nn.Module):
    """
    A simple but effective LSTM-based classifier for sequence data.
    """
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
    """
    Injects positional information into the input embeddings.
    Standard implementation from the "Attention is All You Need" paper.
    """
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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class GripTransformerClassifier(nn.Module):
    """
    A Transformer-based classifier using an encoder architecture.
    (This is the updated version)
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
        x = self.input_embedding(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)

        attention_weights = None
        for i, layer in enumerate(self.transformer_encoder.layers):
            if i == len(self.transformer_encoder.layers) - 1 and return_attention:
                x, attention_weights = layer.self_attn(x, x, x, need_weights=True)
                x = layer.dropout1(x)
                x = layer.norm1(x) 
                
                x_main = layer(x) 
                x = x_main 
            else:
                 x = layer(x)
        
        if not return_attention:
            transformer_output = x
        else: 
            transformer_output = x

        cls_output = transformer_output[:, 0, :]
        logits = self.classifier(cls_output)
        
        if return_attention:
            return logits, attention_weights
        
        return logits

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