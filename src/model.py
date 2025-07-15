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
    """
    def __init__(self, input_features, num_classes, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, seq_length):
        super(GripTransformerClassifier, self).__init__()
        self.d_model = d_model
        
        # 1. Input Embedding Layer
        # Projects the 18 input features to the model's internal dimension (d_model)
        self.input_embedding = nn.Linear(input_features, d_model)
        
        # 2. Positional Encoding
        # We add 1 to seq_length because we will prepend a [CLS] token
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_length + 1)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: expects (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 4. CLS (Classification) Token
        # A learnable parameter that will act as an aggregate representation of the sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 5. Classification Head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_features)
        batch_size = x.shape[0]

        # Project input features to d_model
        x = self.input_embedding(x) # -> (batch_size, seq_length, d_model)
        
        # Prepend the [CLS] token to each sequence in the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # -> (batch_size, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1) # -> (batch_size, seq_length + 1, d_model)
        
        # Add positional encoding. Note: PyTorch's PE implementation expects (seq, batch, feat)
        # so we permute, apply, and permute back.
        x = x.permute(1, 0, 2) # -> (seq_length + 1, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2) # -> (batch_size, seq_length + 1, d_model)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(x) # -> (batch_size, seq_length + 1, d_model)
        
        # Extract the output of the [CLS] token (it's the first one)
        cls_output = transformer_output[:, 0, :] # -> (batch_size, d_model)
        
        # Pass the [CLS] token's output through the classifier
        logits = self.classifier(cls_output) # -> (batch_size, num_classes)
        
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