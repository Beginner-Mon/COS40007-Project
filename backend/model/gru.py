import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.0):
        super().__init__()

        self.num_classes = num_classes
        self.is_binary = (num_classes == 2)

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout
        )

        self.fc1 = nn.Linear(hidden_size, 128) 
        self.fc2 = nn.Linear(128, 128)
        
        out_dim = 1 if self.is_binary else num_classes
        self.fc_out = nn.Linear(128, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]                     

        # Block 1: FC(128) → Dropout → ReLU
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)

        # Block 2: FC(128) → Dropout → ReLU
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.fc_out(out)
        return out