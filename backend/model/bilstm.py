import torch
import torch.nn as nn
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super().__init__()

        self.num_classes = num_classes
        self.is_binary = (num_classes == 2)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        out_dim = 1 if self.is_binary else num_classes
        self.fc = nn.Linear(hidden_size * 2, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)   # logits
