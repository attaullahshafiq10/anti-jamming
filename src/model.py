import torch
import torch.nn as nn

class HybridPerceptionModel(nn.Module):
    def __init__(self, num_features, seq_length, num_classes, cnn_f1, cnn_f2, lstm_hidden):
        super(HybridPerceptionModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=cnn_f1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_f1, out_channels=cnn_f2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=cnn_f2, 
            hidden_size=lstm_hidden, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, num_features)
        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_length)
        
        c_out = self.cnn(x)
        c_out = c_out.permute(0, 2, 1)  # (batch_size, seq_length, cnn_f2)
        
        lstm_out, _ = self.lstm(c_out)
        
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        
        return logits
