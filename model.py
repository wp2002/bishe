import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.5, bidirectional=False):
        super(LSTMModel, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 双向LSTM（可选）
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        # 层归一化（可选）
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        
        # LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 使用层归一化
        lstm_out = self.layer_norm(lstm_out)
        
        # 只取最后一个时刻的输出
        out = self.fc(lstm_out[:, -1, :])
        return out, hidden
