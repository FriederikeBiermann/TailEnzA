import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicFFNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(BasicFFNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class IntermediateFFNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(IntermediateFFNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class AdvancedFFNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(AdvancedFFNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.fc5(x)
        return F.softmax(x, dim=1)


class VeryAdvancedFFNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(VeryAdvancedFFNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.softmax(x, dim=1)


class LSTM(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, num_classes: int):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(in_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) != 3:
            raise RuntimeError(f"Input must have 3 dimensions, got {x.shape}")

        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return F.softmax(out, dim=1)
