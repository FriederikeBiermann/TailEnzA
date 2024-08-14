import torch
import torch.nn as nn
import torch.nn.functional as F


# FFNN Model 1: Basic Feedforward Neural Network
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
        x = F.softmax(x, dim=1)
        return x


# FFNN Model 2: Intermediate Feedforward Neural Network
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
        x = F.softmax(x, dim=1)
        return x


# FFNN Model 3: Advanced Feedforward Neural Network
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
        x = F.softmax(x, dim=1)
        return x


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
        x = F.softmax(x, dim=1)
        return x


# Define the Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self, total_features: int, num_fragments: int, num_classes: int):
        super(CNN, self).__init__()
        self.num_fragments = num_fragments
        self.features_per_fragment = (
            total_features - num_fragments - 1
        ) // num_fragments
        assert (
            total_features - num_fragments - 1
        ) % num_fragments == 0, "Total features do not evenly divide into fragments"

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.2)

        self.fc1_input_dim = (
            64 * (self.num_fragments // 4) * (self.features_per_fragment // 4 + 1)
        )
        self.fc1 = nn.Linear(self.fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        main_features = x[:, : -self.num_fragments]
        fragment_additions = x[:, -self.num_fragments :]

        # Reshape the main features to split them across fragments
        main_features = main_features.view(
            batch_size, self.num_fragments, self.features_per_fragment
        )

        # Add the corresponding fragment-specific features
        x = torch.cat((main_features, fragment_additions.unsqueeze(2)), dim=2)

        # Reshape for CNN input
        x = x.unsqueeze(1)

        # CNN layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)

        # Flatten and fully connected layers
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


# Define the Recurrent Neural Network (RNN)
class RNN(nn.Module):
    def __init__(
        self, in_features: int, hidden_size: int, num_fragments: int, num_classes: int
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_fragments = num_fragments
        self.features_per_fragment = (in_features - num_fragments - 1) // num_fragments
        assert (
            in_features - num_fragments - 1
        ) % num_fragments == 0, "Total features do not evenly divide into fragments"

        self.rnn = nn.RNN(self.features_per_fragment + 1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size = x.size(0)
        main_features = x[:, : -self.num_fragments]
        fragment_additions = x[:, -self.num_fragments :]

        # Reshape the main features to split them across fragments
        main_features = main_features.view(
            batch_size, self.num_fragments, self.features_per_fragment
        )

        # Add the corresponding fragment-specific features
        x = torch.cat((main_features, fragment_additions.unsqueeze(2)), dim=2)

        # RNN layer
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)

        # Fully connected layer
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out


# Define the Long Short-Term Memory (LSTM)
class LSTM(nn.Module):
    def __init__(
        self, in_features: int, hidden_size: int, num_fragments: int, num_classes: int
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_fragments = num_fragments
        self.features_per_fragment = (in_features - num_fragments - 1) // num_fragments
        assert (
            in_features - num_fragments - 1
        ) % num_fragments == 0, "Total features do not evenly divide into fragments"

        self.lstm = nn.LSTM(
            self.features_per_fragment + 1, hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size = x.size(0)
        main_features = x[:, : -self.num_fragments]
        fragment_additions = x[:, -self.num_fragments :]

        # Reshape the main features to split them across fragments
        main_features = main_features.view(
            batch_size, self.num_fragments, self.features_per_fragment
        )

        # Add the corresponding fragment-specific features
        x = torch.cat((main_features, fragment_additions.unsqueeze(2)), dim=2)

        # LSTM layer
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        # Fully connected layer
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out
