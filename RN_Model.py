import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

device = torch.device('cpu')
X = np.load('data/X_cnn_sequences.npy')
y = np.load('data/y_labels_cnn.npy')

embedding_matrix = np.load('data/embedding_matrix.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train = torch.LongTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.LongTensor(X_test)
y_test = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class CNNDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim, matrix_w):
        super(CNNDetector, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(matrix_w))
        self.embedding.weight.requires_grad = False

        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128,64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0,2,1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

vocab_size, embedding_dim = embedding_matrix.shape
model = CNNDetector(vocab_size, embedding_dim, embedding_matrix).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device).view(-1,1)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    preds = model(X_test).cpu().numpy()
    y_pred = (preds > 0.5).astype(int)
    print(classification_report(y_test, y_pred))
torch.save(model.state_dict(), 'detector_ia_cnn.pth')