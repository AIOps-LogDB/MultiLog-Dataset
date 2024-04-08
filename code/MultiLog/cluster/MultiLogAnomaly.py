import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from dis_common import get_data_and_time_label
from sklearn.preprocessing import StandardScaler

case_list = ["multi_anomaly_multi_node", "multi_anomaly_single_node"]
node_num = 6

input_dim = 128
output_dim = 32
batch_size = 10
epochs = 100


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def preprocess_data_for_AE(data, input_dim):
    processed_data = []
    for time_segment_data in data:
        for segment in time_segment_data:
            if len(segment) > input_dim:
                segment = segment[:input_dim]
            else:
                segment = np.pad(segment, (0, max(0, input_dim - len(segment))), 'constant')
            processed_data.append(segment)
    return np.array(processed_data)


def preprocess_data(data, input_dim):
    processed_data = []
    for segment in data:
        if len(segment) > input_dim:
            segment = segment[:input_dim]
        else:
            segment = np.pad(segment, (0, max(0, input_dim - len(segment))), 'constant')
        processed_data.append(segment)
    return np.array(processed_data)


def extract_features(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for data in data_loader:
            features.append(model.encoder(data))
    return torch.cat(features).numpy()

MODEL = sys.argv[1]

for case in case_list:
    data, time_label = get_data_and_time_label(f"result_time_predict_{MODEL}_{case}.log")

    X_AE = preprocess_data_for_AE(data, input_dim)

    autoencoder = Autoencoder(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    X_AE_tensor = torch.tensor(X_AE, dtype=torch.float32)

    dataset = TensorDataset(X_AE_tensor, X_AE_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 50
    for epoch in range(num_epochs):
        for curr_data in dataloader:
            inputs, targets = curr_data
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    train_datasets = []
    test_datasets = time_label
    for i in range(len(time_label)):
        X = preprocess_data(data[i], input_dim)
        train_datasets.append(extract_features(autoencoder, torch.FloatTensor(X)))

    X_train, X_test, y_train, y_test = train_test_split(train_datasets, test_datasets, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    meta_classifier = LogisticRegression(max_iter=1000, solver='saga')
    meta_classifier.fit(X_train, y_train)
    predictions = meta_classifier.predict(X_test)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    print(f"{case}:[{precision},{recall},{f1}]")
