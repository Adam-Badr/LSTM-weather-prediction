import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
from WeatherData import get_weather_data
import matplotlib.pyplot as plt

df = get_weather_data()
print(df.head())

# Pytorch input expectation: (batch_size, sequence_length, num_features)

df_features = df.drop(columns=["date"])
scalar = MinMaxScaler()
scaledArray = scalar.fit_transform(df_features)
# ScaledArray is numHours x 6

timesteps = 48

X, Y = [], []

for i in range(0, len(scaledArray) - timesteps, 1):
    X.append(scaledArray[i:i + timesteps])
    Y.append(scaledArray[i + timesteps])  
X = np.array(X)
Y = np.array(Y)

print(X.shape)  # 840, 48, 6
print(Y.shape)  #840, 6

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
X_Train_Tensor = torch.tensor(X_train, dtype=torch.float32)
X_Test_Tensor = torch.tensor(X_test, dtype=torch.float32)
y_Train_Tensor = torch.tensor(y_train, dtype=torch.float32)
y_Test_Tensor = torch.tensor(y_test, dtype=torch.float32)

train_Dataset = TensorDataset(X_Train_Tensor, y_Train_Tensor)
train_Loader = DataLoader(train_Dataset, batch_size=20, shuffle=True)

#create model

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #Creates the model
        self.fc = nn.Linear(hidden_size, input_size) #TO MAP HDIDEN to input

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters; switched from Python 3.13 to 3.9 so I could use tensorflow- tweak hyperparameters again as model got a little less accurate after the conversion
input_size = 6
hidden_size = 64
num_layers = 2

network = LSTM(input_size, hidden_size, num_layers)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

losses = []
        
#training loop

num_epochs = 40
for epoch in range(num_epochs):
    network.train()

    running_loss = 0.0

    for X_batch, y_batch in train_Loader:
        output = network(X_batch)
        loss = loss_fn(output, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_Loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    losses.append(avg_loss)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show() 



network.eval()
with torch.no_grad():
    predictions = network(X_Test_Tensor)
    test_loss = loss_fn(predictions, y_Test_Tensor)
    print(f"Test Loss: {test_loss.item():.4f}")

variable_names = [
    "Temperature (2m)",
    "Relative Humidity (2m)",
    "Apparent Temperature",
    "Precipitation Probability",
    "Cloud Cover (High)",
    "Wind Direction (80m)"
]

plt.figure(figsize=(15, 10))

for i in range(6):
    plt.subplot(3, 2, i + 1)
    plt.plot(predictions[:, i], label="Predicted")
    plt.plot(y_Test_Tensor[:, i], label="Actual")
    plt.title(variable_names[i])
    plt.xlabel("Time Step")
    plt.ylabel("Scaled Value")
    plt.legend()

plt.tight_layout()
plt.show()

#RENDER REAL TIME PREDICTION

df_live = get_weather_data()
df_live_features = df_live.drop(columns=["date"])
scaled_live = scalar.transform(df_live_features)
print(df_live_features.iloc[-1][0]) #Debugging to see if the temperature is similar, as the API's weather collection differs from e.g. weatherchannel, etc

X_live = scaled_live[-48:]                      
X_live = np.expand_dims(X_live, axis=0)       
X_live_tensor = torch.tensor(X_live, dtype=torch.float32)
network.eval()
with torch.no_grad():
    live_prediction = network(X_live_tensor)    
    pred_real = scalar.inverse_transform(live_prediction.numpy())[0]  
feature_names = [
    "Temperature (2m)",
    "Relative Humidity (2m)",
    "Apparent Temperature",
    "Precipitation Probability",
    "Cloud Cover (High)",
    "Wind Direction (80m)"
]

plt.figure(figsize=(10, 6))
bars = plt.bar(feature_names, pred_real)
plt.title("LSTM-Predicted Current Weather (Based on Past 48 Hours)")
plt.ylabel("Predicted Value")
plt.xticks(rotation=20)
plt.tight_layout()


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1f}', ha='center', va='bottom')

plt.show()









def run_pytorch_model():
    df = get_weather_data()
    df_features = df.drop(columns=["date"])
    scalar = MinMaxScaler()
    scaledArray = scalar.fit_transform(df_features)

    timesteps = 48
    X, Y = [], []

    for i in range(0, len(scaledArray) - timesteps):
        X.append(scaledArray[i:i + timesteps])
        Y.append(scaledArray[i + timesteps])

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    X_Train_Tensor = torch.tensor(X_train, dtype=torch.float32)
    X_Test_Tensor = torch.tensor(X_test, dtype=torch.float32)
    y_Train_Tensor = torch.tensor(y_train, dtype=torch.float32)
    y_Test_Tensor = torch.tensor(y_test, dtype=torch.float32)

    train_Dataset = TensorDataset(X_Train_Tensor, y_Train_Tensor)
    train_Loader = DataLoader(train_Dataset, batch_size=20, shuffle=True)

    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, input_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    input_size = 6
    hidden_size = 64
    num_layers = 2

    network = LSTM(input_size, hidden_size, num_layers)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    for epoch in range(40):
        network.train()
        for X_batch, y_batch in train_Loader:
            output = network(X_batch)
            loss = loss_fn(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    network.eval()
    with torch.no_grad():
        predictions = network(X_Test_Tensor)

    return predictions, y_Test_Tensor.numpy(), scalar



if __name__ == "__main__":
    run_pytorch_model()