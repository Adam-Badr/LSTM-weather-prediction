import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from WeatherData import get_weather_data
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

df = get_weather_data()
print(df.head())

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

timesteps = 48
features = 6
output_dim = 6

model = Sequential([
    LSTM(
        units=64,
        activation='tanh',           
        recurrent_activation='sigmoid', 
        return_sequences=True,     
        input_shape=(timesteps, features)
    ),
    LSTM(
        units=64,
        activation='tanh',           
        recurrent_activation='sigmoid', 
        return_sequences=False,
    ),
    Dense(units=output_dim)
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=40, batch_size=20, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print("Test loss:", loss)
predictions = model.predict(X_test)
X_live = scaledArray[-timesteps:]
X_live = np.expand_dims(X_live, axis=0)
live_prediction = model.predict(X_live)
pred_real = scalar.inverse_transform(live_prediction)[0]

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()

X_live = scaledArray[-timesteps:]
X_live = np.expand_dims(X_live, axis=0)
live_prediction = model.predict(X_live)
pred_real = scalar.inverse_transform(live_prediction)[0]

# Label predictions
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
plt.title(" Predicted Current Weather (LSTM)")
plt.ylabel("Predicted Value")
plt.xticks(rotation=20)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

actual_current = df_features.iloc[-1].values  
print("\n Actual Current Weather (from API):")
for name, value in zip(feature_names, actual_current):
    print(f"{name}: {value:.2f}")