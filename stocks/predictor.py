import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, Conv1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import root_mean_squared_error
import holidays

# FEATURE ENGINEERING
# Load and prepare data
data = pd.read_csv("./SBUX.csv")
data = data.iloc[::-1]
data = data.reset_index(drop=True)
data['Date'] = pd.to_datetime(data['Date'])
data.rename(columns={'Price': 'Close'}, inplace=True)
data['Vol.'] = data['Vol.'].str.replace('M', '').astype(float)

# Calculate Technical Indicators
data['SMA_20'] = data['Close'].rolling(window=20).mean()
delta = data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()
rs = avg_gain / avg_loss
data['RSI_14'] = 100 - (100 / (1 + rs))
ema_fast = data['Close'].ewm(span=12, adjust=False).mean()
ema_slow = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_fast - ema_slow

# Add Time-Based & Holiday Features
data['dayofweek'] = data['Date'].dt.dayofweek
data['month'] = data['Date'].dt.month
data['quarter'] = data['Date'].dt.quarter
us_holidays = holidays.US()
data['Is_Holiday'] = data['Date'].apply(lambda date: 1 if date in us_holidays else 0)

data.dropna(inplace=True)
data = data.reset_index(drop=True)

features = [
    'Open', 'High', 'Low', 'Close', 'Vol.',
    'SMA_20', 'RSI_14', 'MACD',
    'dayofweek', 'month', 'quarter', 'Is_Holiday'
]
num_features = len(features)
feature_data = data[features].values

scaler = StandardScaler()
data_scaled = scaler.fit_transform(feature_data)
targets = data_scaled[:, 0]

# HYPERPARAMETER AND DATASET SETUP
time_step = 21
split_ratio = 0.8
split_index = round(len(data_scaled) * split_ratio)
batch_size = 32

train_dataset = tf.keras.utils.timeseries_dataset_from_array(
    data=data_scaled, targets=targets, sequence_length=time_step,
    batch_size=batch_size, end_index=split_index
)
validation_dataset = tf.keras.utils.timeseries_dataset_from_array(
    data=data_scaled, targets=targets, sequence_length=time_step,
    batch_size=batch_size, start_index=split_index
)

## MODEL TRAINING OR LOADING
MODEL_FILENAME = "sbux_predictor.keras"
RETRAIN = False

if os.path.exists(MODEL_FILENAME) and not RETRAIN:
    print("Loading existing model from disk...")
    regressor = tf.keras.models.load_model(MODEL_FILENAME)
else:
    print("No existing model found. Training a new model...")
    regressor = Sequential()
    regressor.add(Input(shape=(time_step, num_features)))
    regressor.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    regressor.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    regressor.add(Dropout(0.1))
    regressor.add(Bidirectional(LSTM(units=64)))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(units=1))

    optimizer = Adam(learning_rate=0.0005)
    regressor.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = regressor.fit(
        train_dataset, epochs=100, validation_data=validation_dataset, callbacks=[early_stopping]
    )
    print(f"Model training complete. Saving model as {MODEL_FILENAME}...")
    regressor.save(MODEL_FILENAME)

    ## PREDICTIONS AND EVALUATION
    y_test_actual_scaled = np.concatenate([y for x, y in validation_dataset], axis=0)
    y_pred_scaled = regressor.predict(validation_dataset)

    dummy_pred = np.zeros((len(y_pred_scaled), num_features))
    dummy_pred[:, 0] = y_pred_scaled.flatten()
    y_pred_of_test = scaler.inverse_transform(dummy_pred)[:, 0]

    dummy_actual = np.zeros((len(y_test_actual_scaled), num_features))
    dummy_actual[:, 0] = y_test_actual_scaled.flatten()
    y_test_actual = scaler.inverse_transform(dummy_actual)[:, 0]

    plt.figure(figsize=(15, 6))
    plt.plot(y_pred_of_test, label="Predicted Price", c="orange")
    plt.plot(y_test_actual, label="Actual Price", c="g", alpha=0.7)
    plt.title("CNN-LSTM Model with Time-Based Features")
    plt.legend()
    plt.show()

    rmse = root_mean_squared_error(y_test_actual, y_pred_of_test)
    mape = np.mean(np.abs((y_test_actual - y_pred_of_test) / y_test_actual)) * 100

    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")

regressor.summary()

# FORECASTING FUTURE STOCK PRICES
print("\nForecasting the next 5 days...")
last_sequence = data_scaled[-time_step:]
current_sequence = np.reshape(last_sequence, (1, time_step, num_features))

future_predictions_scaled = []
forecast_days = 5

for _ in range(forecast_days):
    next_pred_scaled = regressor.predict(current_sequence, verbose=0)
    future_predictions_scaled.append(next_pred_scaled[0, 0])
    new_row = np.full((1, num_features), next_pred_scaled.flatten()[0])
    new_sequence = np.append(current_sequence[:, 1:, :], [new_row], axis=1)
    current_sequence = new_sequence

dummy_future = np.zeros((len(future_predictions_scaled), num_features))
dummy_future[:, 0] = future_predictions_scaled
future_predictions = scaler.inverse_transform(dummy_future)[:, 0]

print("\n--- Forecast for the Next 10 Days")
for i, price in enumerate(future_predictions):
    print(f"Day {i+1}: ${price:.2f}")

# PLOTTING THE FORECAST
last_date = data['Date'].iloc[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
historical_data = data[['Date', 'Open']].iloc[-60:]

plt.figure(figsize=(15, 6))
plt.plot(historical_data['Date'], historical_data['Open'], label="Actual Price", color="g")
plt.plot(forecast_dates, future_predictions, label="Forecasted Price", color="red", marker='o')
plt.title("SBUX Stock Price: Historical and 10-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Open Price (USD)")
plt.legend()
plt.grid(True)
plt.show()
